from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, cubic_sampling, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion, cubic_sampling)
    elif name == "loss-second-moment":
        # 二阶动量平滑loss
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()  # [1, 1, 1, 1, 1, 1 , ...T..., 1]   1000个1
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)  # 从[1，2，3.。。T]中。在概率为p的情况下抽取batch_size个数字
        # 0-999 大家概率都为p 选取batch_size个数字出来
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion,cubic_sampling):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])
        self.cubic_sampling = cubic_sampling
    def weights(self):
        pass
        return self._weights

    def sample(self, batch_size, device):
        w = self._weights
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)

        # cubic sampling for training
        if self.cubic_sampling:
            indices_np = np.array([int((1 - (x / self.diffusion.num_timesteps) ** 3) * self.diffusion.num_timesteps) for x in indices_np])

        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)

        return indices, weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        # history_per_term: 每个时间步存储的损失历史长度。这表示对于每个时间步，我们要保留最近多少个损失值来计算二阶矩
        # uniform_prob: 一个均匀分布概率，用于在最终权重中混入一些均匀分布的随机性
        # _loss_history: 一个二维数组，用于存储每个时间步的损失历史
        # _loss_counts: 一个一维数组，用于记录每个时间步已经存储了多少个损失值
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))    # 计算每个时间步的损失历史的二阶矩(平方的平均值)
        weights /= np.sum(weights)      # 归一化
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        # 根据uniform_prob混入一些均匀分布的权重，这是为了确保每个时间步都有一定几率被选中，即使它们的损失很小
        return weights

    def update_with_all_losses(self, ts, losses):       # 每个时间步，对应的损失值数组
        for t, loss in zip(ts, losses):
            # 对于每一对时间步和损失值，如果该时间步的损失历史已经满了，就移除最旧的损失值，并将新的损失值添加到末尾。
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                # 如果损失历史还没有满，就在当前位置添加新的损失值，并更新已存储的损失数量。
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        # 检查每个时间步的损失历史是否都已经填充完毕。如果都填充完毕, 返回True, 否则返回False
        return (self._loss_counts == self.history_per_term).all()
