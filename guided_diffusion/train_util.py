import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.nn.functional as F

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from torch.utils.tensorboard import SummaryWriter

import random

# from scripts.AutomaticWeightedLoss import AutomaticWeightedLoss

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
writer = None

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        arcface_model,
        # face_parser_model,
        diffusion,
        data,
        # data_target,
        batch_size,
        microbatch,
        lr,
        ema_rate,   # "0.9999"
        log_interval,   # 10
        save_interval,  # 10000
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,  # UniformSampler(diffusion)
        weight_decay=0.0,
        lr_anneal_steps=0,
        image_augmentations,
        drop_rate=0.0,
        train_mode="train",
        save_dir="",
        use_P2_weight=False
    ):
        self.model = model
        self.arcface_model = arcface_model
        # self.face_parser_model = face_parser_model
        self.diffusion = diffusion
        self.data = data
        # self.data_target = data_target
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        # self.awl = AutomaticWeightedLoss(2)
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion, False)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.image_augmentations = image_augmentations

        self.drop_rate = drop_rate

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()     # batch_size

        self.train_mode = train_mode
        self.save_dir = save_dir
        self.use_P2_weight = use_P2_weight
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.transform = transforms.Grayscale(num_output_channels=1)
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        # self.opt = AdamW([
        #     {'params': self.mp_trainer.master_params, 'lr': self.lr, 'weight_decay': self.weight_decay},
        #     {'params': self.awl.parameters(), 'weight_decay': 0},
        #     ])
        if self.resume_step:    # 0
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            print('cuda is available')
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            print("cuda is not available")
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint      # find_resume_checkpoint(): None ./models/model020000.pt

        if resume_checkpoint:   # ./models/model020000.pt
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)   # 20000
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint        # find_resume_checkpoint(): None ./models/model020000.pt
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )       # ./models/opt006.pt
        if bf.exists(opt_checkpoint):       # false
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        global writer
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            try:
                batch, src_cond = next(self.data)
                # batch_target, batch_source, src_cond = next(self.data_source)
                # batch, cond_target = next(self.data_target)
            except Exception as e:
                print(e)
                print('run_loop except')
                return
            if self.train_mode == "finetune":
                raise NotImplementedError("we change the train strategy, "
                                          "now the inputs are batch_source and batch_target")
                random_numbers = [random.random() for _ in range(self.batch_size)]
                positive_batch_index = [idx for idx,x in enumerate(random_numbers) if x > self.drop_rate]
                negative_batch_index = list(set(range(self.batch_size)) - set(positive_batch_index))

                if len(positive_batch_index) != 0:
                    batch_source_positive = batch_source.index_select(0, th.tensor(positive_batch_index)).to(dist_util.dev())
                    src_cond_postitive = {}
                    src_cond_postitive["landmarks"] = src_cond["landmarks"].index_select(0, th.tensor(positive_batch_index))
                    src_cond_postitive["landmarks"] = src_cond_postitive["landmarks"].reshape(src_cond_postitive["landmarks"].shape[0], -1).float()
                    src_cond_postitive["id"] = self.arcface_model(F.interpolate(batch_source_positive, (112, 112), mode='bicubic'))

                    self.run_step(batch_source_positive, src_cond_postitive, self.step)
                if len(negative_batch_index) != 0:
                    batch_source_negative = batch_source.index_select(0, th.tensor(negative_batch_index)).to((dist_util.dev()))
                    src_cond_negative = {}
                    src_cond_negative["id"] = th.zeros(batch_source_negative.shape[0], 512)
                    src_cond_negative["landmarks"] = th.zeros(batch_source_negative.shape[0], 68 * 2)

                    self.run_step(batch_source_negative, src_cond_negative, self.step)

            elif self.train_mode == "train":
                src_cond["landmarks"] = src_cond["landmarks"].reshape(src_cond["landmarks"].shape[0], -1).float()
                # src_cond["landmarks"] = src_cond["landmarks"] * 2.0 - 1.0
                with th.no_grad():

                    src_cond["id"] = self.arcface_model(F.interpolate(batch.to(dist_util.dev()), (112, 112), mode='bicubic'))
                    # src_cond["face_parser"] = self.face_parser_model(face_parser_img)[0]
                # max_landmark, min_landmark = th.max(src_cond["landmarks"]), th.min(src_cond["landmarks"])
                # max_id, min_id = th.max(src_cond["id"]), th.min(src_cond["id"])
                # # min-max标准化到[-1, 1]
                # src_cond["landmarks"] = 2.0 * th.div(src_cond["landmarks"] - min_landmark, max_landmark - min_landmark) - 1.0
                # src_cond["id"] = 2.0 * th.div(src_cond["id"] - min_id, max_id - min_id) - 1.0

                self.run_step(batch, src_cond, self.step)
            else:
                raise NotImplementedError(f"unknown train_mode's value: {self.train_mode} "
                                          f"which should be 'train' or 'finetune'")

            if self.step % self.log_interval == 0:      # log_interval=10
                logger.dumpkvs()
            if self.step % self.save_interval == 0:     # save_interval
                self.save()
                if writer is not None:
                    writer.close()
                writer = SummaryWriter(f"logs/{self.save_dir}_event")

                # # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    # def run_step(self, batch, cond_target, batch_source):
    def run_step(self, batch, src_cond, step):

        # self.forward_backward(batch, cond_target, batch_source)

        self.forward_backward(batch, src_cond, step)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()


    # def forward_backward(self, batch, cond_target, batch_source):
    def forward_backward(self, batch, src_cond, step):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], batch.shape[0]):
            try:
                self.microbatch = batch.shape[0]
                micro_target = batch[i : i + self.microbatch].to(dist_util.dev())  # batch[:batchsize]
                micro_cond = {
                    k: v[i : i + self.microbatch].to(dist_util.dev())
                    for k, v in src_cond.items()
                }
                micro_cond["p2weight"] = self.use_P2_weight
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro_target.shape[0], dist_util.dev())      # UniformSampler(diffusion)
                # print(f"t: {t}, weights: {weights}")
                # [12, 25, 222, 666, 888, 123]  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

                compute_losses = functools.partial(     # training_losses(self.ddp_model, micro, t, model_kwargs=micro_cond)
                    self.diffusion.training_losses,
                    self.ddp_model,
                    self.arcface_model,
                    micro_target,
                    t,
                    mp_trainer=self.mp_trainer,
                    # self.image_augmentations,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):     # UniformSampler(diffusion)
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                if th.all(micro_cond["id"] == 0):
                    loss_mse = (losses["loss_noid"] * weights).mean()
                    loss_id = 0
                else:
                    loss_mse = (losses["loss"] * weights).mean()
                    losses["id"] = losses["id"] * 0.01
                    loss_id = (losses["id"]).mean()

                if writer is not None:
                    if th.all(micro_cond["id"] == 0):
                        writer.add_scalar("mse_loss_noid", loss_mse, step + self.resume_step)
                    else:
                        # loss_mse_xt = losses["mse_xt"].mean()
                        # loss_mse_xmid = losses["mse_mid"].mean()
                        writer.add_scalar("id_loss", loss_id, step + self.resume_step)
                        writer.add_scalar("mse_loss", loss_mse, step + self.resume_step)
                        # writer.add_scalar("mse_xt", loss_mse_xt, step + self.resume_step)
                        # writer.add_scalar("mse_xmid", loss_mse_xmid, step + self.resume_step)
                loss = loss_mse + loss_id
                # loss = self.awl(loss_mse, loss_id)
                # losses["mse_weight"] = self.awl.params[0]
                # losses["id_weight"] = self.awl.params[1]
                # print(f"loss: {loss}")
                # print(f"type(loss): {type(loss)}")
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )
                self.mp_trainer.backward(loss)
            except Exception as e:
                print(e)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):       # ./models/model020000.pt
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")     # ['./', 's/', '020000.pt']
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]    # '020000'
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
