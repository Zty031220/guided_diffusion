"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision.utils import make_grid, save_image
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    create_arcface_model,
    arcface_defaults,
    cosin_metric
)
from resizer import Resizer

def main():
    args = create_argparser().parse_args()
    print(f"args: {args}")
    dist_util.setup_dist()
    if args.classifier_path != '':
        logger.configure('./logs/classifier_sample')
    else:
        logger.configure('./logs/arcface_sample')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.classifier_path != '':
        logger.log("loading classifier...")
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        # print(f"model_dict1.eln: {len(model_dict1)}")
        # print(f"model_dict2.len: {len(model_dict2)}")
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.to(dist_util.dev())
        if args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

    arcface_model = None
    if args.arcface_path != '':
        logger.log("loading arcface model...")
        arcface_model = create_arcface_model(**args_to_dict(args, arcface_defaults().keys()))

        # arcface_model.load_state_dict(
        #     dist_util.load_state_dict(args.arcface_path, map_location="cpu")
        # )
        arcface_model = DataParallel(arcface_model,  device_ids=[0, 2])
        arcface_model.load_state_dict(torch.load(args.arcface_path))
        arcface_model.to(dist_util.dev())
        arcface_model.eval()

    if args.data_dir != '':         # arcface 人脸数据
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            random_crop=True,
        )
        data = next(data)[0].to(dist_util.dev())

    if args.y_data_dir != '':           # ILVR
        y_data = load_data(
            data_dir=args.y_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            random_crop=True,
        )
        y_data = next(y_data)[0].to(dist_util.dev())
    # output1 = model(x_t)
    # output2 = model(x_0)
    # def cosin_metric(x1, x2):
    #     return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    # logits = 1 - cosin_metric(output1, output2)
    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    def arcface_fn(x, x_t, x_0):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            new_x_0, new_x_t = torch.narrow(x_0,1,0,1), torch.narrow(x_t,1,0,1)
            # x_0_output, x_t_output = arcface_model(x_0).data.cpu().numpy(), arcface_model(x_in).data.cpu().numpy()
            x_0_output, x_t_output = arcface_model(new_x_0).data, arcface_model(new_x_t).data
            print(f"x_0_output.shape: {x_0_output.shape}, x_t_output.shape:{x_t_output.shape}")   # [8, 512]
            # x_0_output, x_t_output = np.hstack((x_0_output[::2], x_0_output[1::2])), np.hstack((x_t_output[::2], x_t_output[1::2]))
            # x_0_output, x_t_output = np.hstack((x_0_output[::2], x_0_output[1::2])), np.hstack((x_t_output[::2], x_t_output[1::2]))
            # x_0_output, x_t_output = np.hstack((x_0_output[::2], x_0_output[1::2])), np.hstack((x_t_output[::2], x_t_output[1::2]))
            # x_0_output, x_t_output = np.hstack((x_0_output[::2], x_0_output[1::2])), np.hstack((x_t_output[::2], x_t_output[1::2]))
            print(f"x_0_output: {x_0_output.shape}, x_t_output: {x_t_output.shape}")
            # logits = torch.from_numpy(cosin_metric(x_0_output, x_t_output))
            logits = 1 - torch.cosine_similarity(x_0_output, x_t_output, dim=1, eps=1e-08)
            selected = logits
            selected.requires_grad_(True)
            print(logits)
            print(f"logits.size(): {logits.size()}")
            print(f"{len(logits)}")
            print(f"logits.sum(): {logits.sum()}")
            # log_probs = F.log_softmax(logits, dim=-1)
            # selected = log_probs[range(len(logits)), logits.view(-1)]
            # print(th.autograd.grad(logits.sum(), x_in))
            gradient = th.autograd.grad(selected.mean(), x_in, allow_unused=True)[0]
            if gradient is None:
                gradient = torch.zeros_like(x_in)
            # return th.autograd.grad(selected.sum(), x_in, allow_unused=True)[0] * args.classifier_scale        # 对数似然的梯度 * scale
            return gradient * args.classifier_scale        # 对数似然的梯度 * scale
            # return logits.sum()        # 对数似然的梯度 * scale

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            print(f"y: {y}")        # [10]
            x_in = x.detach().requires_grad_(True)
            print(f"x_in.size(): {x_in.size()}")
            logits = classifier(x_in, t)
            print(f"logits:{logits}")
            print(f"logits.size():{logits.size()}")     # [16, 10]
            print(f"len(logits): {len(logits)}")        # 16
            log_probs = F.log_softmax(logits, dim=-1)
            print(f"{log_probs}")
            print(f"{log_probs.size()}")        # [16, 10]
            print(f"{y.view(-1)}")
            selected = log_probs[range(len(logits)), y.view(-1)]        # 选择了每一个真实标签所对应的预测值来作为梯度
            print(f"selected:{selected}")
            print(f"selected.size():{selected.size()}")     # [16]
            print(f"{selected.sum()}")                      # 标量值
            # print(f"th.autograd.grad(selected.sum(), x_in)[0]: {th.autograd.grad(selected.sum(), x_in)[0]}")
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale        # 对数似然的梯度 * scale

    def model_fn(x, t, y=None):
        # assert y is not None
        return model(x, None, t, y if args.class_cond else None)

    logger.log("sampling...")
    # all_images = []
    # all_labels = []
    # while len(all_images) * args.batch_size < args.num_samples:
    model_kwargs = {}
    if args.data_dir != '':
        model_kwargs["x_0"] = data
    if args.classifier_path != '':
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        print(f"classes: {classes}")

    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    sample = sample_fn(
        model,
        arcface_model,
        (args.batch_size, 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn if args.classifier_path != '' else None,
        arcface_fn=arcface_fn if args.arcface_path != '' else None,
        device=dist_util.dev(),
        up=up,
        down=down,
        y_0=y_data if args.y_data_dir != '' else None
    )
    if args.data_dir != '':
        sample = torch.cat((data, sample))
    if args.y_data_dir != '':
        sample = torch.cat((y_data, sample))
    # sample = torch.cat((sample, data))
    grid = (make_grid(sample) + 1) / 2
    if args.classifier_path != '':
        output_path = './outputs/tlg10_classifier480000_model620000_sample.png'
    elif args.arcface_path != '':
        output_path = './outputs/train_arcface_model230000_sample.png'
    else:
        output_path = './outputs/test.png'
    save_image(grid, output_path)
    #     sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    #     sample = sample.permute(0, 2, 3, 1)
    #     sample = sample.contiguous()
    #
    #     gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    #     dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
    #     all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    #     gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
    #     dist.all_gather(gathered_labels, classes)
    #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
    #     logger.log(f"created {len(all_images) * args.batch_size} samples")
    #
    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # label_arr = np.concatenate(all_labels, axis=0)
    # label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=8,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        arcface_path="",
        data_dir="./imgs/VGGface2/n000004",
        # data_dir="",
        y_data_dir="",
        down_N=32,
    )
    # image_size = 128,
    # num_channels = 128,
    # num_res_blocks = 2,
    # num_heads = 4,
    # num_heads_upsample = -1,
    # num_head_channels = -1,
    # attention_resolutions = "16,8",
    # channel_mult = "",
    # dropout = 0.0,
    # class_cond = False,
    # use_checkpoint = False,
    # use_scale_shift_norm = True,
    # resblock_updown = False,
    # use_fp16 = False,
    # use_new_attention_order = False,
    # learn_sigma = False,
    # diffusion_steps = 1000,
    # noise_schedule = "linear",
    # timestep_respacing = "",
    # use_kl = False,
    # predict_xstart = False,
    # rescale_timesteps = False,
    # rescale_learned_sigmas = False,
    # image_size = 64,

    # classifier_use_fp16 = False,
    # classifier_width = 128,
    # classifier_depth = 2,
    # classifier_attention_resolutions = "32,16,8",  # 16
    # classifier_use_scale_shift_norm = True,  # False
    # classifier_resblock_updown = True,  # False
    # classifier_pool = "attention",

    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults.update(arcface_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    # print(defaults)
    # print(parser)
    return parser


if __name__ == "__main__":
    main()
