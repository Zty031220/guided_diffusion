"""
Train a diffusion model on images.
"""

import argparse
import sys

from augmentations import ImageAugmentations
from torch.nn import DataParallel
import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    create_arcface_model,
    arcface_defaults,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader

from guided_diffusion.arcface_torch.backbones import get_model
from guided_diffusion.arcface_torch.utils.utils_config import get_config

import pickle
import os
# from face_parser.model import BiSeNet
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    assert args.save_dir != ""

    logger.configure(f'./logs/{args.save_dir}')

    # image_augmentations = ImageAugmentations(112, 8)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log(f"timestep_respacing: {args.timestep_respacing}")
    if args.resume_checkpoint != '':
        state_dict = dist_util.load_state_dict(args.resume_checkpoint, map_location="cpu")
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'module' in key:
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(torch.load(args.resume_checkpoint))
        # model.load_state_dict(
        #     dist_util.load_state_dict(args.resume_checkpoint, map_location="cpu")
        # )
    model = DataParallel(model, device_ids=[4, 5], output_device=4)
    # model = DataParallel(model, device_ids=[7], output_device=7)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, args.cubic_sampling, diffusion)

    logger.log("loading arcface model...")

    cfg = get_config(f'guided_diffusion/arcface_torch/configs/{args.face_dataset}_{args.face_model}.py')
    arcface_model = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    ckpt_path = f'checkpoints/{args.face_dataset}_{args.face_model}.pth'
    arcface_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    arcface_model = arcface_model.to(dist_util.dev()).eval()

    # face_parser_model = BiSeNet(n_classes=19)
    # face_parser_model.load_state_dict(torch.load('checkpoints/79999_iter.pth', map_location="cpu"))
    # face_parser_model.eval()

    landmark_pth = './imgs/ffhq/landmarks'
    landmarks = pickle.load(open(os.path.join(landmark_pth, f'landmark_ori_256.pkl'), 'rb'))
    landmark_issue_txt = './imgs/ffhq/landmarks/issues_images256.txt'
    landmarks_issue = []
    if os.path.exists(landmark_issue_txt):
        with open(landmark_issue_txt, 'r') as file:
            data = file.read()
        lines = data.split('\n')
        landmarks_issue = [line.split(',')[0] for line in lines]

    logger.log("creating data loader...")
    data_target = load_data(
        data_dir=args.data_source_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        # arface_model=arcface_model,
        landmark=landmarks,
        landmarks_issue=landmarks_issue
    )
    logger.log("training...")
    TrainLoop(
        model=model,
        arcface_model=arcface_model,
        # face_parser_model=face_parser_model,
        diffusion=diffusion,
        data=data_target,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,                             # 1e-4
        ema_rate=args.ema_rate,                 # 0.9999
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,  # UniformSampler(diffusion)
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        image_augmentations=None,
        drop_rate=args.drop_rate,
        train_mode=args.train_mode,
        save_dir=args.save_dir,
        use_P2_weight=args.use_P2_weight
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_source_dir="",
        data_target_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        # diffswap=2e-6     diffface=0.0001
        weight_decay=0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        face_dataset='glint360k',
        face_model='r100',
        drop_rate=0.2,
        train_mode="train",     # train, finetune
        save_dir="",
        use_P2_weight=False,
        cubic_sampling=False,
        isSample=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(arcface_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
