"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from torch.nn import DataParallel
import numpy as np
import torch
import torch as th
import torch.distributed as dist
from torchvision.utils import make_grid, save_image
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_arcface_model,
    arcface_defaults
)
import torch.nn.functional as F
from guided_diffusion.image_datasets import load_data
from resizer import Resizer
from torchvision import transforms
from module import SpecificNorm
# from models.parsing import BiSeNet
from guided_diffusion.data.portrait import Portrait
from tqdm import tqdm

from utils.blending.blending_mask import gaussian_pyramid, laplacian_pyramid, laplacian_pyr_join, laplacian_collapse
import cv2
import matplotlib.pyplot as plt
import matplotlib

from guided_diffusion.arcface_torch.backbones import get_model
from guided_diffusion.arcface_torch.utils.utils_config import get_config

import pickle

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(f'./logs/{args.output_path_dir}')

    transform = transforms.Grayscale(num_output_channels=1)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
    model.load_state_dict(
        new_state_dict
    )
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    cfg = get_config(f'guided_diffusion/arcface_torch/configs/{args.face_dataset}_{args.face_model}.py')
    arcface_model = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    ckpt_path = f'checkpoints/{args.face_dataset}_{args.face_model}.pth'
    arcface_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    arcface_model = arcface_model.to(dist_util.dev()).eval()

    # arcface_model = create_arcface_model(**args_to_dict(args, arcface_defaults().keys()))
    # arcface_model = DataParallel(arcface_model, device_ids=[4, 5], output_device=4)
    # arcface_model.load_state_dict(torch.load(args.arcface_path))
    # arcface_model = arcface_model.to(dist_util.dev())
    # arcface_model.eval()

    # spNorm = SpecificNorm()
    # netSeg = BiSeNet(n_classes=19).to(dist_util.dev())
    # netSeg.load_state_dict(torch.load('./checkpoints/FaceParser.pth'))
    # netSeg.eval()

    # face_parser_model = BiSeNet(n_classes=19)
    # face_parser_model.load_state_dict(torch.load('checkpoints/79999_iter.pth', map_location="cpu"))
    # face_parser_model.eval()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    model_kwargs = {}

    # landmark_pth = './imgs/ffhq/landmarks'
    # landmarks1 = pickle.load(open(os.path.join(landmark_pth, f'landmark_ori_256.pkl'), 'rb'))

    landmark_pth = './imgs/faceforensics/original_sequences/c23/images_256_landmark256'
    landmarks1 = pickle.load(open(os.path.join(landmark_pth, f'landmark_ori_256.pkl'), 'rb'))

    landmark_pth = '/hdd/zhengyang/shiyan/deep3dfacerencon_pytorch/landmarks'
    landmarks2 = pickle.load(open(os.path.join(landmark_pth, f'landmark_target.pkl'), 'rb'))
    # if args.data_dir != '':
    #     data = load_data(
    #         data_dir=args.data_dir,
    #         batch_size=args.batch_size,
    #         image_size=args.image_size,
    #         class_cond=True,
    #         random_crop=True,
    #     )
    #     data = next(data)[0].to(dist_util.dev())
    #     model_kwargs["x_0"] = data
    data_source = load_data(
        data_dir=args.data_source_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        # arface_model=arcface_model,
        deterministic=True,
        random_flip=False
    )
    data_target = load_data(
        data_dir=args.data_target_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        landmark=landmarks1,
        landmark2=landmarks2,
        isSample=args.isSample,
        deterministic=True,
        random_flip=False
    )


    ######## diffswap
    # dataset = Portrait(args.data_dir,  args.image_size,  args.image_size)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=False,
    #     drop_last=False,
    # )
    # for batch_idx, batch in enumerate(tqdm(dataloader)):
    #     batch_source = batch["image_src"].to(dist_util.dev())
    #     batch_target = batch["image_target"].to(dist_util.dev())
    #     # print(f'batch_source.shape: {batch_source.shape}')
    #     # print(f'batch_target.shape: {batch_target.shape}')
    #
    #     # model_kwargs["id"] = arcface_model(transform(batch_source))
    #     model_kwargs["id"] = arcface_model(F.interpolate(batch_source, (112, 112), mode='bicubic'))
    #
    #     # targ_mask = batch_target.detach().clone()
    #     # targ_mask = transforms.Resize((512, 512))(targ_mask)
    #     # targ_mask = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(targ_mask)
    #     # targ_mask = netSeg(spNorm(targ_mask))[0]
    #     # targ_mask = transforms.Resize((128, 128))(targ_mask)
    #     # mask = makeMask(targ_mask)
    #     # model_kwargs["mask"] = torch.from_numpy(mask).unsqueeze(0).to(dist_util.dev()).float()
    #     # model_kwargs["mask"].requires_grad_()
    #
    #     mask = batch["mask"].float()
    #     mask = F.interpolate(mask, size=(args.image_size, args.image_size), mode='nearest')
    #     mask[mask > 0] = 1
    #     mask[mask <= 0] = 0
    #     model_kwargs["mask"] = mask.to(dist_util.dev())
    #
    #     print(f"batch['mask].shape: {model_kwargs['mask'].shape}")
    #     sample_fn = (
    #         diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    #     )
    #     sample = sample_fn(
    #         model,
    #         arcface_model,
    #         (batch_source.shape[0], 3, args.image_size, args.image_size),
    #         clip_denoised=args.clip_denoised,
    #         model_kwargs=model_kwargs,
    #         progress=True,
    #         init_img=batch_target,
    #         masking_threshold=args.masking_threshold
    #     )
    #     # sample_img = torch.cat((batch_source, batch_target))
    #     # sample_img = torch.cat((sample_img, sample))
    #     grid = (make_grid(sample) + 1) / 2
    #     # print(batch['name'])
    #     # path = f'./outputs/image_sample15_20240108_1537/result_{batch_idx}.png'
    #     path = f'./outputs/image_sample16_20240219_0924/{batch["name"][0][:-4]}.png'
    #     save_image(grid, path)


    ########   diffFace
    sample_source = None
    sample_target = None
    sample_swap = None

    sample_source_1 = None
    sample_target_1 = None
    sample_swap_1 = None

    for i in range(8):
        # batch_source,_,source_dir = next(data_source)[0].to(dist_util.dev())
        batch_source, _, source_dir = next(data_source)
        batch_source = batch_source.to(dist_util.dev())
        batch_target, cond_target, target_dir = next(data_target)
        batch_target = batch_target.to(dist_util.dev())
        with torch.no_grad():
            model_kwargs["id"] = arcface_model(F.interpolate(batch_source, (112, 112), mode='bicubic'))
        cond_target["landmarks"] = cond_target["landmarks"].reshape(cond_target["landmarks"].shape[0], -1).float()
        model_kwargs["landmarks"] = cond_target["landmarks"].to(dist_util.dev())
        # model_kwargs["mask_organ"] = cond_target["mask_organ"].to(dist_util.dev())
        # model_kwargs["face_parser"] = face_parser_model(face_parser_img)[0].to(dist_util.dev())
        # max_landmark, min_landmark = th.max(model_kwargs["landmarks"]), th.min(model_kwargs["landmarks"])
        # max_id, min_id = th.max(model_kwargs["id"]), th.min(model_kwargs["id"])
        # min-max标准化到[-1, 1]
        # model_kwargs["landmarks"] = 2.0 * th.div(model_kwargs["landmarks"] - min_landmark, max_landmark - min_landmark) - 1.0
        # model_kwargs["id"] = 2.0 * th.div(model_kwargs["id"] - min_id, max_id - min_id) - 1.0

        model_kwargs["mask"] = cond_target["mask"].to(dist_util.dev())
        if args.scale:
            model_kwargs["s"] = args.scale

        # model_kwargs["mask"] = None

        # targ_mask = batch_target.detach().clone()
        # targ_mask = transforms.Resize((512, 512))(targ_mask)
        # targ_mask = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(targ_mask)
        # targ_mask = netSeg(spNorm(targ_mask))[0]
        # targ_mask = transforms.Resize((128, 128))(targ_mask)
        # mask = makeMask(targ_mask)
        # model_kwargs["mask"] = torch.from_numpy(mask).unsqueeze(0).to(dist_util.dev()).float()
        # model_kwargs["mask"].requires_grad_()
        # model_kwargs["mask"] = None

        # print(f"mask.shape: {model_kwargs['mask'].shape}")

        logger.log("sampling...")
        print("sampleing")
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            diffusion.timestep_map,
            arcface_model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            init_img=batch_target,
            source_img=batch_source,
            masking_threshold=args.masking_threshold,
            output_path_dir_type=args.output_path_dir_type,
            up=up,
            down=down,
        )
        print("sample done!")
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        res_swap = sample[0]
        output = res_swap.to('cpu')
        output = np.array(output)
        output = output[..., ::-1]      # rgb ==> bgr for cv2
        cv2.imwrite('./output_ff/' + source_dir[0] + '_' + target_dir[0] + '_' + f'{i}.png', output)

        # mask = ((cond_target["mask"] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # mask = mask.permute(1, 2, 0)
        # mask = mask.contiguous()
        # mask = mask.to('cpu')
        # mask = np.array(mask)
        # cv2.imwrite('./output_mask_test/' + f'{i}.png', mask)


        # mask = cond_target["mask"].permute(1, 2, 0)
        #
        # res_swap = sample[0].permute(1, 2, 0)
        # res_swap = (res_swap + 1) / 2 * 255
        #
        # target_swap = batch_target[0].permute(1, 2, 0)
        # target_swap = (target_swap + 1) / 2 * 255
        #
        # mask, res_swap, target_swap = np.uint8(mask.cpu()), np.int32(res_swap.cpu()), np.int32(target_swap.cpu())
        #
        # gp_1, gp_2 = [gaussian_pyramid(im) for im in [res_swap, target_swap]]
        # mask_gp = [cv2.resize(mask, (gp.shape[1], gp.shape[0])) for gp in gp_1]
        # lp_1, lp_2 = [laplacian_pyramid(gp) for gp in [gp_1, gp_2]]
        # lp_join = laplacian_pyr_join(lp_1, lp_2, mask_gp)
        # im_join = laplacian_collapse(lp_join)
        # np.clip(im_join, 0, 255, out=im_join)
        # im_join = np.uint8(im_join)

        # sample = torch.from_numpy(im_join).permute(2, 0, 1).unsqueeze(0).to(dist_util.dev())
        # sample = sample / 127.5 - 1

        # if sample_source_1 is None:
        #     sample_source_1 = batch_source
        #     sample_target_1 = batch_target
        #     sample_swap_1 = sample
        # else:
        #     sample_source_1 = torch.cat((sample_source_1, batch_source), dim=0)
        #     sample_target_1 = torch.cat((sample_target_1, batch_target), dim=0)
        #     sample_swap_1 = torch.cat((sample_swap_1, sample), dim=0)
        #
        #
        #
        #
        #
        # if sample_source is None:
        #     sample_source = batch_source
        #     sample_target = batch_target
        #     sample_swap = sample
        # else:
        #     sample_source = torch.cat((sample_source, batch_source), dim=0)
        #     sample_target = torch.cat((sample_target, batch_target), dim=0)
        #     sample_swap = torch.cat((sample_swap, sample), dim=0)

        # print(f"res_swap.shape: {res_swap.shape}")
        # print(f"target_swap.shape: {target_swap.shape}")
        # if sample_img is None:
        #     img = torch.cat((batch_source, batch_target), dim=0)
        #     sample_img = torch.cat((img, sample), dim=0)
        # else:
        #     img = torch.cat((batch_source, batch_target), dim=0)
        #     img = torch.cat((img, sample), dim=0)
        #     sample_img = torch.cat((sample_img, img), dim=0)

    # img = torch.cat((sample_source, sample_target), dim=0)
    # sample_img = torch.cat((img, sample_swap), dim=0)
    # grid = (make_grid(sample_img, nrow=8) + 1) / 2


    # path = './outputs/image_sample16_20240219_0924/result_3.png'
    # path_dir = './outputs/image_sample24_20240317_1741/'

    # path_dir = f'./outputs/{args.output_path_dir}/'
    # os.makedirs(path_dir, exist_ok=True)
    # path_dir = path_dir + f'{args.output_path_dir_type}/'
    # os.makedirs(path_dir, exist_ok=True)
    # path = path_dir + f'result_{args.timestep_respacing}_{args.output_pt}pt_mask.png'

    # path_1 = path_dir + f'result_{args.timestep_respacing}_{args.output_pt}pt_mask_nolaplacian.png'

    # print(f"sample_path: {path}")
    # save_image(grid, path)

    # save_image(grid_1, path_1)

    # diffswap
    # repair_by_mask()

    dist.barrier()
    logger.log("sampling complete")

def repair_by_mask(tgt_path='./imgs/testimg_dir/align/target', swap_path='./outputs/image_sample16_20240219_0924',
                   save_path='./outputs', mask_path='./imgs/testimg_dir/mask/mask'):

    img_list = os.listdir(swap_path)
    for img in tqdm(img_list, leave=False):
        # if ((img == '0013.png' and src == '0006') is False) and ((img == '1086.png' and src == '0005') is False) and \
        #     ((img == '0208.png' and src == '0006') is False) and ((img == '0092.png' and src == '0005') is False) and \
        #         ((img == '0021.png' and src == '0002') is False) and ((img == '0021.png' and src == '0006') is False):
        #     continue
        swap_img = cv2.imread(os.path.join(swap_path, img))
        im1 = cv2.cvtColor(swap_img, cv2.COLOR_BGR2RGB)
        tgt_img = cv2.imread(os.path.join(tgt_path, img))
        im2 = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
        mask = matplotlib.image.imread(os.path.join(mask_path, img))
        im1, im2 = np.int32(im1), np.int32(im2)
        mask = np.uint8(mask)

        gp_1, gp_2 = [gaussian_pyramid(im) for im in [im1, im2]]
        mask_gp = [cv2.resize(mask, (gp.shape[1], gp.shape[0])) for gp in gp_1]
        lp_1, lp_2 = [laplacian_pyramid(gp) for gp in [gp_1, gp_2]]
        lp_join = laplacian_pyr_join(lp_1, lp_2, mask_gp)
        im_join = laplacian_collapse(lp_join)
        np.clip(im_join, 0, 255, out=im_join)
        im_join = np.uint8(im_join)

        os.makedirs(os.path.join(save_path), exist_ok=True)
        plt.imsave(os.path.join(save_path, img), im_join)

def create_argparser():
    defaults = dict(
        data_source_dir="./imgs/testimg_dir/align/source_ff",
        data_target_dir="./imgs/testimg_dir/align/target_ff",
        # data_source_dir="./imgs/testimg_dir/align/new_source",
        # data_target_dir="./imgs/testimg_dir/align/new_target",
        data_dir="./imgs/testimg_dir",
        clip_denoised=True,
        num_samples=100,
        batch_size=1,
        use_ddim=False,
        model_path="",
        down_N=32,
        arcface_path='',
        masking_threshold=30,
        num_workers=4,
        face_dataset='glint360k',
        face_model='r100',
        scale=0.0,
        output_path_dir="",
        output_path_dir_type="mask_alter",
        output_pt='',
        isSample=True,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(arcface_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def makeMask(origin_mask):
    numpy = origin_mask.squeeze(0).detach().cpu().numpy().argmax(0)
    # numpy = origin_mask.detach().cpu().numpy().argmax(0)
    numpy = numpy.copy().astype(np.uint8)
    print(numpy.shape)
    # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
    ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]

    mask     = np.zeros([128, 128])
    for id in ids:
        index = np.where(numpy == id)
        mask[index] = 1

    return np.expand_dims(mask, axis=0)
if __name__ == "__main__":
    main()
