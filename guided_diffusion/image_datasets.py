import math
import random
import sys
import time

import torch
import PIL
from PIL import Image, ImageFile
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import dlib
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import glob
from torchvision import transforms

from scipy.spatial import ConvexHull

import os

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    # arface_model=None,
    landmark=None,
    landmark2=None,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    isSample=False,
    landmarks_issue=None
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir, landmarks_issue)         # 将每张图片添加到all_files中
    # if landmark is not None:
    #     # source_seq = 034, 045, 069, 073, 121,241,493,553
    #     target_seq = ['590', '889', '961', '024', '093', '210', '538', '545']
    #
    #     # source_seq=171,177,184,190
    #     # target_seq = ['173', '211', '205', '176']
    #     new_all_files = []
    #     for i in range(8):
    #         for item in all_files:
    #             if target_seq[i] in item:
    #                 new_all_files.append(item)
    #     all_files = new_all_files

    print(all_files)
    # './imgs/testimg_dir/align/source/08000/08017.png'
    # './imgs/ffhq/align512/08000/08017.png
    all_files_512 = None
    # print("test")
    # if not isSample:
    # print(data_dir)

    # all_files_512 = []
    # for file in all_files:
    #     file_lists = file.split('/')
    #     all_files_512.append(f'./imgs/ffhq/align512/{file_lists[-2]}/{file_lists[-1]}')
    # else:
    #     if "256" in data_dir:
    #         data_dir = data_dir.replace("256", "512")
    #     else:
    #         sys.exit()
    #     all_files_512 = _list_image_files_recursively(data_dir, landmarks_issue)         # 将每张图片添加到all_files中
    # print(all_files_512)
    # sys.exit()
    # classes = None
    # if class_cond:  # False
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]       # basename 获取图片名字 ./imgs/data1/n000012\0001_01.jpg ==> 0001_01.jpg
    #     # print(class_names)      # 0001  0003
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}     # '0001':0
    #     # class id   1_1.jpg 1_2.jpg 2_1.jpg 2_2.jpg  [0, 1]
    #     classes = [sorted_classes[x] for x in class_names]  # 0.1.2.3

    dataset = ImageDataset(
        image_size,
        all_files,
        # all_files_512,
        # classes=classes,
        # arface_model=arface_model,
        landmark=landmark,
        landmark2=landmark2,
        # shard=MPI.COMM_WORLD.Get_rank(),
        # num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        isSample=isSample,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir, landmarks_issue=None):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            if landmarks_issue is not None:
                if "512" in full_path:
                    new_full_path = full_path.replace("512", "256")
                    if new_full_path in landmarks_issue: continue
                if full_path in landmarks_issue: continue
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path, landmarks_issue=landmarks_issue))        # extend 在list列表末尾追加 宁外一个列表中的值
    # print(results)
    return results

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        # image512_paths,
        classes=None,
        # arface_model=None,
        landmark=None,
        landmark2=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        isSample=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        # self.local_images512 = image512_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        # self.arcface_model = arface_model
        self.landmark = landmark
        self.landmark2 = landmark2
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.isSample = isSample
        self.predictor = None
        self.detector = None
        self.get_detector()
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        # all_indices = np.arange(0, 68)
        # self.landmark_indices = {
        #     # 'face': all_indices[:17].tolist() + all_indices[17:27].tolist(),
        #     'l_eye': all_indices[36:42].tolist(),
        #     'r_eye': all_indices[42:48].tolist(),
        #     'nose': all_indices[27:36].tolist(),
        #     'mouth': all_indices[48:68].tolist(),
        # }
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]       # 1_1.jpg 1_2.jpg 2_1.jpg 2_2.jpg  [0, 1]

        # while True:
        #     source_idx = random.randint(0, len(self.local_images) - 1)
        #     if source_idx != idx:
        #         break

        # source_path = self.local_images[source_idx]
        flag = 0
        image = Image.open(path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), resample=PIL.Image.BICUBIC)
        arr = np.array(image)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            flag = 1
        arr = arr.astype(np.float32) / 127.5 - 1


        # source_image = Image.open(source_path).convert('RGB')
        # source_image = source_image.resize((self.resolution, self.resolution), resample=PIL.Image.BICUBIC)
        # source_arr = np.array(source_image)
        # source_arr = source_arr.astype(np.float32) / 127.5 - 1
        out_dict = {}
        img_dir = path.split('/')[-2]
        img = path.split('/')[-1]
        # print(f"img_dir: {img_dir}, img: {img}")
        if self.landmark:
            # train_setting  on ffhq
            # out_dict["landmarks"] = torch.tensor(self.landmark[img_dir][img + f"_{flag}"]) / self.resolution # torch.size([68, 2])
            # test_setting on faceforensics
            out_dict["landmarks"] = torch.tensor(self.landmark[img_dir][img]) / self.resolution # torch.size([68, 2])
            # lmk1 = self.landmark[img_dir][img + f"_{flag}"] / self.resolution # torch.size([68, 2])
            # lmk2 = self.landmark2['target'][img] / 2 / self.resolution # torch.size([68, 2])
            # c_lmk = np.vstack((lmk1, lmk2))
            # out_dict["landmarks"] = torch.tensor(lmk2)
            # c_lmk = torch.tensor(c_lmk)
            # out_dict["mask_organ"] = self.extract_convex_hulls(out_dict["landmarks"])
            if self.isSample:
                out_dict["mask"] = self.extract_convex_hull(out_dict["landmarks"])      # numpy (128, 128)
        # C H W
        # return np.transpose(arr, [2, 0, 1]), np.transpose(source_arr, [2, 0, 1]), out_dict
        # face_parser image
        # if not self.isSample:
        # path512 = self.local_images512[idx]
        # face_parser_img = Image.open(path512)
        # if flag:
        #     face_parser_img = face_parser_img.transpose(Image.FLIP_LEFT_RIGHT)
        # face_parser_img = self.transform(face_parser_img)
        # return np.transpose(arr, [2, 0, 1]), out_dict, face_parser_img
        return np.transpose(arr, [2, 0, 1]), out_dict, img_dir

    def extract_convex_hulls(self, landmark):
        mask_dict = {}
        mask_organ = []
        for key, indices in self.landmark_indices.items():
            mask_key = self.extract_convex_hull(landmark[indices])
            # if self.dilate:
            #     # mask_key = mask_key[:, :, None]
            #     # mask_key = repeat(mask_key, 'h w -> h w k', k=3)
            #     # print(mask_key.shape, type(mask_key))
            #     mask_key = mask_key.astype(np.uint8)
            #     mask_key = cv2.dilate(mask_key, self.dilate_kernel, iterations=1)
            mask_organ.append(mask_key)
        mask_organ = np.stack(mask_organ) # (4, 256, 256)
        return mask_organ
        # mask_dict['mask_organ'] = mask_organ
        # mask_dict['mask'] = self.extract_convex_hull(landmark)
        # return mask_dict

    def extract_convex_hull(self, landmark):
        landmark = landmark * self.resolution
        hull = ConvexHull(landmark)
        image = np.zeros((self.resolution, self.resolution))
        points = [landmark[hull.vertices, :1], landmark[hull.vertices, 1:]]
        points = np.concatenate(points, axis=-1).astype('int32')
        mask = cv2.fillPoly(image, pts=[points], color=(255,255,255))
        mask = mask > 0
        mask = mask.astype(np.uint8)
        return mask

    def get_mask(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        landmarks = self.get_landmarks(image)
        mask = self.get_image_hull_mask(np.shape(image), landmarks).astype(np.uint8)
        print(f"mask.shape: {mask.shape}")
        return mask

    def get_detector(self):
        predictor_path = "/hdd/zhengyang/shiyan/guided-diffusion/models/dlib/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)
        self.detector = dlib.get_frontal_face_detector()

    def get_landmark(self, image_path):
        image = Image.open(image_path).convert("L")
        image_array = np.array(image)
        faces = self.detector(image_array)
        # print(f"faces: {faces}")
        landmarks_np = []
        for face in faces:
            landmarks = self.predictor(image_array, face)
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
        if len(landmarks_np) == 0:
            print(image_path)
            import os
            os.system(f"rm {image_path}")
        return landmarks_np

    def get_landmarks(self, image):

        predictor_model = '/hdd/zhengyang/shiyan/guided-diffusion/models/dlib/shape_predictor_68_face_landmarks.dat'
        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor(predictor_model)
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = self.detector(img_gray, 0)
        # landmarks = []
        assert len(rects) == 1
        landmarks = np.matrix([[p.x, p.y] for p in self.predictor(image, rects[0]).parts()])        #
        return landmarks

    def get_image_hull_mask(self, image_shape, image_landmarks, ie_polys=None):
        # get the mask of the image
        if image_landmarks.shape[0] != 68:
            raise Exception(
                'get_image_hull_mask works only with 68 landmarks')
        int_lmrks = np.array(image_landmarks, dtype=np.int)

        # hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
        hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

        # cv2.convexHull(points[, hull[, clockwise[, returnPoints]]])  用于找到一个点集的凸包
        # points: 二维点的数组。 hull: 输出的凸包的数组。如果这个参数为 None，则函数不会返回任何值。

        # cv2.fillConvexPoly(img, points, color[, lineType[, shift]])   用于填充凸多边形的内部。这个函数在二值图像上工作，所以通常用于标记、分割或识别任务。
        # img: 输入/输出图像。  points: 多边形的顶点坐标。
        # color: 填充颜色。对于 8位单通道图像，颜色通常是一个介于0-255之间的整数。对于32位浮点图像，颜色是一个浮点数数组。

        # numpy.concatenate((a1, a2, ...), axis=0)    用于将多个数组沿着指定的轴连接起来

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[0:9],
                            int_lmrks[17:18]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[8:17],
                            int_lmrks[26:27]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[17:20],
                            int_lmrks[8:9]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[24:27],
                            int_lmrks[8:9]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[19:25],
                            int_lmrks[8:9],
                            ))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[17:22],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            ))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[22:27],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            ))), (1,))

        # nose
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

        if ie_polys is not None:
            ie_polys.overlay_mask(hull_mask)
        return hull_mask


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
