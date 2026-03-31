"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""

import os

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data.dataset import Dataset
from dataset.range_transform import xray_normalization
from dataset.util import all_to_onehot

palette = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64,
    128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0,
    64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64,
    128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128,
    128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64,
    128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64,
    192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0,
    32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128,
    96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160,
    192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64,
    160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128,
    64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64,
    32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192,
    224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160,
    128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128,
    0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0,
    64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128,
    160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32,
    192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192,
    0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192,
    192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96,
    32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0,
    32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224,
    224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32,
    192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32,
    192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224,
    192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224,
    192
]


class XACVTestDataset(Dataset):

    def __init__(self, root, resolution=512):
        self.root = root
        vid_list = sorted(os.listdir(os.path.join(root, 'JPEGImages')))
        self.resolution = resolution

        self.mask_dir = os.path.join(root, 'AnnotationVOS')
        self.image_dir = os.path.join(root, 'JPEGImages')
        # _imset_dir = os.path.join(root, 'ImageSets')
        # _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.frames = {}
        self.num_objects = {}
        self.shape = {}

        for vid in vid_list:
            self.videos.append(vid)

            frames = sorted(os.listdir(os.path.join(self.mask_dir, vid + 'CATH')))
            self.frames[vid] = frames

            first_mask = sorted(os.listdir(os.path.join(self.mask_dir, vid + 'CATH')))[0]
            _mask = np.array(Image.open(os.path.join(self.mask_dir, vid + 'CATH', first_mask)).convert("P"))
            self.num_objects[vid] = np.max(_mask)
            self.shape[vid] = np.shape(_mask)

        if resolution == 480:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                xray_normalization,
            ])
        else:
            self.im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    xray_normalization,
                    transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                ]
            )
            self.mask_transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=InterpolationMode.NEAREST),
            ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['frames'] = self.frames[video]
        info['size'] = self.shape[video]

        vid_im_path = os.path.join(self.image_dir, video)
        vid_gt_path = os.path.join(self.mask_dir, video + 'CATH')

        frames = self.frames[video]

        images = []
        masks = []
        for i, f in enumerate(frames):
            img_file = os.path.join(self.image_dir, video, f)
            images.append(self.im_transform(Image.open(img_file).convert('RGB')))

            mask_file = os.path.join(self.mask_dir, video + 'CATH', f)
            if os.path.exists(mask_file):
                masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
            else:
                # Test-set maybe?
                masks.append(np.zeros_like(masks[0]))

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        labels = np.unique(masks[0])
        labels = labels[labels != 0]
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        if self.resolution != 480:
            masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
        }

        return data
