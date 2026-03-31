import os
import json

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import xray_normalization
from dataset.util import all_to_onehot


class XRayTestDataset(Dataset):

    def __init__(self, data_root, imset='val.txt', setjson="val_first_mask.json", resolution=-1):
        self.image_dir = os.path.join(data_root, 'JPEGImages')
        self.mask_dir = os.path.join(data_root, 'Annotations')
        _imset_dir = os.path.join(data_root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)
        _setjson_f = os.path.join(_imset_dir, setjson)

        self.videos = []
        self.shape = {}
        self.frames = {}

        with open(os.path.join(_setjson_f), "r") as set_infos:
            self.set_dict = json.load(set_infos)['videos']

        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                _frames = sorted(os.listdir(os.path.join(self.image_dir, _video)))
                self.frames[_video] = _frames
                self.videos.append(_video)
                _mask = np.array(
                    Image.open(os.path.join(self.mask_dir, _video, self.set_dict[_video][0] + '.png')).convert("P")
                )
                self.shape[_video] = np.shape(_mask)

        if resolution != -1:
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
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                xray_normalization,
            ])

            self.mask_transform = transforms.Compose([])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video
        info['frames'] = self.frames[video]
        info['size'] = self.shape[video]  # Real sizes
        info['gt_obj'] = {}  # Frames with labelled objects

        vid_im_path = os.path.join(self.image_dir, video)
        vid_gt_path = os.path.join(self.mask_dir, video)

        frames = self.frames[video]

        images = []
        masks = []
        exist_labels = []
        for i, f in enumerate(frames):
            img = Image.open(os.path.join(vid_im_path, f)).convert('RGB')
            images.append(self.im_transform(img))

            if f[:-4] in self.set_dict[video]:
                mask_file = os.path.join(vid_gt_path, f[:-4] + '.png')
                mask = Image.open(mask_file).convert('P')
                palette = mask.getpalette()
                mask = np.array(mask, dtype=np.uint8)
                this_labels = np.unique(mask)
                this_labels = this_labels[this_labels != 0]
                this_labels = [this_label for this_label in this_labels if this_label not in exist_labels]
                this_labels = np.array(this_labels)
                exist_labels.extend(this_labels)
                select_mask = np.zeros(self.shape[video], dtype=np.uint8)
                for this_label in this_labels:
                    select_mask[mask == this_label] = this_label
                masks.append(np.array(select_mask, dtype=np.uint8))
                info['gt_obj'][i] = this_labels
            else:
                # Mask not exists -> nothing in it
                masks.append(np.zeros(self.shape[video]))

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels != 0]
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        # Resize to 480p
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
            'palette': np.array(palette),
        }

        return data

    def __len__(self):
        return len(self.videos)
