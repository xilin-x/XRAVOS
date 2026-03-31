import os
import json
import time
from argparse import ArgumentParser
from progressbar import progressbar

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from util.tensor_util import unpad
from model.eval_network import XRAVOS
from inference.inference_core_xacv import InferenceCore
from dataset.xacv_test_dataset import palette, XACVTestDataset

from progressbar import progressbar
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/xravos_sw15_s3_125000')
parser.add_argument('--xacv_path')
parser.add_argument('--output')
parser.add_argument('--s_w', type=int, default=7)
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

xacv_path = args.xacv_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
with open("util/palette.json") as f:
    palette = json.load(f)

torch.autograd.set_grad_enabled(False)

# Setup Dataset
test_dataset = XACVTestDataset(xacv_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load our checkpoint
s_w = args.s_w
top_k = args.top
prop_model = XRAVOS().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64, 1, 7, 7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    # with torch.cuda.amp.autocast(enabled=args.amp):  --- IGNORE ---
    with torch.amp.autocast(device_type='cuda', enabled=args.amp):
        rgb = data['rgb'].cuda()
        msk = data['gt'][0].cuda()
        info = data['info']
        name = info['name'][0]
        k = len(info['labels'][0])
        size = info['size']

        torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCore(
            prop_model, rgb, k, s_w=s_w, top_k=top_k, mem_every=args.mem_every, include_last=args.include_last
        )
        processor.interact(msk[:, 0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size
        out_masks = torch.zeros(processor.t, 1, *size, dtype=torch.uint8, device='cuda')
        for ti in range(processor.t):
            prob = unpad(processor.prob[:, ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

        out_masks = (out_masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)

        torch.cuda.synchronize()
        total_process_time += time.time() - process_begin
        total_frames += out_masks.shape[0]

        # Save the results
        this_out_path = os.path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, '{}'.format(info['frames'][f][0])))

        del rgb
        del msk
        del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)
