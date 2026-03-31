import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from util.tensor_util import unpad
from model.eval_network import XRAVOS
from inference.inference_core_mosxav import InferenceCore
from dataset.xray_test_dataset import XRayTestDataset

from progressbar import progressbar
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/xravos_sw15_s3_125000.pth')
parser.add_argument('--data_path')
parser.add_argument('--split', default='val', choices=['val', 'test'])
parser.add_argument('--output')
parser.add_argument('--s_w', type=int, default=7)
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp_off', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

data_path = args.data_path
out_path = args.output
args.amp = not args.amp_off

# Simple setup
os.makedirs(out_path, exist_ok=True)
torch.autograd.set_grad_enabled(False)

# Setup Dataset
test_dataset = XRayTestDataset(
    data_root=data_path, imset=args.split + '.txt', setjson=args.split + '_first_mask.json', resolution=-1
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

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
        rgb = data['rgb']
        msk = data['gt'][0]
        info = data['info']
        name = info['name'][0]
        num_objects = len(info['labels'][0])
        gt_obj = info['gt_obj']
        size = info['size']
        palette = data['palette'][0]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        print('Processing', name, '...')

        # Frames with labels, but they are not exhaustively labeled
        frames_with_gt = sorted(list(gt_obj.keys()))
        processor = InferenceCore(
            prop_model,
            rgb,
            num_objects=num_objects,
            s_w=s_w,
            top_k=top_k,
            mem_every=args.mem_every,
            include_last=args.include_last
        )

        # min_idx tells us the starting point of propagation
        # Propagating before there are labels is not useful
        min_idx = 99999
        for i, frame_idx in enumerate(frames_with_gt):
            min_idx = min(frame_idx, min_idx)
            # Note that there might be more than one label per frame
            obj_idx = gt_obj[frame_idx][0].tolist()
            # Map the possibly non-continuous labels into a continuous scheme
            obj_idx = [info['label_convert'][o].item() for o in obj_idx]

            # Append the background label
            with_bg_msk = torch.cat([
                1 - torch.sum(msk[:, frame_idx], dim=0, keepdim=True),
                msk[:, frame_idx],
            ], 0).cuda()

            # We perform propagation from the current frame to the next frame with label
            if i == len(frames_with_gt) - 1:
                processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
            else:
                processor.interact(with_bg_msk, frame_idx, frames_with_gt[i + 1] + 1, obj_idx)

        # Do unpad -> upsample to original size (we made it 480p)
        out_masks = torch.zeros(processor.t, 1, *size, dtype=torch.uint8, device='cuda')

        for ti in range(processor.t):
            prob = unpad(processor.prob[:, ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

        out_masks = (out_masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)

        end.record()
        torch.cuda.synchronize()
        total_process_time += (start.elapsed_time(end) / 1000)
        total_frames += out_masks.shape[0]

        # Remap the indices to the original domain
        idx_masks = np.zeros_like(out_masks)
        for i in range(1, num_objects + 1):
            backward_idx = info['label_backward'][i].item()
            idx_masks[out_masks == i] = backward_idx

        # Save the results
        this_out_path = os.path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(idx_masks.shape[0]):
            if f >= min_idx:
                img_E = Image.fromarray(idx_masks[f])
                img_E.putpalette(palette)
                img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg', '.png')))

        del rgb
        del msk
        del processor

print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')
