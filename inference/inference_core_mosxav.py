import torch

from model.eval_network import XRAVOS
from model.aggregate import aggregate
from model.sample import sample_overlapped
from inference.memory_manager import MemoryBank

from util.tensor_util import pad_divide_by


class InferenceCore:

    def __init__(
        self, prop_net: XRAVOS, images, num_objects, s_w=7, top_k=20, mem_every=5, include_last=False, req_frames=None
    ):
        self.s_w = s_w
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last

        # We HAVE to get the output for these frames
        # None if all frames are required
        self.req_frames = req_frames

        self.top_k = top_k

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k + 1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh // 16
        self.kw = self.nw // 16

        # attn mask
        attn_mask = torch.ones(1, 1, self.kh, self.kw)
        attn_mask = sample_overlapped(attn_mask, self.s_w, pad_value=-1).view(1, 1, -1, self.kh, self.kw)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(-100.0)).masked_fill(attn_mask >= 0, float(0.0))
        self.attn_mask = attn_mask

        # list of objects with usable memory
        self.enabled_obj = []

        self.mem_banks = dict()

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:, idx].cuda())
        return result

    def do_pass(self, key_k, key_v, mask, idx, end_idx):
        closest_ti = end_idx

        # _, CK, _, N, H, W = key_k.shape
        # K, CV, _, _, _, _ = key_v.shape

        for i, oi in enumerate(self.enabled_obj):
            if oi not in self.mem_banks:
                self.mem_banks[oi] = MemoryBank(k=1, top_k=self.top_k)
            self.mem_banks[oi].add_memory(key_k, key_v[i:i + 1], mask)

        last_ti = idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx + 1, closest_ti)
        step = +1
        end = closest_ti - 1

        for ti in this_range:
            is_mem_frame = (abs(ti - last_ti) >= self.mem_every)
            # Why even work on it if it is not required for memory/output
            if (not is_mem_frame) and (not self.include_last) and (self.req_frames is not None) and (ti not in self.req_frames):
                continue

            k16, qv16, qf16, qf8, qf4 = self.encode_key(ti)

            # After this step all keys will have the same size
            out_mask = torch.cat(
                [self.prop_net.segment_with_query(self.mem_banks[oi], qf8, qf4, k16, qv16) for oi in self.enabled_obj], 0
            )

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[0, ti] = out_mask[0]
            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi, ti] = out_mask[i + 1]

            if ti != end:
                if self.include_last or is_mem_frame:
                    prev_value = self.prop_net.encode_value(self.images[:, ti].cuda(), qf16, out_mask[1:])
                    prev_value = prev_value.transpose(1, 2).contiguous().view(prev_value.shape[0], -1, self.kh, self.kw)
                    prev_value = sample_overlapped(prev_value, self.s_w
                                                  ).view(prev_value.shape[0], 1, -1, self.s_w * self.s_w, self.kh,
                                                         self.kw).transpose(1, 2).contiguous()
                    prev_key = sample_overlapped(k16, self.s_w).view(1, -1, self.s_w * self.s_w, self.kh, self.kw).unsqueeze(2)
                    for i, oi in enumerate(self.enabled_obj):
                        self.mem_banks[oi].add_memory(
                            prev_key, prev_value[i:i + 1], self.attn_mask.cuda(), is_temp=not is_mem_frame
                        )

                    if is_mem_frame:
                        last_ti = ti

        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)

        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx]

        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx], keep_bg=True)

        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_k = sample_overlapped(key_k, self.s_w).view(1, -1, self.s_w * self.s_w, self.kh, self.kw).unsqueeze(2)
        key_v = self.prop_net.encode_value(
            self.images[:, frame_idx].cuda(), qf16, self.prob[self.enabled_obj, frame_idx].cuda()
        )
        key_v = key_v.transpose(1, 2).contiguous().view(key_v.shape[0], -1, self.kh, self.kw)
        key_v = sample_overlapped(key_v, self.s_w).view(key_v.shape[0], 1, -1, self.s_w * self.s_w, self.kh,
                                                        self.kw).transpose(1, 2).contiguous()

        # Propagate
        self.do_pass(key_k, key_v, self.attn_mask.cuda(), frame_idx, end_idx)
