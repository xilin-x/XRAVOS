import torch
import torch.nn.functional as F


def sample_overlapped(x: torch.Tensor, s_w: int, pad_value=None):
    B, C, H, W = x.shape
    pad_l, pad_r, pad_t, pad_b = s_w // 2, s_w // 2, s_w // 2, s_w // 2
    x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0 if pad_value is None else pad_value)
    x_sample = x.unfold(2, s_w, 1).unfold(3, s_w, 1).reshape(B, C, H, W, s_w * s_w)
    return x_sample.permute(0, 1, 4, 2, 3).contiguous().view(B, -1, H, W)
