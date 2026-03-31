import math
import torch


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=2)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=2, keepdim=True)
    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(2, indices, x_exp.type(x.dtype))  # B * THW * HW

    return x


class MemoryBank:

    def __init__(self, k, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None
        self.mask = None

        self.num_objects = k

    def _global_matching(self, mk, qk):
        # NE means number of elements -- typically T*H*W
        B, CK, _, NE = mk.shape

        # See supplementary material
        a_sq = mk.pow(2).sum(1, keepdim=True)
        ab = (qk * mk).sum(1, keepdim=True)

        affinity = (2 * ab - a_sq) / math.sqrt(CK)  # B, NE, HW
        if self.mem_mask is not None:
            affinity = affinity + self.mem_mask
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW

        return affinity

    def _readout(self, affinity, mv):
        return (affinity * mv).sum(2)

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2).unsqueeze(2)

        mk = torch.cat([self.mem_k, self.temp_k], 2) if self.temp_k is not None else self.mem_k
        mv = torch.cat([self.mem_v, self.temp_v], 2) if self.temp_v is not None else self.mem_v

        affinity = self._global_matching(mk, qk)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k, -1, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, mask, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        self.temp_mask = None
        key = key.view(*key.shape[:2], -1, key.shape[-2] * key.shape[-1])
        value = value.view(*value.shape[:2], -1, value.shape[-2] * value.shape[-1])
        mask = mask.view(*mask.shape[:3], -1)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.mem_mask = mask
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
                self.temp_mask = mask
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)
                self.mem_mask = torch.cat([self.mem_mask, mask], 2)
