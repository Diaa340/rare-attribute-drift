"""Forward hooks on the 25 FLUX.2 [klein] 4B blocks (D0-D4 double-stream, S0-S19 single-stream) to read or shift the
hidden states of the target-image tokens (the 1024 tokens being generated; reference and text tokens are left alone).

Double block output: (txt, img) with img = [target 1024, ref 1024]. Single block output: [txt L, target 1024, ref 1024].
"""
import torch

N_TGT = 1024


class Hooks:
    def __init__(self, transformer):
        self.capture = None          # dict layer -> target-token hidden states, filled when not None
        self.pool = None             # (1024,) bool: if set, capture stores the mean over these tokens (3072,) instead
        self.add = None              # dict layer -> (3072,) vector to add on masked target tokens
        self.mask = None             # (1024,) bool; None = all target tokens
        self.layers = [f"D{i}" for i in range(len(transformer.transformer_blocks))] + \
                      [f"S{i}" for i in range(len(transformer.single_transformer_blocks))]
        blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
        self.handles = [b.register_forward_hook(self._hook(name)) for name, b in zip(self.layers, blocks)]

    def _target(self, name, out):
        """Return (tensor holding target tokens, start index)."""
        if name.startswith("D"):
            return out[1], 0
        return out, out.shape[1] - 2 * N_TGT          # single stream: text tokens first, then target, then ref

    def _hook(self, name):
        def fn(module, inp, out):
            h, s = self._target(name, out)
            if self.capture is not None:
                tgt = h[0, s:s + N_TGT].float()
                self.capture[name] = (tgt[self.pool.to(h.device)].mean(0) if self.pool is not None else tgt).cpu()
            if self.add is not None and name in self.add:
                vec = self.add[name].to(h.device, h.dtype)
                idx = torch.arange(s, s + N_TGT, device=h.device)
                if self.mask is not None:
                    idx = idx[self.mask.to(h.device)]
                h[0, idx] += vec                       # in place: the block output is consumed downstream
            return out
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()
