"""Minimal LoRA for the single-stream blocks of FLUX.2 [klein] 4B, plus a velocity function that keeps gradients."""
import torch
from torch import nn

BLOCKS = list(range(4, 13))                  # S4-S12, where attribute identity is most decodable
TARGETS = ["to_qkv_mlp_proj", "to_out"]


class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=16):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / r, True
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)
        return y


def add_lora(transformer, r=16, alpha=16, blocks=BLOCKS):
    transformer.requires_grad_(False)
    mods = {}
    for b in blocks:
        attn = transformer.single_transformer_blocks[b].attn
        for name in TARGETS:
            base = getattr(attn, name)
            assert isinstance(base, nn.Linear), (b, name, type(base))
            m = LoRALinear(base, r, alpha).to(base.weight.device)
            setattr(attn, name, m); mods[f"S{b}.{name}"] = m
    return mods


def set_enabled(mods, on):
    for m in mods.values():
        m.enabled = on


def state(mods):
    return {k: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()} for k, m in mods.items()}


def load_state(mods, sd):
    for k, m in mods.items():
        m.A.data.copy_(sd[k]["A"]); m.B.data.copy_(sd[k]["B"])


def vel(E, x, t, text, st):
    """Same as Editor.velocity but without no_grad."""
    emb, ids = text
    out = E.pipe.transformer(hidden_states=torch.cat([x, st["ref"]], 1).to(torch.bfloat16), timestep=t.expand(1).to(x.dtype) / 1000,
                             guidance=None, encoder_hidden_states=emb, txt_ids=ids,
                             img_ids=torch.cat([st["x_ids"], st["ref_ids"]], 1), return_dict=False)[0]
    return out[:, : x.shape[1]]
