"""Diagnostic for label-free "reference-consistency" steering.

For every fixed dev drift case (rare view) and its common-view counterpart, on the vanilla trajectory at steps 0-1:
  g_l  = mean_{subject ref tokens} h_l  -  mean_{subject target tokens} h_l          (25 blocks, 3072-d)
and, for the rare view only, the known correction with the same target mask:
  dh_l = h_l(detailed) - h_l(vanilla) pooled on the subject target tokens.
Subject masks (label-free): top 25% attention to the subject word, blocks S13-S19, step 0, separately for target
queries and reference queries. Saves refdir/<id>_<seed>.pt. Usage: python refdir_collect.py
"""
from pathlib import Path

import pandas as pd
import torch
from diffusers.models.transformers.transformer_flux2 import apply_rotary_emb
from PIL import Image

from attn_mask import subject_token_ids, top_mask
from velocity import Editor

N_TXT, N_TGT = 512, 1024
out = Path("refdir"); out.mkdir(exist_ok=True)


class Probe:
    """Attention-to-subject maps for target and reference queries (blocks S13-S19) + pooled hidden states of all blocks."""

    def __init__(self, tr):
        self.ids, self.maps = None, {}
        self.cap, self.mt, self.mr = None, None, None
        for b in range(13, 20):
            attn = tr.single_transformer_blocks[b].attn
            attn.set_processor(self._wrap(attn.processor, b))
        self.layers = [f"D{i}" for i in range(len(tr.transformer_blocks))] + [f"S{i}" for i in range(len(tr.single_transformer_blocks))]
        for name, blk in zip(self.layers, list(tr.transformer_blocks) + list(tr.single_transformer_blocks)):
            blk.register_forward_hook(self._hook(name))

    def _wrap(self, proc, b):
        probe = self

        class W:
            def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None, **kw):
                if probe.ids:
                    qkv = attn.to_qkv_mlp_proj(hidden_states)[..., : 3 * attn.inner_dim]
                    q, k, _ = qkv.chunk(3, dim=-1)
                    q = attn.norm_q(q.unflatten(-1, (-1, attn.head_dim))); k = attn.norm_k(k.unflatten(-1, (-1, attn.head_dim)))
                    q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1); k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
                    p = torch.softmax((q[:, N_TXT:].transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)).float() / attn.head_dim ** 0.5, -1)
                    probe.maps[b] = p[0][:, :, probe.ids].sum(-1).mean(0).cpu()          # (2048,): target then ref queries
                return proc(attn, hidden_states, attention_mask=attention_mask, image_rotary_emb=image_rotary_emb, **kw)
        return W()

    def _hook(self, name):
        def fn(module, inp, o):
            if self.cap is None:
                return
            h, s = (o[1], 0) if name.startswith("D") else (o, o.shape[1] - 2 * N_TGT)
            t = h[0, s:s + N_TGT].float(); r = h[0, s + N_TGT:s + 2 * N_TGT].float()
            self.cap[name] = (t[self.mt].mean(0).cpu(), r[self.mr].mean(0).cpu())
        return fn


d = pd.read_parquet("drift_table_dev.parquet"); d = d[d.fixed_detailed].reset_index(drop=True)
meta = pd.read_json("data/metadata.jsonl", lines=True).set_index(["id", "edit_seed"])
E = Editor(); P = Probe(E.pipe.transformer); tok = E.pipe.tokenizer
jobs = [(r.id, r.edit_seed, True) for r in d.itertuples()] + \
       sorted({(r.pair_id + "_standard", r.edit_seed, False) for r in d.itertuples()})
for n, (cid, seed, rare) in enumerate(jobs):
    f = out / f"{cid}_{seed}.pt"
    if f.exists():
        continue
    r = meta.loc[(cid, seed)]
    st = E.prepare(Image.open("data/" + r.source_image), int(seed)); tv = E.encode_text(r.edit_prompt)
    P.ids, P.maps = subject_token_ids(tok, r.edit_prompt, r.subject), {}
    E.velocity(st["x"], st["ts"][0], tv, st); P.ids = None
    amap = torch.stack(list(P.maps.values())).mean(0) if P.maps else torch.ones(2 * N_TGT)
    P.mt = top_mask(amap[:N_TGT], 0.25).to(E.device); P.mr = top_mask(amap[N_TGT:], 0.25).to(E.device)
    td = E.encode_text(r.detailed_prompt) if rare else None
    x = st["x"]; G, DH, HT = [], [], []
    for i, t in enumerate(st["ts"][:2]):
        P.cap = {}; v = E.velocity(x, t, tv, st); c = P.cap
        G.append(torch.stack([c[l][1] - c[l][0] for l in P.layers])); HT.append(torch.stack([c[l][0] for l in P.layers]))
        if rare:
            P.cap = {}; E.velocity(x, t, td, st)
            DH.append(torch.stack([P.cap[l][0] - c[l][0] for l in P.layers]))
        P.cap = None
        x = (x.float() + (st["sigmas"][i + 1] - st["sigmas"][i]) * v.float()).to(v.dtype)
    torch.save(dict(g=torch.stack(G), ht=torch.stack(HT), dh=torch.stack(DH) if rare else None, amap=amap.half(), layers=P.layers), f)
    if n % 40 == 0:
        print(n, len(jobs), flush=True)
print("done")
