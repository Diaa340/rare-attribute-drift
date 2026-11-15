"""Text-to-reference attention boost on the dev drift cases (vanilla prompt, no label, no training).

Finding behind it: patching the target-image tokens fails because every block reads the text tokens, which carry the typical
prior. Here the subject word's TEXT tokens are made to read the reference image: attention logits from the subject-word tokens
(queries) to the reference-image subject tokens (keys) get + log(g), at steps 0-1, via an additive attention mask.
Sequence in every block = [text 512, target 1024, reference 1024].
  txt_g{g}         subject-word rows -> reference subject columns, all 25 blocks
  txt_D_g{g}       same, double-stream blocks D0-D4 only
  txt_S_g{g}       same, single-stream blocks S0-S19 only
  txtall_g{g}      every prompt token -> reference subject columns, all blocks
  txt_refall_g{g}  subject-word rows -> all reference columns, all blocks
  txt_tgt_g{g}     txt_g{g} + target subject rows -> reference subject columns (g / 2), all blocks
Reference subject columns: top 25% of reference tokens by attention to the subject word (S13-S19, step 0).
Usage: python txtref.py [--limit N] [--out txtref_dev] [--check]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers.models.transformers.transformer_flux2 import apply_rotary_emb
from PIL import Image

from attn_mask import subject_token_ids, top_mask
from judge import Judge
from velocity import Editor

N_TXT, N_TGT = 512, 1024
L = N_TXT + 2 * N_TGT
DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
            "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? "
            "Answer only yes or no.")
VARIANTS = [("txt", 2), ("txt", 4), ("txt", 8), ("txt", 16), ("txt_D", 8), ("txt_S", 8), ("txtall", 4), ("txtall", 8),
            ("txt_refall", 8), ("txt_tgt", 8)]
ap = argparse.ArgumentParser(); ap.add_argument("--per_flag", type=int, default=14); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="txtref_dev"); ap.add_argument("--check", action="store_true")
a = ap.parse_args()
out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)


class Boost:
    """Wraps every block's attention processor. probe: record attention-to-subject maps (S13-S19). bias: pass an additive mask."""

    def __init__(self, tr):
        self.ids, self.maps, self.bias = None, {}, {}
        self.D = [f"D{i}" for i in range(len(tr.transformer_blocks))]; self.S = [f"S{i}" for i in range(len(tr.single_transformer_blocks))]
        for name, blk in zip(self.D + self.S, list(tr.transformer_blocks) + list(tr.single_transformer_blocks)):
            blk.attn.set_processor(self._wrap(blk.attn.processor, name))

    def _wrap(self, proc, name):
        me = self

        class W:
            # Flux2Attention (D blocks) calls processor(attn, hs, encoder_hs, attention_mask, rope);
            # Flux2ParallelSelfAttention (S blocks) calls processor(attn, hs, attention_mask, rope).
            def __call__(self, attn, hidden_states, *args, **kw):
                if name.startswith("D"):
                    enc, mask, rope = (list(args) + [None, None, None])[:3]
                else:
                    mask, rope = (list(args) + [None, None])[:2]
                enc = kw.pop("encoder_hidden_states", enc if name.startswith("D") else None)
                mask = kw.pop("attention_mask", mask); rope = kw.pop("image_rotary_emb", rope)
                if me.ids and name in ("S13", "S14", "S15", "S16", "S17", "S18", "S19"):
                    qkv = attn.to_qkv_mlp_proj(hidden_states)[..., : 3 * attn.inner_dim]
                    q, k, _ = qkv.chunk(3, dim=-1)
                    q = attn.norm_q(q.unflatten(-1, (-1, attn.head_dim))); k = attn.norm_k(k.unflatten(-1, (-1, attn.head_dim)))
                    q = apply_rotary_emb(q, rope, sequence_dim=1); k = apply_rotary_emb(k, rope, sequence_dim=1)
                    p = torch.softmax((q[:, N_TXT:].transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)).float() / attn.head_dim ** 0.5, -1)
                    me.maps[name] = p[0][:, :, me.ids].sum(-1).mean(0).cpu()            # (2048,): target then reference queries
                if name in me.bias:
                    mask = me.bias[name]
                if name.startswith("D"):
                    return proc(attn, hidden_states, enc, mask, rope, **kw)
                return proc(attn, hidden_states, mask, rope, **kw)
        return W()


meta = pd.read_json("data/metadata.jsonl", lines=True); M = meta.set_index(["id", "edit_seed"])
d = pd.read_parquet("drift_table_dev.parquet"); d = d[d.drifted]
cases = pd.concat([g.sample(min(len(g), a.per_flag), random_state=0) for _, g in d.groupby("flag")])[["id", "edit_seed"]].sort_values(["id", "edit_seed"])
if a.limit:
    cases = cases.sample(a.limit, random_state=0)
print(len(cases), "dev drift cases", flush=True)
E = Editor(); B = Boost(E.pipe.transformer); dev = E.device
J = None if a.check else Judge()


def make_bias(rows, cols, g):
    m = torch.zeros(1, 1, L, L, device=dev, dtype=torch.bfloat16)
    m[0, 0, rows[:, None], cols[None, :]] = math.log(g)
    return m


def run(st, tv, bias=None):
    x = st["x"]
    for i, t in enumerate(st["ts"]):
        B.bias = bias if (bias and i < 2) else {}
        v = E.velocity(x, t, tv, st)
        x = (x.float() + (st["sigmas"][i + 1] - st["sigmas"][i]) * v.float()).to(v.dtype)
    B.bias = {}
    return x


rows = []
for n, (cid, seed) in enumerate(cases.itertuples(index=False)):
    r = M.loc[(cid, seed)]
    src = Image.open("data/" + r.source_image).convert("RGB"); st = E.prepare(src, int(seed)); tv = E.encode_text(r.edit_prompt)
    ids = subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject)
    B.ids, B.maps = ids, {}
    E.velocity(st["x"], st["ts"][0], tv, st); B.ids = None
    amap = torch.stack(list(B.maps.values())).mean(0) if B.maps else torch.ones(2 * N_TGT)
    tgt_s = top_mask(amap[:N_TGT], 0.25).nonzero()[:, 0].to(dev) + N_TXT
    ref_s = top_mask(amap[N_TGT:], 0.25).nonzero()[:, 0].to(dev) + N_TXT + N_TGT
    ref_all = torch.arange(N_TXT + N_TGT, L, device=dev)
    sw = torch.tensor(ids if ids else [0], device=dev)
    n_txt = int(tv[0].shape[1]); prompt = torch.arange(0, max(sw.max().item() + 3, 1), device=dev)   # tokens up to just past the subject word
    if a.check:                                                   # does the mask reach the attention at all?
        v0 = E.velocity(st["x"], st["ts"][0], tv, st).float()
        B.bias = {k: torch.zeros(1, 1, L, L, device=dev, dtype=torch.bfloat16) for k in B.D + B.S}; vz = E.velocity(st["x"], st["ts"][0], tv, st).float()
        B.bias = {k: make_bias(sw, ref_s, 1e4) for k in B.D + B.S}; vb = E.velocity(st["x"], st["ts"][0], tv, st).float(); B.bias = {}
        print("check: |v(zero mask) - v| / |v| =", round(float((vz - v0).norm() / v0.norm()), 5), "| |v(boost 1e4) - v| / |v| =", round(float((vb - v0).norm() / v0.norm()), 4),
              "| text length", n_txt, "| subject ids", ids, flush=True)
        continue
    imgs = {"baseline": run(st, tv)}
    for kind, g in VARIANTS:
        blocks = B.D if kind == "txt_D" else B.S if kind == "txt_S" else B.D + B.S
        if kind == "txtall":
            m = make_bias(prompt, ref_s, g)
        elif kind == "txt_refall":
            m = make_bias(sw, ref_all, g)
        else:
            m = make_bias(sw, ref_s, g)
            if kind == "txt_tgt":
                m[0, 0, tgt_s[:, None], ref_s[None, :]] = math.log(g / 2)
        imgs[f"{kind}_g{g}"] = run(st, tv, {k: m for k in blocks})
    names = list(imgs); ims = [E.decode(x, st) for x in imgs.values()]; small = [im.resize((256, 256)) for im in ims]
    p = J.attr(ims, [r.subject] * len(ims), [r.rare_attribute] * len(ims), [r.typical_attribute] * len(ims))
    pe = J.edit([src] * len(ims), ims, [r.edit_prompt] * len(ims))
    l = torch.cat([J._logits([(DEFECT_Q.format(subject=r.subject), [im]) for im in small[i:i + 6]]) for i in range(0, len(small), 6)])
    pdef = torch.sigmoid(torch.logsumexp(l[:, [J.ids["yes"], J.ids["Yes"]]], -1) - torch.logsumexp(l[:, [J.ids["no"], J.ids["No"]]], -1)).float().cpu().numpy()
    base = np.asarray(ims[0]).astype(float)
    for j, nm in enumerate(names):
        small[j].save(out / "img" / f"{cid}_{seed}_{nm}.jpg", quality=90)
        rows.append(dict(id=cid, edit_seed=seed, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute, flag=r.flag, edit_family=r.edit_family,
                         method=nm, p_rare=float(p[j]), p_edit=float(pe[j]), p_defect=float(pdef[j]),
                         dpix=float(np.abs(np.asarray(ims[j]).astype(float) - base).mean())))
    print(n, len(cases), cid, r.rare_attribute, "| subject ids", ids, flush=True)
    if n % 10 == 0:
        pd.DataFrame(rows).to_parquet(out / "results.parquet")
if rows:
    pd.DataFrame(rows).to_parquet(out / "results.parquet")
print("done")
