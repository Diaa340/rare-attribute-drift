"""Baselines on the same test cases as steer_final.py --mode test (hyperparameters fixed in advance, all reported).

label (prompt)    min_prompt     edit prompt + " The {subject} is {rare}."  (all steps)
label (guidance)  textguid_s2    v + 2 * (v(c_attr) - v(c)) at steps 0-1             (semantic-guidance style)
                  negguid_w1     v + 1 * (v(c) - v(c_neg)) at steps 0-1, c_neg names the typical attribute
label-free        refboost_g2/4  attention logits from subject target tokens to reference tokens += log(g),
                                 every block, steps 0-1 (stronger reliance on the reference image)
                  sdedit_k1/k2   start from the noised reference latent at step k instead of pure noise
ours              hid_attn25_a2  hidden-state steering as in steer_final.py, alpha 2 (alpha 4 is already in final_test)
Writes baselines_test/results_<shard>.parquet with p_rare, p_edit, p_defect, pixel change. Usage: --shard 0 --nshards 2
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from attn_mask import AttnProbe, subject_token_ids, top_mask
from hidden import Hooks
from judge import Judge
from velocity import Editor

ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
out = Path("baselines_test"); (out / "img").mkdir(parents=True, exist_ok=True)
N_TXT, N_TGT = 512, 1024
DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
            "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? "
            "Answer only yes or no.")                       # same question as rescore_defects.py


class RefBoost:
    """Additive attention bias in every block: subject target queries -> reference keys get + log(g)."""

    def __init__(self, transformer):
        self.bias = None
        for b in transformer.transformer_blocks:
            b.attn.set_processor(self._wrap_double(b.attn.processor))
        for b in transformer.single_transformer_blocks:
            b.attn.set_processor(self._wrap_single(b.attn.processor))

    def _wrap_double(self, proc):
        rb = self

        class W:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kw):
                m = rb.bias if rb.bias is not None else attention_mask
                return proc(attn, hidden_states, encoder_hidden_states, m, image_rotary_emb)
        return W()

    def _wrap_single(self, proc):
        rb = self

        class W:
            def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None, **kw):
                m = rb.bias if rb.bias is not None else attention_mask
                return proc(attn, hidden_states, attention_mask=m, image_rotary_emb=image_rotary_emb, **kw)
        return W()

    def make(self, mask, g, device):
        L = N_TXT + 2 * N_TGT
        B = torch.zeros(1, 1, L, L, device=device, dtype=torch.bfloat16)
        rows = N_TXT + torch.where(mask.to(device))[0]
        B[0, 0, rows.unsqueeze(1), torch.arange(N_TXT + N_TGT, L, device=device).unsqueeze(0)] = math.log(g)
        return B


d = pd.read_parquet("drift_table_dev.parquet"); d = d[d.fixed_detailed].reset_index(drop=True)
DH = {(r.id, r.edit_seed): torch.load(f"dh/{r.id}_{r.edit_seed}.pt")["dh"].float() for r in d.itertuples()}
LAYERS = torch.load(f"dh/{d.id[0]}_{d.edit_seed[0]}.pt")["layers"]; STEER = [f"S{i}" for i in range(4, 13)]
done = pd.concat([pd.read_parquet(f) for f in Path("final_test").glob("results_*.parquet")])
m = pd.read_json("data/metadata.jsonl", lines=True).set_index(["id", "edit_seed"])
cases = done[["id", "edit_seed"]].drop_duplicates().sort_values(["id", "edit_seed"]).iloc[a.shard::a.nshards]

E = Editor(); tr = E.pipe.transformer
H = Hooks(tr); P = AttnProbe(tr, range(13, 20)); RB = RefBoost(tr); J = Judge(); rows = []
METHODS = ["baseline", "min_prompt", "textguid_s2", "negguid_w1", "refboost_g2", "refboost_g4", "sdedit_k1", "sdedit_k2", "hid_attn25_a2"]
for n, (cid, seed) in enumerate(cases.itertuples(index=False)):
    r = m.loc[(cid, seed)]
    src = Image.open("data/" + r.source_image).convert("RGB")
    st = E.prepare(src, int(seed)); tv = E.encode_text(r.edit_prompt)
    t_attr = E.encode_text(f"{r.edit_prompt}. The {r.subject} is {r.rare_attribute}.")
    t_neg = E.encode_text(f"{r.edit_prompt}. The {r.subject} is {r.typical_attribute}.")
    P.ids, P.maps = subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject), {}
    E.velocity(st["x"], st["ts"][0], tv, st); P.ids = None
    mask = top_mask(torch.stack(list(P.maps.values())).mean(0), 0.25) if P.maps else torch.ones(1024, dtype=torch.bool)
    tr_rows = d[d.rare_attribute == r.rare_attribute]
    Dh = torch.stack([DH[(x.id, x.edit_seed)] for x in tr_rows.itertuples()]).mean(0) if len(tr_rows) else None
    imgs, names = [], []
    for meth in METHODS:
        if meth == "hid_attn25_a2" and Dh is None:
            continue
        start = {"sdedit_k1": 1, "sdedit_k2": 2}.get(meth, 0)
        x = st["x"]
        if start:
            s0 = st["sigmas"][start]
            x = ((1 - s0) * st["ref"].float() + s0 * st["x"].float()).to(st["x"].dtype)   # same noise, noised source latent
        for i in range(start, len(st["ts"])):
            t = st["ts"][i]; early = i < 2
            H.add, H.mask, RB.bias = None, None, None
            if meth == "hid_attn25_a2" and early:
                H.add = {l: 2.0 * Dh[i, LAYERS.index(l)] for l in STEER}; H.mask = mask
            if meth.startswith("refboost") and early:
                RB.bias = RB.make(mask, float(meth[-1]), x.device)
            v = E.velocity(x, t, t_attr if meth == "min_prompt" else tv, st)
            RB.bias = None
            if meth == "textguid_s2" and early:
                v = (v.float() + 2.0 * (E.velocity(x, t, t_attr, st).float() - v.float())).to(v.dtype)
            if meth == "negguid_w1" and early:
                v = (v.float() + 1.0 * (v.float() - E.velocity(x, t, t_neg, st).float())).to(v.dtype)
            x = (x.float() + (st["sigmas"][i + 1] - st["sigmas"][i]) * v.float()).to(v.dtype)
        H.add, H.mask = None, None
        names.append(meth); imgs.append(E.decode(x, st))
    p = J.attr(imgs, [r.subject] * len(imgs), [r.rare_attribute] * len(imgs), [r.typical_attribute] * len(imgs))
    pe = J.edit([src] * len(imgs), imgs, [r.edit_prompt] * len(imgs))
    small = [im.resize((256, 256)) for im in imgs]
    l = J._logits([(DEFECT_Q.format(subject=r.subject), [im]) for im in small])
    pdef = torch.sigmoid(torch.logsumexp(l[:, [J.ids["yes"], J.ids["Yes"]]], -1) - torch.logsumexp(l[:, [J.ids["no"], J.ids["No"]]], -1)).cpu().numpy()
    base = np.asarray(imgs[0]).astype(float)                       # unsteered regeneration, 512 px as in steer_final.py
    big = mask.reshape(32, 32).numpy().repeat(16, 0).repeat(16, 1)
    for k, (nm, im) in enumerate(zip(names, imgs)):
        diff = np.abs(np.asarray(im).astype(float) - base).mean(-1)
        small[k].save(out / "img" / f"{cid}_{seed}_{nm}.jpg", quality=90)
        rows.append(dict(id=cid, edit_seed=seed, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute, flag=r.flag,
                         method=nm, p_rare=float(p[k]), p_edit=float(pe[k]), p_defect=float(pdef[k]),
                         dpix_subject=float(diff[big].mean()), dpix_background=float(diff[~big].mean())))
    pd.DataFrame(rows).to_parquet(out / f"results_{a.shard}.parquet")
    print(a.shard, n, len(cases), cid, flush=True)
print("done", a.shard)
