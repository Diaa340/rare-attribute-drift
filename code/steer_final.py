"""Steering with a label-free subject mask (attention from target tokens to the subject word, blocks S13-S19, step 0).

Methods (all at steps 0-1, alpha 4):
  vel_attnF  velocity steering, v += alpha * d_v on the top-F attention tokens
  hid_attnF  hidden steering, h_l += alpha * d_h,l at S4-S12 on the top-F attention tokens
  hid_full   hidden steering at S4-S12 on all target tokens (for attributes that live in the background, e.g. place)
Directions per rare attribute. --mode dev: learned on other dev subjects (or other pairs when the attribute has one
subject). --mode test: learned on all dev rows, evaluated on test rows (dataset seed, <= per_attr per attribute).
Usage: python steer_final.py --mode dev|test --shard 0 --nshards 2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from attn_mask import AttnProbe, subject_token_ids, top_mask
from hidden import Hooks
from judge import Judge
from velocity import Editor

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="dev"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--per_attr", type=int, default=12); ap.add_argument("--alpha", type=float, default=4.0)
a = ap.parse_args()
out = Path(f"final_{a.mode}"); (out / "img").mkdir(parents=True, exist_ok=True)

d = pd.read_parquet("drift_table_dev.parquet"); d = d[d.fixed_detailed].reset_index(drop=True)
DV = {(r.id, r.edit_seed): torch.load(f"dv/{r.id}_{r.edit_seed}.pt")["pooled_mask"][:2].float() for r in d.itertuples()}  # (2, 128)
DH = {(r.id, r.edit_seed): torch.load(f"dh/{r.id}_{r.edit_seed}.pt")["dh"].float() for r in d.itertuples()}              # (2, 25, 3072)
LAYERS = torch.load(f"dh/{d.id[0]}_{d.edit_seed[0]}.pt")["layers"]
STEER_LAYERS = [f"S{i}" for i in range(4, 13)]
n_subj = d.groupby("rare_attribute").subject.nunique()

if a.mode == "dev":
    ev = pd.concat([g.sample(min(len(g), a.per_attr), random_state=0) for _, g in d[d.seed_kind == "dataset"].groupby("rare_attribute")])
else:
    m = pd.read_json("data/metadata.jsonl", lines=True)
    t = m[(m.split == "test") & (m["view"] == "rare") & (m.seed_kind == "dataset") & m.rare_attribute.isin(set(d.rare_attribute))]
    ev = pd.concat([g.sample(min(len(g), a.per_attr), random_state=0) for _, g in t.groupby("rare_attribute")])
ev = ev.iloc[a.shard::a.nshards]


def train_rows(r):
    if a.mode == "test":
        return d[d.rare_attribute == r.rare_attribute]
    excl = "subject" if n_subj[r.rare_attribute] > 1 else "pair_id"
    return d[(d.rare_attribute == r.rare_attribute) & (d[excl] != getattr(r, excl))]


METHODS = ["baseline", "vel_attn10", "vel_attn25", "hid_attn10", "hid_attn25", "hid_full"]
E = Editor(); H = Hooks(E.pipe.transformer); P = AttnProbe(E.pipe.transformer, range(13, 20)); J = Judge(); rows = []
for n, r in enumerate(ev.itertuples()):
    tr = train_rows(r)
    if len(tr) == 0:
        continue
    Dv = torch.stack([DV[(x.id, x.edit_seed)] for x in tr.itertuples()]).mean(0)
    Dh = torch.stack([DH[(x.id, x.edit_seed)] for x in tr.itertuples()]).mean(0)
    src = Image.open("data/" + r.source_image).convert("RGB")
    st = E.prepare(src, int(r.edit_seed)); tv = E.encode_text(r.edit_prompt)
    # label-free mask from one unsteered forward pass at step 0
    P.ids, P.maps = subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject), {}
    E.velocity(st["x"], st["ts"][0], tv, st); P.ids = None
    amap = torch.stack(list(P.maps.values())).mean(0) if P.maps else torch.ones(1024)
    masks = {"10": top_mask(amap, 0.10), "25": top_mask(amap, 0.25), "full": torch.ones(1024, dtype=torch.bool)}
    imgs, names = [], []
    for meth in METHODS:
        kind, mk = (meth.split("_attn") + [None])[:2] if "_attn" in meth else (meth.replace("_full", ""), "full")
        mask = masks.get(mk) if meth != "baseline" else None
        x = st["x"]
        for i, t in enumerate(st["ts"]):
            H.add, H.mask = None, None
            if kind == "hid" and i < 2:
                H.add = {l: a.alpha * Dh[i, LAYERS.index(l)] for l in STEER_LAYERS}; H.mask = mask
            v = E.velocity(x, t, tv, st)
            if kind == "vel" and i < 2:
                v = v.clone(); v[0, mask.to(v.device)] += (a.alpha * Dv[i]).to(v.device, v.dtype)
            x = (x.float() + (st["sigmas"][i + 1] - st["sigmas"][i]) * v.float()).to(v.dtype)
        H.add, H.mask = None, None
        names.append(meth); imgs.append(E.decode(x, st))
    xd, _ = E.run(st, E.encode_text(r.detailed_prompt)); names.append("detailed"); imgs.append(E.decode(xd, st))
    p = J.attr(imgs, [r.subject] * len(imgs), [r.rare_attribute] * len(imgs), [r.typical_attribute] * len(imgs))
    pe = J.edit([src] * len(imgs), imgs, [r.edit_prompt] * len(imgs))
    base = np.asarray(imgs[0]).astype(float); big = masks["25"].reshape(32, 32).numpy().repeat(16, 0).repeat(16, 1)
    for k, (nm, im) in enumerate(zip(names, imgs)):
        diff = np.abs(np.asarray(im).astype(float) - base).mean(-1)
        im.resize((256, 256)).save(out / "img" / f"{r.id}_{r.edit_seed}_{nm}.jpg", quality=90)
        rows.append(dict(id=r.id, edit_seed=r.edit_seed, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute,
                         flag=r.flag, edit_family=r.edit_family, method=nm, n_train=len(tr), p_rare=float(p[k]), p_edit=float(pe[k]),
                         dpix_subject=float(diff[big].mean()), dpix_background=float(diff[~big].mean())))
    pd.DataFrame(rows).to_parquet(out / f"results_{a.shard}.parquet")
    print(a.shard, n, len(ev), r.id, r.rare_attribute, flush=True)
print("done", a.shard)
