"""Collect subject-pooled hidden-state differences dh_l = h_l(detailed) - h_l(vanilla) at the same x_t, steps 0 and 1,
for all 25 blocks, along the vanilla trajectory. Subject mask from collect_dv.py (top 10% |dv0|).
Saves dh/<id>_<seed>.pt with dh (2 steps, 25 layers, 3072) and hnorm (25,) = norm of the pooled h_l(vanilla).
Usage: python collect_h.py --shard 0 --nshards 2
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from hidden import Hooks
from velocity import Editor

ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
out = Path("dh"); out.mkdir(exist_ok=True)
d = pd.read_parquet("drift_table_dev.parquet"); d = d[d.fixed_detailed].reset_index(drop=True).iloc[a.shard::a.nshards]
E = Editor(); H = Hooks(E.pipe.transformer)
for n, r in enumerate(d.itertuples()):
    f = out / f"{r.id}_{r.edit_seed}.pt"
    if f.exists():
        continue
    mask = torch.load(f"dv/{r.id}_{r.edit_seed}.pt")["mask"]
    st = E.prepare(Image.open("data/" + r.source_image), int(r.edit_seed))
    tv, td = E.encode_text(r.edit_prompt), E.encode_text(r.detailed_prompt)
    x = st["x"]; dh = []; H.pool = mask
    with torch.no_grad():
        for i, t in enumerate(st["ts"][:2]):
            H.capture = {}; v = E.velocity(x, t, tv, st); hv = H.capture
            H.capture = {}; E.velocity(x, t, td, st); hd = H.capture; H.capture = None
            dh.append(torch.stack([hd[l] - hv[l] for l in H.layers]))
            s, s_next = st["sigmas"][i], st["sigmas"][i + 1]
            x = (x.float() + (s_next - s) * v.float()).to(v.dtype)
    hn = torch.stack([hv[l].norm() for l in H.layers])         # norm of the pooled vanilla hidden state, step 1
    torch.save(dict(dh=torch.stack(dh), hnorm=hn, layers=H.layers), f)
    if n % 20 == 0:
        print(a.shard, n, len(d), flush=True)
print("done", a.shard)
