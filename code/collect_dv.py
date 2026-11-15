"""Collect dv_k = v(x_k, detailed) - v(x_k, vanilla) along the vanilla trajectory, k = 0..3, for every dev drift case
that the detailed prompt fixed (drift_table_dev.parquet). Saves the full per-token dv for steps 0-1, a subject mask
(top 10% tokens of |dv0|), and masked / global pooled 128-d vectors for every step.
Usage: python collect_dv.py --shard 0 --nshards 2
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from velocity import Editor

ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
out = Path("dv"); out.mkdir(exist_ok=True)
d = pd.read_parquet("drift_table_dev.parquet")
d = d[d.fixed_detailed].reset_index(drop=True)
d = d.iloc[a.shard::a.nshards]
E = Editor()
for n, (_, r) in enumerate(d.iterrows()):
    f = out / f"{r.id}_{r.edit_seed}.pt"
    if f.exists():
        continue
    st = E.prepare(Image.open("data/" + r.source_image), int(r.edit_seed))
    _, rec = E.run(st, E.encode_text(r.edit_prompt), probe_texts={"det": E.encode_text(r.detailed_prompt)})
    dv = [(rr["v_det"] - rr["v"])[0] for rr in rec]                       # 4 x (1024, 128)
    e0 = dv[0].norm(dim=-1); mask = e0 >= e0.quantile(0.9)
    torch.save(dict(dv0=dv[0].half(), dv1=dv[1].half(), mask=mask,
                    pooled_mask=torch.stack([x[mask].mean(0) for x in dv]),          # (4, 128)
                    pooled_all=torch.stack([x.mean(0) for x in dv]),                 # (4, 128)
                    vnorm=torch.tensor([rr["v"][0].norm(dim=-1).mean() for rr in rec])), f)
    if n % 20 == 0:
        print(a.shard, n, len(d), flush=True)
print("done", a.shard)
