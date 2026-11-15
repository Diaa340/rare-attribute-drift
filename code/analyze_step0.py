"""Summarize exp_step0.py: which steps carry the correction, how localized dv0 is, and whether dv0 shares a direction."""
import glob

import numpy as np
import pandas as pd
import torch

pd.set_option("display.width", 220)
r = pd.read_parquet("step0/results.parquet")
print(len(r), "cases;", (r.p_rare_vanilla < 0.2).sum(), "still drift in our regeneration")
r = r[r.p_rare_vanilla < 0.2]
for n in ["vanilla", "detailed", "det_step0_only", "det_steps0-1", "det_steps1-3"]:
    print(f"{n:16s} kept={(r['p_rare_' + n] > 0.5).mean():.2f}  edit applied={(r['p_edit_' + n] > 0.5).mean():.2f}")
print(r[[c for c in r if c.startswith("rel_dv")] + ["top10pct_energy"]].describe().round(3).loc[["mean", "50%", "min", "max"]])
print(pd.crosstab(r.flag, [(r.p_rare_det_step0_only > 0.5).rename("fix@0"), (r["p_rare_det_steps1-3"] > 0.5).rename("fix@1-3")]))

keep = set(zip(r.id, r.edit_seed.astype(str)))
D, ids = [], []
for f in sorted(glob.glob("step0/dv0/*.pt")):
    rid, seed = f.split("/")[-1][:-3].rsplit("_", 1)
    if (rid, seed) not in keep:
        continue
    d = torch.load(f)["dv0"].float()[0]                       # (1024 tokens, 128 ch)
    w = d.norm(dim=-1); w = w / w.sum()
    D.append((w[:, None] * d).sum(0)); ids.append(rid)        # energy-weighted pooled 128-d direction
D = torch.stack(D); Dn = torch.nn.functional.normalize(D, dim=-1); C = (Dn @ Dn.T).numpy()
meta = pd.read_json("data/metadata.jsonl", lines=True).drop_duplicates("id").set_index("id")
fl = meta.loc[ids, "flag"].values; ra = meta.loc[ids, "rare_attribute"].values; ty = meta.loc[ids, "typical_attribute"].values
off = ~np.eye(len(ids), dtype=bool)
same = lambda a: off & (a[:, None] == a[None])
print("pooled dv0 cosine | all", C[off].mean().round(3), "| same flag", C[same(fl)].mean().round(3),
      "| same rare attr", C[same(ra)].mean().round(3) if same(ra).any() else "-",
      "| same rare+typical", C[same(ra) & same(ty)].mean().round(3) if (same(ra) & same(ty)).any() else "-")
S = torch.linalg.svdvals(D - D.mean(0)); print("pooled dv0 PCA explained:", (S ** 2 / (S ** 2).sum())[:5].numpy().round(3))
print("norm of mean / mean norm:", float(D.mean(0).norm() / D.norm(dim=-1).mean()))
