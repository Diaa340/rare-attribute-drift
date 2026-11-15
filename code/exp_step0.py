"""Does the attribute get decided at step 0 in general?

For drift cases (vanilla drifted, detailed kept), integrate with the detailed conditioning at only some steps, judge the
result, and save the step-0 velocity difference dv0 = v(x_T, detailed) - v(x_T, vanilla) for direction analysis.
Usage: python exp_step0.py --judge judge_dev.parquet --n 40 --out step0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from judge import Judge
from velocity import Editor

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="judge_dev.parquet"); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", default="step0"); a = ap.parse_args()
out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True); (out / "dv0").mkdir(exist_ok=True)

meta = pd.read_json("data/metadata.jsonl", lines=True)
j = pd.read_parquet(a.judge).merge(meta, on=["id", "edit_seed"])
cases = j[(j["view"] == "rare") & (j.p_rare_vanilla < 0.2) & (j.p_rare_detailed > 0.8) & (j.p_edit_detailed > 0.5)]
cases = cases.sample(min(a.n, len(cases)), random_state=0)
print(len(cases), "drift cases fixed by the detailed prompt")

E = Editor(); J = Judge()
SCHEDULES = {"vanilla": "vvvv", "detailed": "dddd", "det_step0_only": "dvvv", "det_steps1-3": "vddd", "det_steps0-1": "ddvv"}
rows = []
for _, r in cases.iterrows():
    src = Image.open("data/" + r.source_image)
    st = E.prepare(src, int(r.edit_seed))
    T = {"v": E.encode_text(r.edit_prompt), "d": E.encode_text(r.detailed_prompt)}
    imgs = {}
    for name, sch in SCHEDULES.items():
        x, rec = E.run(st, T["v"], text_per_step=[T[c] for c in sch], probe_texts={"det": T["d"]} if name == "vanilla" else None)
        imgs[name] = E.decode(x, st); imgs[name].save(out / "img" / f"{r.id}_{r.edit_seed}_{name}.png")
        if name == "vanilla":
            dv0 = rec[0]["v_det"] - rec[0]["v"]
            torch.save({"dv0": dv0.half(), "v0": rec[0]["v"].half(), "x0_van": rec[0]["x0"].half(), "x0_det": rec[0]["x0_det"].half()},
                       out / "dv0" / f"{r.id}_{r.edit_seed}.pt")
            e = dv0[0].norm(dim=-1) ** 2
            top = e.sort(descending=True).values
            rel = [float((rec[i]["v_det"] - rec[i]["v"]).norm() / rec[i]["v"].norm()) for i in range(4)]
    names = list(imgs)
    p = J.attr([imgs[n] for n in names], [r.subject] * len(names), [r.rare_attribute] * len(names), [r.typical_attribute] * len(names))
    pe = J.edit([src.convert("RGB")] * len(names), [imgs[n] for n in names], [r.edit_prompt] * len(names))
    row = dict(id=r.id, edit_seed=r.edit_seed, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute, flag=r.flag,
               edit_family=r.edit_family, top10pct_energy=float(top[: len(top) // 10].sum() / top.sum()),
               **{f"rel_dv_step{i}": rel[i] for i in range(4)},
               **{f"p_rare_{n}": float(p[k]) for k, n in enumerate(names)}, **{f"p_edit_{n}": float(pe[k]) for k, n in enumerate(names)})
    rows.append(row); print(row["id"], {n: round(row[f"p_rare_{n}"], 2) for n in names}, flush=True)
    pd.DataFrame(rows).to_parquet(out / "results.parquet")
