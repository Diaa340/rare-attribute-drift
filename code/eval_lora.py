"""Evaluate a velocity-matching LoRA checkpoint (active at steps 0-1 only, vanilla prompt, no label).

--mode val     drifted dev rows of the held-out validation subjects (both seeds) + their common views
--mode test    the 285 held-out test cases used for steering and baselines + 120 of their common views
--mode unseen  test rare rows (dataset seed) whose attribute never appears in dev; also runs the two prompt baselines
Writes eval_lora/<mode>_<tag>.parquet and 256 px images. Usage: python eval_lora.py --ckpt lora_r16/ckpt_3000.pt --mode val --tag 3000
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from judge import Judge
from lora import add_lora, load_state, set_enabled
from velocity import Editor

DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
            "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? "
            "Answer only yes or no.")                       # same question as rescore_defects.py

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--mode", default="val"); ap.add_argument("--tag", default="x")
ap.add_argument("--rank", type=int, default=16)
a = ap.parse_args()
out = Path("eval_lora"); (out / "img").mkdir(parents=True, exist_ok=True)
meta = pd.read_json("data/metadata.jsonl", lines=True)
M = meta.set_index(["id", "edit_seed"])
d = pd.read_parquet("drift_table_dev.parquet")

if a.mode == "val":
    vs = json.load(open("lora_val_subjects.json"))
    rare = d[d.drifted & d.subject.isin(vs)][["id", "edit_seed"]]
elif a.mode == "test":
    rare = pd.concat([pd.read_parquet(f) for f in glob.glob("final_test/results_*.parquet")])[["id", "edit_seed"]].drop_duplicates()
else:
    t = meta[(meta.split == "test") & (meta["view"] == "rare") & (meta.seed_kind == "dataset")]
    rare = t[~t.rare_attribute.isin(set(d[d.fixed_detailed].rare_attribute))][["id", "edit_seed"]]
rare = rare.sort_values(["id", "edit_seed"]).reset_index(drop=True)
common = rare.assign(id=rare.id.str.replace("_rare", "_standard"))
if a.mode == "test":
    common = common.sample(120, random_state=0)
cases = [(i, s, "rare") for i, s in rare.itertuples(index=False)] + ([] if a.mode == "unseen" else [(i, s, "common") for i, s in common.itertuples(index=False)])
print(a.mode, len(rare), "rare +", len(cases) - len(rare), "common cases", flush=True)

E = Editor(); mods = add_lora(E.pipe.transformer, r=a.rank, alpha=a.rank); load_state(mods, torch.load(a.ckpt)); J = Judge()


@torch.no_grad()
def gen(st, text, lora_steps=()):
    x = st["x"]
    for i, t in enumerate(st["ts"]):
        set_enabled(mods, i in lora_steps)
        v = E.velocity(x, t, text, st)
        x = (x.float() + (st["sigmas"][i + 1] - st["sigmas"][i]) * v.float()).to(v.dtype)
    set_enabled(mods, False)
    return E.decode(x, st)


rows = []
for n, (cid, seed, view) in enumerate(cases):
    r = M.loc[(cid, seed)]
    src = Image.open("data/" + r.source_image).convert("RGB"); st = E.prepare(src, int(seed)); tv = E.encode_text(r.edit_prompt)
    imgs = {"baseline": gen(st, tv), "lora": gen(st, tv, (0, 1))}
    if a.mode == "unseen":
        imgs["min_prompt"] = gen(st, E.encode_text(f"{r.edit_prompt}. The {r.subject} is {r.rare_attribute}."))
        imgs["detailed"] = gen(st, E.encode_text(r.detailed_prompt))
    names, ims = list(imgs), list(imgs.values())
    p = J.attr(ims, [r.subject] * len(ims), [r.rare_attribute] * len(ims), [r.typical_attribute] * len(ims))
    pe = J.edit([src] * len(ims), ims, [r.edit_prompt] * len(ims))
    small = [im.resize((256, 256)) for im in ims]
    l = J._logits([(DEFECT_Q.format(subject=r.subject), [im]) for im in small])
    pdef = torch.sigmoid(torch.logsumexp(l[:, [J.ids["yes"], J.ids["Yes"]]], -1) - torch.logsumexp(l[:, [J.ids["no"], J.ids["No"]]], -1)).cpu().numpy()
    base = np.asarray(ims[0]).astype(float)
    for k, nm in enumerate(names):
        small[k].save(out / "img" / f"{a.mode}_{a.tag}_{cid}_{seed}_{nm}.jpg", quality=90)
        rows.append(dict(id=cid, edit_seed=seed, view=view, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute, flag=r.flag,
                         edit_family=r.edit_family, method=nm, p_rare=float(p[k]), p_edit=float(pe[k]), p_defect=float(pdef[k]),
                         dpix=float(np.abs(np.asarray(ims[k]).astype(float) - base).mean())))
    if n % 20 == 0:
        pd.DataFrame(rows).to_parquet(out / f"{a.mode}_{a.tag}.parquet"); print(n, len(cases), flush=True)
pd.DataFrame(rows).to_parquet(out / f"{a.mode}_{a.tag}.parquet")
print("done")
