"""Re-score the dev SDEdit-sweep images with the second judge. Writes sdedit_dev2/judge2.parquet. Usage: python score2_sdedit.py [--limit N]"""
import argparse
import pandas as pd
from PIL import Image

from judge2 import Judge2

ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
KEEP = ["baseline", "rep1_full_subj_b2", "rep1_full_subj_b3", "rep1_full_subj_b3_g1", "rep1_full_subj_b4", "rep1_full_subj_b5", "rep1_full_subj_b5_g1",
        "sdedit_k1", "min_prompt", "detailed"]
r = pd.read_parquet("sdedit_dev2/results.parquet"); r = r[r.method.isin(KEEP)].sort_values(["id", "edit_seed", "method"]).reset_index(drop=True)
meta = pd.read_json("data/metadata.jsonl", lines=True).set_index(["id", "edit_seed"])
keys = list(r.groupby(["id", "edit_seed"]).groups)
if a.limit:
    keys = keys[:a.limit]
J = Judge2(); out, srows = [], []
for n, (cid, seed) in enumerate(keys):
    g = r[(r.id == cid) & (r.edit_seed == seed)].copy(); m = meta.loc[(cid, seed)]; k = len(g)
    src = Image.open("data/" + m.source_image).convert("RGB"); src.thumbnail((448, 448))
    ims = [Image.open(f"sdedit_dev2/img/{cid}_{seed}_{x}.jpg").convert("RGB") for x in g.method]
    g["p_rare2"] = J.attr(ims, [m.subject] * k, [m.rare_attribute] * k, [m.typical_attribute] * k)
    g["p_edit2"] = J.edit([src] * k, ims, [m.edit_prompt] * k)
    g["p_defect2"] = J.defect(ims, [m.subject] * k)
    g["p_same2"] = J.same([src] * k, ims, [m.subject] * k)
    out.append(g)
    srows.append(dict(id=cid, edit_seed=seed, p_rare2_source=float(J.attr([src], [m.subject], [m.rare_attribute], [m.typical_attribute])[0])))
    if n % 10 == 0:
        pd.concat(out).to_parquet("sdedit_dev2/judge2.parquet"); print(n, len(keys), flush=True)
pd.concat(out).to_parquet("sdedit_dev2/judge2.parquet"); pd.DataFrame(srows).to_parquet("sdedit_dev2/judge2_source.parquet")
print("done")
