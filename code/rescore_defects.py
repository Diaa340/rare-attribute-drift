"""Add a defect score to steer_final.py results: P(yes) that the subject has visual defects (erased features, smeared /
washed-out regions, fragmented parts). Scored on the saved 256 px images for every method, incl. baseline and detailed.
Usage: python rescore_defects.py dev|test
"""
import glob
import sys

import pandas as pd
import torch
from PIL import Image

from judge import Judge

DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
            "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? "
            "Answer only yes or no.")

mode = sys.argv[1]
J = Judge()
for f in sorted(glob.glob(f"final_{mode}/results_*.parquet")):
    r = pd.read_parquet(f); out = []
    for i in range(0, len(r), 16):
        b = r.iloc[i:i + 16]
        ims = [Image.open(f"final_{mode}/img/{x.id}_{x.edit_seed}_{x.method}.jpg").convert("RGB") for x in b.itertuples()]
        l = J._logits([(DEFECT_Q.format(subject=s), [im]) for s, im in zip(b.subject, ims)])
        y = torch.logsumexp(l[:, [J.ids["yes"], J.ids["Yes"]]], -1); n = torch.logsumexp(l[:, [J.ids["no"], J.ids["No"]]], -1)
        out += torch.sigmoid(y - n).cpu().tolist()
    r["p_defect"] = out
    r.to_parquet(f.replace("results_", "defects_")); print(f, "done", flush=True)
