"""SDEdit variants on dev drift cases (vanilla prompt, no label, no training).

SDEdit keeps the rare attribute but loses the edit. Variants try to keep the first and restore the second:
  sdedit_k1 / sdedit_k2      start from the noised source at step 1 / 2 (reference points, as in baselines.py)
  s92                        on-schedule start at sigma 0.92, then the usual last steps
  k2_g1 / k2_g3              sdedit_k2 + guidance on the edit instruction: v + w (v(edit) - v(empty prompt))
  rep{k}_{band}_{region}[_b] run the vanilla trajectory, then at step k replace part of the model's x0 estimate by the source:
                             x_k += b (1 - sigma_k) M F(source - x0_hat); band F: full | lp2 | lp4 (low-pass), region M: subj | all
Writes sdedit_dev/results.parquet. Usage: python sdedit_plus.py [--per_flag 14] [--limit N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from attn_mask import AttnProbe, subject_token_ids, top_mask
from judge import Judge
from velocity import Editor

DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
            "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? "
            "Answer only yes or no.")
ap = argparse.ArgumentParser(); ap.add_argument("--per_flag", type=int, default=14); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="sdedit_dev2")
a = ap.parse_args()
out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
meta = pd.read_json("data/metadata.jsonl", lines=True); M = meta.set_index(["id", "edit_seed"])
d = pd.read_parquet("drift_table_dev.parquet"); d = d[d.drifted]
cases = pd.concat([g.sample(min(len(g), a.per_flag), random_state=0) for _, g in d.groupby("flag")])[["id", "edit_seed"]].sort_values(["id", "edit_seed"])
if a.limit:
    cases = cases.sample(a.limit, random_state=0)
print(len(cases), "dev drift cases", flush=True)

E = Editor(); tr = E.pipe.transformer; P = AttnProbe(tr, range(13, 20)); J = Judge()
t_null = E.encode_text("")


def lowpass(z, k):                                   # z (1, 1024, C) on a 32x32 grid
    g = z.float().reshape(1, 32, 32, -1).permute(0, 3, 1, 2)
    g = F.interpolate(F.avg_pool2d(g, k), size=32, mode="bilinear", align_corners=False)
    return g.permute(0, 2, 3, 1).reshape(1, 1024, -1)


@torch.no_grad()
def steps(x, sig, text, st, w=0.0):
    """Euler steps along the sigma list; returns final x and the (x, x0_hat) pair before each step."""
    trace = []
    for i in range(len(sig) - 1):
        t = (sig[i] * 1000).to(st["ts"].dtype)
        v = E.velocity(x, t, text, st).float()
        if w:
            v = v + w * (v - E.velocity(x, t, t_null, st).float())
        trace.append((x, x.float() - sig[i] * v))
        x = (x.float() + (sig[i + 1] - sig[i]) * v).to(st["x"].dtype)
    return x, trace


rows = []
for n, (cid, seed) in enumerate(cases.itertuples(index=False)):
    r = M.loc[(cid, seed)]
    src = Image.open("data/" + r.source_image).convert("RGB"); st = E.prepare(src, int(seed)); tv = E.encode_text(r.edit_prompt)
    sg = st["sigmas"].float(); ref = st["ref"].float(); eps = st["x"].float()
    if n == 0:
        print("sigmas", sg.tolist(), "ts", st["ts"].tolist(), flush=True)
    P.ids, P.maps = subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject), {}
    E.velocity(st["x"], st["ts"][0], tv, st); P.ids = None
    mask = top_mask(torch.stack(list(P.maps.values())).mean(0), 0.25) if P.maps else torch.ones(1024, dtype=torch.bool)
    msk = {"subj": mask.to(ref.device).float().reshape(1, 1024, 1), "all": torch.ones(1, 1024, 1, device=ref.device)}
    noised = lambda s: ((1 - s) * ref + s * eps).to(st["x"].dtype)
    imgs = {}
    xv, tracev = steps(st["x"], sg, tv, st); imgs["baseline"] = xv
    imgs["sdedit_k1"], _ = steps(noised(sg[1]), sg[1:], tv, st)
    imgs["min_prompt"], _ = steps(st["x"], sg, E.encode_text(f"{r.edit_prompt}. The {r.subject} is {r.rare_attribute}."), st)   # label-based reference
    imgs["detailed"], _ = steps(st["x"], sg, E.encode_text(r.detailed_prompt), st)
    m40 = top_mask(torch.stack(list(P.maps.values())).mean(0), 0.40) if P.maps else torch.ones(1024, dtype=torch.bool)
    msk["subj40"] = m40.to(ref.device).float().reshape(1, 1024, 1)

    def inject(k, band, region, b):
        xk, x0 = tracev[k][0].float(), tracev[k - 1][1]
        diff = ref - x0
        diff = {"full": diff, "lp2": lowpass(diff, 2)}[band]
        return (xk + b * (1 - sg[k]) * msk[region] * diff).to(st["x"].dtype)

    for band, region, b in [("full", "subj", 2), ("full", "subj", 3), ("full", "subj", 4), ("full", "subj", 5), ("lp2", "subj", 3), ("lp2", "subj", 5),
                            ("full", "subj40", 3), ("full", "all", 3)]:
        imgs[f"rep1_{band}_{region}_b{b}"] = steps(inject(1, band, region, b), sg[1:], tv, st)[0]
    # amplified injection + guidance on the edit instruction for the remaining steps
    imgs["rep1_full_subj_b3_g1"] = steps(inject(1, "full", "subj", 3), sg[1:], tv, st, w=1.0)[0]
    imgs["rep1_full_subj_b5_g1"] = steps(inject(1, "full", "subj", 5), sg[1:], tv, st, w=1.0)[0]
    # inject at step 1 (b3) and again at step 2 (b1), using the new trajectory's own x0 estimate
    x1 = inject(1, "full", "subj", 3); x2, tr2 = steps(x1, sg[1:3], tv, st)
    x2 = (x2.float() + (1 - sg[2]) * msk["subj"] * (ref - tr2[0][1])).to(st["x"].dtype)
    imgs["rep12_full_subj_b3"] = steps(x2, sg[2:], tv, st)[0]
    # no step skipped: source-biased start latent on the subject, all 4 steps from t=1
    for c in [0.05, 0.10, 0.15]:
        imgs[f"init_subj_c{int(c * 100):02d}"] = steps((eps + c * msk["subj"] * ref).to(st["x"].dtype), sg, tv, st)[0]
    names = list(imgs); ims = [E.decode(x, st) for x in imgs.values()]
    p = J.attr(ims, [r.subject] * len(ims), [r.rare_attribute] * len(ims), [r.typical_attribute] * len(ims))
    pe = J.edit([src] * len(ims), ims, [r.edit_prompt] * len(ims))
    small = [im.resize((256, 256)) for im in ims]
    l = torch.cat([J._logits([(DEFECT_Q.format(subject=r.subject), [im]) for im in small[i:i + 6]]) for i in range(0, len(small), 6)])
    pdef = torch.sigmoid(torch.logsumexp(l[:, [J.ids["yes"], J.ids["Yes"]]], -1) - torch.logsumexp(l[:, [J.ids["no"], J.ids["No"]]], -1)).float().cpu().numpy()
    base = np.asarray(ims[0]).astype(float)
    for k, nm in enumerate(names):
        small[k].save(out / "img" / f"{cid}_{seed}_{nm}.jpg", quality=90)
        rows.append(dict(id=cid, edit_seed=seed, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute, flag=r.flag, edit_family=r.edit_family,
                         method=nm, p_rare=float(p[k]), p_edit=float(pe[k]), p_defect=float(pdef[k]), dpix=float(np.abs(np.asarray(ims[k]).astype(float) - base).mean())))
    if n % 10 == 0:
        pd.DataFrame(rows).to_parquet(out / "results.parquet"); print(n, len(cases), flush=True)
pd.DataFrame(rows).to_parquet(out / "results.parquet")
print("done")
