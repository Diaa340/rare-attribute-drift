"""Reference guidance on the dev drift cases (vanilla prompt, no label, no training).

The model reads the rare attribute from the reference image but lets the text prior win. Guidance amplifies what the reference
contributes: v <- v + w * M * (v(prompt, source) - v(prompt, contrast)) at the first `k` steps, with the contrast branch
  noref   no reference image at all (text-to-image with the same prompt: the prior)
  drift   the model's own vanilla output as the reference (same edit, typical attribute)
  gray    the grayscale source as the reference (isolates colour)
M: subj = top-25% subject tokens (attention mask) | all = every token.
Names: {contrast}_{region}_w{w}_k{k}. Writes <out>/results.parquet. Usage: python refguid.py [--limit N] [--out refguid_dev]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from attn_mask import AttnProbe, subject_token_ids, top_mask
from judge import Judge
from velocity import Editor

DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
            "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? "
            "Answer only yes or no.")
VARIANTS = [("noref", "subj", 0.5, 2), ("noref", "subj", 1, 2), ("noref", "subj", 2, 2), ("noref", "subj", 3, 2), ("noref", "all", 1, 2), ("noref", "all", 2, 2),
            ("noref", "subj", 2, 1), ("noref", "subj", 4, 1), ("drift", "subj", 1, 2), ("drift", "subj", 2, 2), ("drift", "subj", 4, 2), ("gray", "subj", 2, 2)]
ap = argparse.ArgumentParser(); ap.add_argument("--per_flag", type=int, default=14); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="refguid_dev")
a = ap.parse_args()
out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
meta = pd.read_json("data/metadata.jsonl", lines=True); M = meta.set_index(["id", "edit_seed"])
d = pd.read_parquet("drift_table_dev.parquet"); d = d[d.drifted]
cases = pd.concat([g.sample(min(len(g), a.per_flag), random_state=0) for _, g in d.groupby("flag")])[["id", "edit_seed"]].sort_values(["id", "edit_seed"])
if a.limit:
    cases = cases.sample(a.limit, random_state=0)
print(len(cases), "dev drift cases", flush=True)
E = Editor(); P = AttnProbe(E.pipe.transformer, range(13, 20)); J = Judge()


@torch.no_grad()
def vel_noref(x, t, text, st):
    emb, ids = text
    return E.pipe.transformer(hidden_states=x.to(torch.bfloat16), timestep=t.expand(1).to(x.dtype) / 1000, guidance=None, encoder_hidden_states=emb,
                              txt_ids=ids, img_ids=st["x_ids"], return_dict=False)[0]


@torch.no_grad()
def run(st, tv, neg=None, w=0.0, mask=None, upto=0):
    x = st["x"]; sg = st["sigmas"]; rel = []
    for i, t in enumerate(st["ts"]):
        v = E.velocity(x, t, tv, st).float()
        if neg is not None and i < upto:
            dv = v - neg(x, t).float(); rel.append(float((mask * dv).norm() / (mask * v).norm()))
            v = v + w * mask * dv
        x = (x.float() + (sg[i + 1] - sg[i]) * v).to(st["x"].dtype)
    return x, rel


rows = []
for n, (cid, seed) in enumerate(cases.itertuples(index=False)):
    r = M.loc[(cid, seed)]
    src = Image.open("data/" + r.source_image).convert("RGB"); st = E.prepare(src, int(seed)); tv = E.encode_text(r.edit_prompt)
    P.ids, P.maps = subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject), {}
    E.velocity(st["x"], st["ts"][0], tv, st); P.ids = None
    m = top_mask(torch.stack(list(P.maps.values())).mean(0), 0.25) if P.maps else torch.ones(1024, dtype=torch.bool)
    msk = {"subj": m.to(E.device).float().reshape(1, 1024, 1), "all": torch.ones(1, 1024, 1, device=E.device)}
    imgs = {}; rels = {}
    imgs["baseline"], _ = run(st, tv); base_img = E.decode(imgs["baseline"], st)
    alt = {k: E.prepare(im, int(seed)) for k, im in [("drift", base_img), ("gray", src.convert("L").convert("RGB"))]}
    neg = {"noref": lambda x, t: vel_noref(x, t, tv, st),
           "drift": lambda x, t: E.velocity(x, t, tv, {**st, "ref": alt["drift"]["ref"], "ref_ids": alt["drift"]["ref_ids"]}),
           "gray": lambda x, t: E.velocity(x, t, tv, {**st, "ref": alt["gray"]["ref"], "ref_ids": alt["gray"]["ref_ids"]})}
    for kind, region, w, k in VARIANTS:
        nm = f"{kind}_{region}_w{w:g}_k{k}"; imgs[nm], rels[nm] = run(st, tv, neg[kind], float(w), msk[region], k)
    names = list(imgs); ims = [base_img] + [E.decode(x, st) for x in list(imgs.values())[1:]]; small = [im.resize((256, 256)) for im in ims]
    p = J.attr(ims, [r.subject] * len(ims), [r.rare_attribute] * len(ims), [r.typical_attribute] * len(ims))
    pe = J.edit([src] * len(ims), ims, [r.edit_prompt] * len(ims))
    l = torch.cat([J._logits([(DEFECT_Q.format(subject=r.subject), [im]) for im in small[i:i + 6]]) for i in range(0, len(small), 6)])
    pdef = torch.sigmoid(torch.logsumexp(l[:, [J.ids["yes"], J.ids["Yes"]]], -1) - torch.logsumexp(l[:, [J.ids["no"], J.ids["No"]]], -1)).float().cpu().numpy()
    base = np.asarray(ims[0]).astype(float)
    for j, nm in enumerate(names):
        small[j].save(out / "img" / f"{cid}_{seed}_{nm}.jpg", quality=90)
        rows.append(dict(id=cid, edit_seed=seed, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute, flag=r.flag, edit_family=r.edit_family,
                         method=nm, p_rare=float(p[j]), p_edit=float(pe[j]), p_defect=float(pdef[j]),
                         dpix=float(np.abs(np.asarray(ims[j]).astype(float) - base).mean()), rel0=(rels.get(nm) or [np.nan])[0]))
    print(n, len(cases), cid, r.rare_attribute, "| |dv|/|v| step 0:", {k: round(v[0], 2) for k, v in rels.items() if k in ("noref_subj_w1_k2", "drift_subj_w1_k2", "gray_subj_w2_k2")}, flush=True)
    if n % 10 == 0:
        pd.DataFrame(rows).to_parquet(out / "results.parquet")
pd.DataFrame(rows).to_parquet(out / "results.parquet"); print("done")
