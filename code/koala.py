"""KOALA edits: the plain instruction, the correction adapter (adapter.py) and the source pull of sdedit_plus2.py
(rep1_full_subj_b{b}_g{g}). 7 editor forwards, no label, no prompt change:
  step 0     one forward with the plain instruction gives v0 and the subject mask M (top 25% of the attention to the
             subject word, S13-S19); x1 = Euler step with v0 + s * delta0 (delta0 = adapter from v0 and M), then the
             pull x1 += b (1 - sigma1) M (z_ref - x0_hat) with x0_hat = x - sigma0 v0 (plain v0, not the corrected one)
  steps 1-3  v + s * delta + g (v - v(empty prompt)), delta from the plain v
Final setting s 2, b 2, g 1. Scored by judge 1 against the vanilla edit, both views of the koala_split.json subjects of
--part at one seed kind. Writes koala_<part>/results.parquet and 256 px images.
Usage: python koala.py --ckpt adapter.pt [--part val] [--seed_kind fresh]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from adapter import Adapter, positions
from attn_mask import AttnProbe, subject_token_ids, top_mask
from judge import Judge
from velocity import Editor

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--part", default="val"); ap.add_argument("--seed_kind", default="fresh")
ap.add_argument("--s", type=float, default=2.0); ap.add_argument("--b", type=float, default=2.0); ap.add_argument("--g", type=float, default=1.0)
a = ap.parse_args()
out = Path(f"koala_{a.part}"); (out / "img").mkdir(parents=True, exist_ok=True)
meta = pd.read_json("data/metadata.jsonl", lines=True)
cases = meta[meta.subject.isin(json.load(open("koala_split.json"))[a.part]) & (meta.seed_kind == a.seed_kind)]
print(len(cases), "cases", flush=True)

E = Editor(); P = AttnProbe(E.pipe.transformer, range(13, 20)); J = Judge(); t_null = E.encode_text("")
A = Adapter().to(E.device); A.load_state_dict(torch.load(a.ckpt)["state"]); pos = positions().to(E.device)


@torch.no_grad()
def koala(st, tv, ids):
    sg, ref = st["sigmas"].float(), st["ref"].float()
    P.ids, P.maps = ids, {}
    v = E.velocity(st["x"], st["ts"][0], tv, st).float(); P.ids = None
    M = top_mask(torch.stack(list(P.maps.values())).mean(0), 0.25) if P.maps else torch.ones(1024, dtype=torch.bool)
    M = M.to(ref.device).float().reshape(1, 1024, 1)
    x = st["x"]
    for i, t in enumerate(st["ts"]):
        if i:
            v = E.velocity(x, t, tv, st).float()
        x0 = x.float() - sg[i] * v                                                       # x0_hat from the plain v
        vu = v + a.s * A(ref, x0, v, M, sg[i], pos)
        if i:
            vu = vu + a.g * (v - E.velocity(x, t, t_null, st).float())                  # steps(..., w=g)
        x = (x.float() + (sg[i + 1] - sg[i]) * vu).to(st["x"].dtype)
        if i == 0:
            x = (x.float() + a.b * (1 - sg[1]) * M * (ref - x0)).to(st["x"].dtype)     # inject(1, "full", "subj", b)
    return x


rows = []
for n, r in enumerate(cases.itertuples()):
    src = Image.open("data/" + r.source_image).convert("RGB"); st = E.prepare(src, int(r.edit_seed)); tv = E.encode_text(r.edit_prompt)
    ims = [E.decode(E.run(st, tv)[0], st), E.decode(koala(st, tv, subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject)), st)]
    p = J.attr(ims, [r.subject] * 2, [r.rare_attribute] * 2, [r.typical_attribute] * 2)
    pe = J.edit([src] * 2, ims, [r.edit_prompt] * 2)
    for k, nm in enumerate(["baseline", "koala"]):
        ims[k].resize((256, 256)).save(out / "img" / f"{r.id}_{r.edit_seed}_{nm}.jpg", quality=90)
        rows.append(dict(id=r.id, edit_seed=r.edit_seed, view=r.view, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute,
                         flag=r.flag, edit_family=r.edit_family, method=nm, p_rare=float(p[k]), p_edit=float(pe[k])))
    if n % 20 == 0:
        pd.DataFrame(rows).to_parquet(out / "results.parquet"); print(n, len(cases), flush=True)
pd.DataFrame(rows).to_parquet(out / "results.parquet")
print("done")
