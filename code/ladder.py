"""How much spatial structure does a correction need? Oracle ladder on the dev drift cases (analysis, not a deployable method).

At steps 0-1, blocks S4-S12, the vanilla run is pulled toward the detailed-prompt run at the same x_t:
    h_l <- h_l + a * T(h_l(detailed) - h_l)          (h_l(detailed) from one extra forward pass per step)
T keeps different amounts of structure of the token-wise difference (1024 tokens x 3072):
  tok_all / tok_subj      everything, on all tokens / on the top-25% subject tokens       (exact patching when a = 1)
  svdK_all / svdK_subj    best rank-K approximation: K directions, each with its own spatial weights
  grid4_all               mean inside each 4x4 cell of the 32x32 token grid (64 regions)
  pool_subj / pool_all    one vector: the mean over the subject tokens / all tokens          (what a steering vector can do)
Names: {T}_a{a}. Writes <out>/results.parquet. Usage: python ladder.py [--limit N] [--out ladder_dev]
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

N_TGT = 1024
DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
            "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? "
            "Answer only yes or no.")
VARIANTS = [("tok_all", 1), ("tok_subj", 1), ("svd16_all", 1), ("svd4_all", 1), ("svd1_all", 1), ("svd1_all", 2), ("svd4_subj", 1), ("svd1_subj", 1), ("svd1_subj", 2),
            ("grid4_all", 1), ("pool_subj", 1), ("pool_subj", 2), ("pool_subj", 4), ("pool_all", 2)]
LAYERS = [f"S{i}" for i in range(4, 13)]
ap = argparse.ArgumentParser(); ap.add_argument("--per_flag", type=int, default=14); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="ladder_dev")
a = ap.parse_args()
out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
meta = pd.read_json("data/metadata.jsonl", lines=True); M = meta.set_index(["id", "edit_seed"])
d = pd.read_parquet("drift_table_dev.parquet"); d = d[d.drifted]
cases = pd.concat([g.sample(min(len(g), a.per_flag), random_state=0) for _, g in d.groupby("flag")])[["id", "edit_seed"]].sort_values(["id", "edit_seed"])
if a.limit:
    cases = cases.sample(a.limit, random_state=0)
print(len(cases), "dev drift cases", flush=True)


class Pull:
    """Forward hooks on S4-S12: store the target-token states of a pass, or pull the current pass toward the stored ones."""

    def __init__(self, tr):
        self.store = None; self.ref = {}; self.kind = None; self.alpha = 1.0; self.idx = None; self.frac = []
        for i in range(4, 13):
            tr.single_transformer_blocks[i].register_forward_hook(self._hook(f"S{i}"))

    def _reduce(self, D):                                           # D (1024, 3072) float32
        k = self.kind
        if k == "tok_all":
            return D
        Z = torch.zeros_like(D)
        if k == "tok_subj":
            Z[self.idx] = D[self.idx]; return Z
        if k == "pool_subj":
            Z[self.idx] = D[self.idx].mean(0); return Z
        if k == "pool_all":
            return D.mean(0, keepdim=True).expand_as(D)
        if k == "grid4_all":
            g = D.reshape(8, 4, 8, 4, -1).mean((1, 3), keepdim=True).expand(8, 4, 8, 4, D.shape[-1]); return g.reshape(N_TGT, -1)
        r = int(k.split("_")[0][3:]); X = D if k.endswith("_all") else D[self.idx]
        U, S_, V = torch.svd_lowrank(X, q=r + 6, niter=3); A = (U[:, :r] * S_[:r]) @ V[:, :r].T
        if k.endswith("_all"):
            return A
        Z[self.idx] = A; return Z

    def _hook(self, name):
        def fn(module, inp, o):
            s = o.shape[1] - 2 * N_TGT
            if self.store:
                self.ref[name] = o[0, s:s + N_TGT].clone()
            elif self.kind is not None:
                D = self.ref[name].float() - o[0, s:s + N_TGT].float(); R = self._reduce(D)
                self.frac.append(float(R.norm() / D.norm()))
                o[0, s:s + N_TGT] += (self.alpha * R).to(o.dtype)
            return o
        return fn


E = Editor(); P = AttnProbe(E.pipe.transformer, range(13, 20)); H = Pull(E.pipe.transformer); J = Judge()


@torch.no_grad()
def run(st, tv, td=None, kind=None, alpha=1.0):
    x = st["x"]; sg = st["sigmas"]; H.frac = []
    for i, t in enumerate(st["ts"]):
        if kind is not None and i < 2:
            H.store = True; E.velocity(x, t, td, st); H.store = False
            H.kind, H.alpha = kind, alpha
        v = E.velocity(x, t, tv, st).float(); H.kind = None
        x = (x.float() + (sg[i + 1] - sg[i]) * v).to(st["x"].dtype)
    return x, float(np.mean(H.frac)) if H.frac else np.nan


rows = []
for n, (cid, seed) in enumerate(cases.itertuples(index=False)):
    r = M.loc[(cid, seed)]
    src = Image.open("data/" + r.source_image).convert("RGB"); st = E.prepare(src, int(seed)); tv = E.encode_text(r.edit_prompt); td = E.encode_text(r.detailed_prompt)
    P.ids, P.maps = subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject), {}
    E.velocity(st["x"], st["ts"][0], tv, st); P.ids = None
    m = top_mask(torch.stack(list(P.maps.values())).mean(0), 0.25) if P.maps else torch.ones(N_TGT, dtype=torch.bool)
    H.idx = m.nonzero()[:, 0].to(E.device)
    imgs = {}; frac = {}
    imgs["baseline"], _ = run(st, tv); imgs["detailed"], _ = run(st, td)
    for kind, al in VARIANTS:
        nm = f"{kind}_a{al}"; imgs[nm], frac[nm] = run(st, tv, td, kind, float(al))
    names = list(imgs); ims = [E.decode(x, st) for x in imgs.values()]; small = [im.resize((256, 256)) for im in ims]
    p = J.attr(ims, [r.subject] * len(ims), [r.rare_attribute] * len(ims), [r.typical_attribute] * len(ims))
    pe = J.edit([src] * len(ims), ims, [r.edit_prompt] * len(ims))
    l = torch.cat([J._logits([(DEFECT_Q.format(subject=r.subject), [im]) for im in small[i:i + 6]]) for i in range(0, len(small), 6)])
    pdef = torch.sigmoid(torch.logsumexp(l[:, [J.ids["yes"], J.ids["Yes"]]], -1) - torch.logsumexp(l[:, [J.ids["no"], J.ids["No"]]], -1)).float().cpu().numpy()
    base = np.asarray(ims[0]).astype(float)
    for j, nm in enumerate(names):
        small[j].save(out / "img" / f"{cid}_{seed}_{nm}.jpg", quality=90)
        rows.append(dict(id=cid, edit_seed=seed, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute, flag=r.flag, edit_family=r.edit_family,
                         method=nm, p_rare=float(p[j]), p_edit=float(pe[j]), p_defect=float(pdef[j]),
                         dpix=float(np.abs(np.asarray(ims[j]).astype(float) - base).mean()), frac=frac.get(nm, np.nan)))
    print(n, len(cases), cid, r.rare_attribute, "| share of the difference norm kept:", {k: round(v, 2) for k, v in frac.items() if k.endswith("_a1")}, flush=True)
    if n % 10 == 0:
        pd.DataFrame(rows).to_parquet(out / "results.parquet")
pd.DataFrame(rows).to_parquet(out / "results.parquet"); print("done")
