"""Does pooled hidden-state steering need more depth than S4-S12? Dev drift cases, vanilla prompt, steps 0-1, top-25% subject mask.

Exact patching of S4-S12 toward the detailed run fixes only half of the cases (ladder.py), so the later blocks may re-impose the
text prior. Here the same pooled directions are added over different block ranges:
  A = S4-S12 (the range used so far) | B = S4-S19 | C = S13-S19 | D = S0-S19 | E = D0-S19 (all 25 blocks)
  label_{set}_a{alpha}    mean dh of other subjects with the same rare attribute
  oracle_{set}_a{alpha}   this case's own pooled dh (upper bound)
Usage: python depth.py [--limit N] [--out depth_dev]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers.models.transformers.transformer_flux2 import apply_rotary_emb
from PIL import Image

from attn_mask import subject_token_ids, top_mask
from velocity import Editor

N_TXT, N_TGT = 512, 1024
DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
            "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? "
            "Answer only yes or no.")
ap = argparse.ArgumentParser(); ap.add_argument("--per_flag", type=int, default=14); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="depth_dev"); ap.add_argument("--no_judge", action="store_true")
a = ap.parse_args()
out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)


class Steer:
    """Attention-to-subject maps (S13-S19), pooled read of target / reference subject tokens, and add / clamp on target tokens."""

    def __init__(self, tr):
        self.ids, self.maps = None, {}
        self.cap = self.add = self.clamp = None
        self.it = self.ir = None                                   # indices of the subject tokens inside the target / reference block
        for b in range(13, 20):
            attn = tr.single_transformer_blocks[b].attn; attn.set_processor(self._wrap(attn.processor, b))
        self.layers = [f"D{i}" for i in range(len(tr.transformer_blocks))] + [f"S{i}" for i in range(len(tr.single_transformer_blocks))]
        for name, blk in zip(self.layers, list(tr.transformer_blocks) + list(tr.single_transformer_blocks)):
            blk.register_forward_hook(self._hook(name))

    def _wrap(self, proc, b):
        probe = self

        class W:
            def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None, **kw):
                if probe.ids:
                    qkv = attn.to_qkv_mlp_proj(hidden_states)[..., : 3 * attn.inner_dim]
                    q, k, _ = qkv.chunk(3, dim=-1)
                    q = attn.norm_q(q.unflatten(-1, (-1, attn.head_dim))); k = attn.norm_k(k.unflatten(-1, (-1, attn.head_dim)))
                    q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1); k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
                    p = torch.softmax((q[:, N_TXT:].transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)).float() / attn.head_dim ** 0.5, -1)
                    probe.maps[b] = p[0][:, :, probe.ids].sum(-1).mean(0).cpu()          # (2048,): target then reference queries
                return proc(attn, hidden_states, attention_mask=attention_mask, image_rotary_emb=image_rotary_emb, **kw)
        return W()

    def _hook(self, name):
        def fn(module, inp, o):
            if self.cap is None and self.add is None and self.clamp is None:
                return
            h, s = (o[1], 0) if name.startswith("D") else (o, o.shape[1] - 2 * N_TGT)
            if self.cap is not None:
                self.cap[name] = (h[0, s + self.it].float().mean(0).cpu(), h[0, s + N_TGT + self.ir].float().mean(0).cpu())
            if self.add is not None and name in self.add:
                h[0, s + self.it] += self.add[name].to(h.device, h.dtype)
            if self.clamp is not None and name in self.clamp:
                u, tau = self.clamp[name]; u = u.to(h.device)
                gap = float(tau) - float(h[0, s + self.it].float().mean(0) @ u)
                if gap > 0:
                    h[0, s + self.it] += (gap * u).to(h.dtype)
            return o
        return fn


meta = pd.read_json("data/metadata.jsonl", lines=True); M = meta.set_index(["id", "edit_seed"])
d = pd.read_parquet("drift_table_dev.parquet")
cases = pd.concat([g.sample(min(len(g), a.per_flag), random_state=0) for _, g in d[d.drifted].groupby("flag")])[["id", "edit_seed"]].sort_values(["id", "edit_seed"])
if a.limit:
    cases = cases.sample(a.limit, random_state=0)
T = d[d.fixed_detailed].reset_index(drop=True)
R = [torch.load(f"refdir/{r.id}_{r.edit_seed}.pt") for r in T.itertuples()]
LAYERS = R[0]["layers"]; LI = [LAYERS.index(f"S{i}") for i in range(4, 13)]; L8 = LAYERS.index("S8")
G = torch.stack([x["g"] for x in R]).float(); DH = torch.stack([x["dh"] for x in R]).float()                    # (N, 2, 25, 3072)
GC = torch.stack([torch.load(f"refdir/{r.pair_id}_standard_{r.edit_seed}.pt")["g"] for r in T.itertuples()]).float()
own = {(r.id, r.edit_seed): i for i, r in enumerate(T.itertuples())}
subj, attr, pair = T.subject.values, T.rare_attribute.values, T.pair_id.values
n_subj = T.groupby("rare_attribute").subject.nunique()
print(len(cases), "dev drift cases |", len(T), "training cases", flush=True)

E = Editor(); H = Steer(E.pipe.transformer); dev = E.device
J = None
if not a.no_judge:
    from judge import Judge
    J = Judge()


def ridge(X, Y, b, lam=1.0):
    """Dual ridge on standardized features; returns the prediction for one feature vector b."""
    X, Y, b = X.double().to(dev), Y.double().to(dev), b.double().to(dev)
    mu, sd = X.mean(0), X.std(0) + 1e-6; A = (X - mu) / sd; B = (b - mu) / sd; ym = Y.mean(0); K = A @ A.T
    al = torch.linalg.solve(K + lam * K.trace() / len(K) * torch.eye(len(K), device=dev, dtype=K.dtype), Y - ym)
    return ((B @ A.T) @ al + ym).float().cpu()


SETS = {"A": [f"S{i}" for i in range(4, 13)], "B": [f"S{i}" for i in range(4, 20)], "C": [f"S{i}" for i in range(13, 20)],
        "D": [f"S{i}" for i in range(20)], "E": LAYERS}
LABEL = [("A", 4), ("B", 1), ("B", 2), ("B", 4), ("C", 2), ("C", 4), ("D", 1), ("D", 2), ("D", 4), ("E", 1), ("E", 2), ("E", 4)]
ORACLE = [("A", 2), ("B", 1), ("B", 2), ("C", 2), ("E", 1), ("E", 2)]


def run(st, tv, Dh=None, layers=(), alpha=0.0):
    x = st["x"]
    for i, t in enumerate(st["ts"]):
        H.cap = H.add = H.clamp = None
        if Dh is not None and i < 2:
            H.add = {l: alpha * Dh[i, LAYERS.index(l)] for l in layers}
        v = E.velocity(x, t, tv, st)
        x = (x.float() + (st["sigmas"][i + 1] - st["sigmas"][i]) * v.float()).to(v.dtype)
    H.cap = H.add = H.clamp = None
    return x


rows = []
for n, (cid, seed) in enumerate(cases.itertuples(index=False)):
    r = M.loc[(cid, seed)]
    src = Image.open("data/" + r.source_image).convert("RGB"); st = E.prepare(src, int(seed)); tv = E.encode_text(r.edit_prompt)
    H.ids, H.maps = subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject), {}
    E.velocity(st["x"], st["ts"][0], tv, st); H.ids = None
    amap = torch.stack(list(H.maps.values())).mean(0) if H.maps else torch.ones(2 * N_TGT)
    H.it = top_mask(amap[:N_TGT], 0.25).nonzero()[:, 0].to(dev); H.ir = top_mask(amap[N_TGT:], 0.25).nonzero()[:, 0].to(dev)
    imgs = {}
    imgs["baseline"] = run(st, tv)
    k = own.get((cid, seed)); info = dict(n_train=0, strict=False, has_own=k is not None)
    strict = n_subj.get(r.rare_attribute, 0) > 1
    tr = (attr == r.rare_attribute) & ((subj != r.subject) if strict else (pair != r.pair_id))
    if tr.any():
        Dh = DH[tr].mean(0); info.update(n_train=int(tr.sum()), strict=bool(strict))
        for ls, al in LABEL:
            imgs[f"label_{ls}_a{al}"] = run(st, tv, Dh, SETS[ls], float(al))
    if k is not None:
        for ls, al in ORACLE:
            imgs[f"oracle_{ls}_a{al}"] = run(st, tv, DH[k], SETS[ls], float(al))
    names = list(imgs); ims = [E.decode(x, st) for x in imgs.values()]; small = [im.resize((256, 256)) for im in ims]
    if J is not None:
        p = J.attr(ims, [r.subject] * len(ims), [r.rare_attribute] * len(ims), [r.typical_attribute] * len(ims))
        pe = J.edit([src] * len(ims), ims, [r.edit_prompt] * len(ims))
        l = torch.cat([J._logits([(DEFECT_Q.format(subject=r.subject), [im]) for im in small[i:i + 6]]) for i in range(0, len(small), 6)])
        pdef = torch.sigmoid(torch.logsumexp(l[:, [J.ids["yes"], J.ids["Yes"]]], -1) - torch.logsumexp(l[:, [J.ids["no"], J.ids["No"]]], -1)).float().cpu().numpy()
    else:
        p = pe = pdef = np.full(len(ims), np.nan)
    base = np.asarray(ims[0]).astype(float)
    for j, nm in enumerate(names):
        small[j].save(out / "img" / f"{cid}_{seed}_{nm}.jpg", quality=90)
        rows.append(dict(id=cid, edit_seed=seed, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute, flag=r.flag, edit_family=r.edit_family,
                         method=nm, p_rare=float(p[j]), p_edit=float(pe[j]), p_defect=float(pdef[j]),
                         dpix=float(np.abs(np.asarray(ims[j]).astype(float) - base).mean()), **info))
    print(n, len(cases), cid, r.rare_attribute, "| n_train", info["n_train"], flush=True)
    if n % 10 == 0:
        pd.DataFrame(rows).to_parquet(out / "results.parquet")
pd.DataFrame(rows).to_parquet(out / "results.parquet"); print("done")
