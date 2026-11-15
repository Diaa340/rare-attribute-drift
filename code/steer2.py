"""Hidden-state steering variants on the dev drift cases (vanilla prompt, steps 0-1, blocks S4-S12, top-25% subject mask).

  baseline            no steering
  label_a4            h += 4 * mean dh of other subjects' cases with the same rare attribute (the existing method)
  oracle_a{1,2,4}     h += a * this case's own pooled dh (upper bound for a pooled direction; needs the detailed prompt)
  clamp_g{2,4,8}      closed loop: at every block push the pooled projection on the label direction up to
                      (vanilla projection of this case) + g * |mean dh|; no push when it is already there
  ridge_a{4,8}        label-free: dh predicted from this case's reference-minus-target gap g by a ridge map
                      trained on other subjects (rare cases -> their dh, common views -> 0)
  route_a4            label-free: ridge classifier on g (S8, step 0) names the attribute, then label_a4 with that attribute
Training rows never include the evaluated subject (label: other subjects, or other pairs when the attribute has one subject).
Usage: python steer2.py [--limit N] [--out steer2_dev]
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
ap.add_argument("--out", default="steer2_dev"); ap.add_argument("--no_judge", action="store_true")
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


def run(st, tv, mode=None, spec=None):
    x = st["x"]; caps = []
    for i, t in enumerate(st["ts"]):
        H.cap = H.add = H.clamp = None
        if i < 2:
            if mode == "cap":
                H.cap = {}
            elif mode == "add":
                H.add = {LAYERS[l]: spec[i, j] for j, l in enumerate(LI)}
            elif mode == "clamp":
                H.clamp = {LAYERS[l]: spec[i][j] for j, l in enumerate(LI)}
        v = E.velocity(x, t, tv, st)
        if mode == "cap" and i < 2:
            caps.append(H.cap)
        x = (x.float() + (st["sigmas"][i + 1] - st["sigmas"][i]) * v.float()).to(v.dtype)
    H.cap = H.add = H.clamp = None
    return x, caps


rows = []
for n, (cid, seed) in enumerate(cases.itertuples(index=False)):
    r = M.loc[(cid, seed)]
    src = Image.open("data/" + r.source_image).convert("RGB"); st = E.prepare(src, int(seed)); tv = E.encode_text(r.edit_prompt)
    H.ids, H.maps = subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject), {}
    E.velocity(st["x"], st["ts"][0], tv, st); H.ids = None
    amap = torch.stack(list(H.maps.values())).mean(0) if H.maps else torch.ones(2 * N_TGT)
    H.it = top_mask(amap[:N_TGT], 0.25).nonzero()[:, 0].to(dev); H.ir = top_mask(amap[N_TGT:], 0.25).nonzero()[:, 0].to(dev)
    imgs = {}
    imgs["baseline"], caps = run(st, tv, "cap")
    g = torch.stack([torch.stack([c[l][1] - c[l][0] for l in LAYERS]) for c in caps])          # (2, 25, 3072)
    ht = torch.stack([torch.stack([c[l][0] for l in LAYERS]) for c in caps])
    k = own.get((cid, seed)); info = dict(n_train=0, strict=False, route="", has_own=k is not None)
    if k is not None and n < 5:
        print("check: cosine of online g with stored g (S8, step 0/1):",
              [round(float(torch.nn.functional.cosine_similarity(g[i, L8], G[k, i, L8], 0)), 3) for i in range(2)], flush=True)
    # label direction
    strict = n_subj.get(r.rare_attribute, 0) > 1
    tr = (attr == r.rare_attribute) & ((subj != r.subject) if strict else (pair != r.pair_id))
    if tr.any():
        Dh = DH[tr].mean(0); info.update(n_train=int(tr.sum()), strict=bool(strict))
        imgs["label_a4"], _ = run(st, tv, "add", 4.0 * Dh[:, LI])
        for gam in (2, 4, 8):
            spec = [[(Dh[i, l] / Dh[i, l].norm(), float(ht[i, l] @ Dh[i, l] / Dh[i, l].norm()) + gam * float(Dh[i, l].norm())) for l in LI] for i in range(2)]
            imgs[f"clamp_g{gam}"], _ = run(st, tv, "clamp", spec)
    if k is not None:
        for al in (1, 2, 4):
            imgs[f"oracle_a{al}"], _ = run(st, tv, "add", float(al) * DH[k][:, LI])
    # label-free: other subjects only
    o = torch.from_numpy(subj != r.subject)
    X = torch.cat([G[o], GC[o]]); Y = torch.cat([DH[o], torch.zeros_like(DH[o])])
    pred = torch.stack([torch.stack([ridge(X[:, i, l], Y[:, i, l], g[i, l]) for l in LI]) for i in range(2)])   # (2, 9, 3072)
    info["ridge_norm_ratio"] = float(pred.norm(dim=-1).mean() / DH[o][:, :, LI].norm(dim=-1).mean())
    for al in (4, 8):
        imgs[f"ridge_a{al}"], _ = run(st, tv, "add", float(al) * pred)
    classes = sorted(set(attr[o.numpy()])); Yc = torch.tensor(np.stack([attr[o.numpy()] == c for c in classes], 1) * 2.0 - 1)
    sc = ridge(G[o][:, 0, L8], Yc, g[0, L8]); info["route"] = classes[int(sc.argmax())]
    Dr = DH[torch.from_numpy((attr == info["route"]) & (subj != r.subject))].mean(0)
    imgs["route_a4"], _ = run(st, tv, "add", 4.0 * Dr[:, LI])

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
    print(n, len(cases), cid, r.rare_attribute, "| route:", info["route"], "| n_train", info["n_train"], "| ridge norm ratio", round(info["ridge_norm_ratio"], 2), flush=True)
    if n % 10 == 0:
        pd.DataFrame(rows).to_parquet(out / "results.parquet")
pd.DataFrame(rows).to_parquet(out / "results.parquet"); print("done")
