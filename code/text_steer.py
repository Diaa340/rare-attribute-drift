"""SHIFT-style steering of the TEXT tokens for attribute drift in FLUX.2 [klein] edits.

Direction for an attribute a (no image labels needed, only contrastive prompts on unrelated dev images):
    d_l(a) = mean over K context images of  h_l(prompt + ". The <subject> is <a>.") - h_l(prompt)
taken at token positions shared by both prompts: 'S' = the subject word, 'A' = all tokens of the vanilla prompt.
Steering: h_l[positions] += alpha * (typical token norm at l) * unit(d_l(a)) at sampling steps 0-1, vanilla prompt.

  python text_steer.py --collect dev            directions for the attributes of the dev sweep cases
  python text_steer.py --mode dev               sweep on the 121 type-balanced dev drift cases (contexts of the same subject left out)
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from attn_mask import subject_token_ids
from velocity import Editor

ap = argparse.ArgumentParser()
ap.add_argument("--collect"); ap.add_argument("--mode"); ap.add_argument("--k", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="text_dev"); ap.add_argument("--no_judge", action="store_true")
ap.add_argument("--dirs", default="text_dirs")
a = ap.parse_args()
dirs = Path(a.dirs); dirs.mkdir(exist_ok=True)
slug = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
meta = pd.read_json("data/metadata.jsonl", lines=True); M = meta.set_index(["id", "edit_seed"])
drift = pd.read_parquet("drift_table_dev.parquet"); drift = drift[drift.drifted]
dev_cases = pd.concat([g.sample(min(len(g), 14), random_state=0) for _, g in drift.groupby("flag")])[["id", "edit_seed"]].sort_values(["id", "edit_seed"])


def positions(tok, prompt, subject):
    """Token positions (in the padded 512 sequence) of the subject word and of the whole vanilla prompt."""
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=512)
    lo = text.find(prompt); hi = lo + len(prompt) - 1               # drop the last character: its token may merge with what follows
    A = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s >= lo and e <= hi and e > s]
    S = [i for i in subject_token_ids(tok, prompt, subject) if i in set(A)] or A
    return torch.tensor(S), torch.tensor(A)


class TextHooks:
    """Read or shift text-token hidden states at the output of every block. Double block: (txt, img); single block: [txt 512, img]."""

    def __init__(self, tr):
        self.read, self.cap, self.add, self.pos = None, {}, None, None
        self.layers = [f"D{i}" for i in range(len(tr.transformer_blocks))] + [f"S{i}" for i in range(len(tr.single_transformer_blocks))]
        for name, b in zip(self.layers, list(tr.transformer_blocks) + list(tr.single_transformer_blocks)):
            b.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(module, inp, out):
            h = out[0] if name.startswith("D") else out               # (1, 512, 3072) text tokens are the first 512 of the single stream
            if self.read is not None:
                for k, idx in self.read.items():
                    t = h[0, idx.to(h.device)].float()
                    self.cap[(name, k)] = t.mean(0); self.cap[(name, k + "_n")] = t.norm(dim=-1).mean()
            if self.add is not None and name in self.add:
                h[0, self.pos.to(h.device)] += self.add[name].to(h.device, h.dtype)
            return out
        return fn

    def grab(self, kind):
        return torch.stack([self.cap[(l, kind)] for l in self.layers]), torch.stack([self.cap[(l, kind + "_n")] for l in self.layers])


E = Editor(device=a.device); tok = E.pipe.tokenizer; H = TextHooks(E.pipe.transformer); L = H.layers
euler = lambda x, v, st, i: (x.float() + (st["sigmas"][i + 1] - st["sigmas"][i]) * v.float()).to(v.dtype)

if a.collect:
    attrs = sorted(set(M.loc[list(dev_cases.itertuples(index=False, name=None))].rare_attribute)) if a.collect == "dev" else a.collect.split("|")
    attrs = [x for x in attrs if not (dirs / f"{slug(x)}.pt").exists()]
    ctx = meta[(meta.split == "dev") & (meta["view"] == "common") & (meta.seed_kind == "dataset")].drop_duplicates("subject").sample(a.k, random_state=0)
    if a.limit:
        attrs, ctx = attrs[:a.limit], ctx.head(a.limit)
    print(len(attrs), "attributes x", len(ctx), "contexts", flush=True)
    acc = {x: {"S": [], "A": []} for x in attrs}; norms = []
    for n, c in enumerate(ctx.itertuples()):
        st = E.prepare(Image.open("data/" + c.source_image).convert("RGB"), int(c.edit_seed)); tv = E.encode_text(c.edit_prompt)
        pS, pA = positions(tok, c.edit_prompt, c.subject); H.read = {"S": pS, "A": pA}
        v0 = E.velocity(st["x"], st["ts"][0], tv, st); base0 = {k: H.grab(k) for k in "SA"}
        x1 = euler(st["x"], v0, st, 0)
        E.velocity(x1, st["ts"][1], tv, st); base1 = {k: H.grab(k) for k in "SA"}
        norms.append(torch.stack([torch.stack([base0[k][1], base1[k][1]]) for k in "SA"]).cpu())            # (2 kinds, 2 steps, 25)
        for x in attrs:
            tm = E.encode_text(f"{c.edit_prompt}. The {c.subject} is {x}.")
            E.velocity(st["x"], st["ts"][0], tm, st); m0 = {k: H.grab(k)[0] for k in "SA"}
            E.velocity(x1, st["ts"][1], tm, st); m1 = {k: H.grab(k)[0] for k in "SA"}
            for k in "SA":
                acc[x][k].append(torch.stack([m0[k] - base0[k][0], m1[k] - base1[k][0]]).half().cpu())      # (2 steps, 25, 3072)
        H.read = None; print(n, len(ctx), c.subject, flush=True)
    for x in attrs:
        torch.save({"S": torch.stack(acc[x]["S"]), "A": torch.stack(acc[x]["A"]), "subjects": list(ctx.subject)}, dirs / f"{slug(x)}.pt")
    torch.save(torch.stack(norms).mean(0), dirs / "_norms.pt")
    d = torch.load(dirs / f"{slug(attrs[0])}.pt")
    for k in "SA":
        u = torch.nn.functional.normalize(d[k].float(), dim=-1)          # consistency of the direction across contexts
        cs = torch.einsum("isld,jsld->ijsl", u, u); off = ~torch.eye(len(u), dtype=bool)
        print(attrs[0], k, "cos across contexts, step 0:", {L[i]: round(cs[off][:, 0, i].mean().item(), 2) for i in [0, 2, 4, 9, 14, 24]}, flush=True)
    print("collected")

if a.mode == "dev":
    out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
    have = [(dirs / f"{slug(M.loc[k].rare_attribute)}.pt").exists() for k in dev_cases.itertuples(index=False, name=None)]
    cases = dev_cases[have].head(a.limit) if a.limit else dev_cases
    hn = torch.load(dirs / "_norms.pt")                                   # (2 kinds, 2 steps, 25)
    LS = {"D": [l for l in L if l.startswith("D")], "DS": [l for l in L if l.startswith("D")] + [f"S{i}" for i in range(13)]}
    VAR = [("S", ls, al) for ls in LS for al in (0.5, 1.0, 2.0, 4.0)] + [("A", ls, al) for ls in LS for al in (0.125, 0.25, 0.5, 1.0)]
    if a.limit:
        VAR = VAR[:2]
    J = None
    if not a.no_judge:
        from judge import Judge
        J = Judge()
    DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
                "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? Answer only yes or no.")
    rows = []
    for n, (cid, seed) in enumerate(cases.itertuples(index=False)):
        r = M.loc[(cid, seed)]; f = dirs / f"{slug(r.rare_attribute)}.pt"
        if not f.exists():
            print("no direction for", r.rare_attribute, flush=True); continue
        D = torch.load(f); keep = torch.tensor([s != r.subject for s in D["subjects"]])
        src = Image.open("data/" + r.source_image).convert("RGB"); st = E.prepare(src, int(seed)); tv = E.encode_text(r.edit_prompt)
        pos = dict(zip("SA", positions(tok, r.edit_prompt, r.subject)))

        def gen(kind=None, ls=None, al=0.0):
            x = st["x"]
            for i, t in enumerate(st["ts"]):
                if kind and i < 2:
                    u = torch.nn.functional.normalize(D[kind][keep].float().mean(0)[i], dim=-1)             # (25, 3072)
                    H.add = {l: al * hn["SA".index(kind), i, L.index(l)] * u[L.index(l)] for l in LS[ls]}; H.pos = pos[kind]
                x = euler(x, E.velocity(x, t, tv, st), st, i); H.add = None
            return E.decode(x, st)

        names = ["baseline"] + [f"{k}_{ls}_a{al:g}" for k, ls, al in VAR]; ims = [gen()] + [gen(*v) for v in VAR]
        base = np.asarray(ims[0]).astype(float); k = len(ims)
        if J is not None:
            p = J.attr(ims, [r.subject] * k, [r.rare_attribute] * k, [r.typical_attribute] * k); pe = J.edit([src] * k, ims, [r.edit_prompt] * k)
            small = [im.resize((256, 256)) for im in ims]
            l = torch.cat([J._logits([(DEFECT_Q.format(subject=r.subject), [im]) for im in small[i:i + 6]]) for i in range(0, k, 6)])
            pdef = torch.sigmoid(torch.logsumexp(l[:, [J.ids["yes"], J.ids["Yes"]]], -1) - torch.logsumexp(l[:, [J.ids["no"], J.ids["No"]]], -1)).float().cpu().numpy()
        else:
            p = pe = pdef = np.full(k, np.nan)
        for j, nm in enumerate(names):
            ims[j].save(out / "img" / f"{cid}_{seed}_{nm}.jpg", quality=90)
            rows.append(dict(id=cid, edit_seed=seed, subject=r.subject, rare=r.rare_attribute, typical=r.typical_attribute, flag=r.flag, edit_family=r.edit_family,
                             method=nm, p_rare=float(p[j]), p_edit=float(pe[j]), p_defect=float(pdef[j]), dpix=float(np.abs(np.asarray(ims[j]).astype(float) - base).mean())))
        if n % 10 == 0:
            pd.DataFrame(rows).to_parquet(out / "results.parquet"); print(n, len(cases), flush=True)
    pd.DataFrame(rows).to_parquet(out / "results.parquet")
    print("done")
