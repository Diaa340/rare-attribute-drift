"""CPU-only: can the rare attribute be read, without a label, from the model's own activations on the reference image?

Features per dev case (vanilla trajectory): hr = pooled hidden state of the reference-image subject tokens, g = hr - target-subject tokens.
(1) attribute identification among rare-view cases, leave-one-subject-out, dual ridge classifier; (2) rare vs common view detection.
"""
import glob, re
import numpy as np, pandas as pd, torch

meta = pd.read_json("data/metadata.jsonl", lines=True).set_index(["id", "edit_seed"])
rows, G, HR = [], [], []
for f in sorted(glob.glob("refdir/*.pt")):
    m = re.match(r"refdir/(.+)_(\d+)\.pt", f); cid, seed = m.group(1), int(m.group(2)); d = torch.load(f); r = meta.loc[(cid, seed)]
    rows.append(dict(id=cid, seed=seed, subject=r.subject, attr=r.rare_attribute, flag=r.flag, rare="_rare" in cid))
    G.append(d["g"]); HR.append(d["g"] + d["ht"])
T = pd.DataFrame(rows); G = torch.stack(G).numpy(); HR = torch.stack(HR).numpy(); layers = d["layers"]
print(len(T), "cases |", T.rare.sum(), "rare |", T.subject.nunique(), "subjects |", T[T.rare].attr.nunique(), "rare attributes")


def ridge_loso(X, y, groups, lam=1.0, topk=(1, 3)):
    """One-vs-all dual ridge on standardized features, leave-one-group-out. Returns top-k accuracies."""
    classes = sorted(set(y)); Y = np.stack([(y == c) for c in classes], 1).astype(np.float64) * 2 - 1; hit = {k: 0 for k in topk}; n = 0
    for gname in np.unique(groups):
        te = groups == gname; tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; A = (X[tr] - mu) / sd; B = (X[te] - mu) / sd
        K = A @ A.T; alpha = np.linalg.solve(K + lam * np.trace(K) / len(K) * np.eye(len(K)), Y[tr]); S = (B @ A.T) @ alpha
        order = np.argsort(-S, 1); yi = np.array([classes.index(c) if c in classes else -1 for c in y[te]])
        for k in topk:
            hit[k] += sum(yi[i] in order[i, :k] for i in range(len(yi)))
        n += te.sum()
    return {k: hit[k] / n for k in topk}


# (1) attribute identification on rare cases; classes = attributes seen for >= 2 subjects (so a held-out subject's class exists in training)
R = T[T.rare]; ns = R.groupby("attr").subject.nunique(); ok = R.attr.isin(ns[ns >= 2].index).values; idx = np.where(T.rare.values)[0][ok]
y, grp = T.attr.values[idx], T.subject.values[idx]
maj = pd.Series(y).value_counts(normalize=True).iloc[0]
print(f"\n(1) attribute identification: {len(idx)} rare cases, {len(set(y))} attributes, {len(set(grp))} subjects | majority class {maj:.2f}")
print("feature  step layer   top1  top3")
best = None
for name, F in [("ref-subject h", HR), ("ref - target g", G)]:
    for s in [0, 1]:
        for li in [0, 2, 4, 7, 9, 11, 13, 15, 17, 20, 24]:
            a = ridge_loso(F[idx, s, li].astype(np.float64), y, grp)
            print(f"{name:15s} {s}   {layers[li]:4s}   {a[1]:.2f}  {a[3]:.2f}")
            if best is None or a[1] > best[0]:
                best = (a[1], name, s, li)
s4, s12 = layers.index("S4"), layers.index("S12")
for name, F in [("ref-subject h", HR), ("ref - target g", G)]:
    a = ridge_loso(F[idx, 0, s4:s12 + 1].reshape(len(idx), -1).astype(np.float64), y, grp); print(f"{name:15s} 0   S4-S12 concat   {a[1]:.2f}  {a[3]:.2f}")
print("best single:", best)
# per attribute type at the best setting
_, name, s, li = best; F = HR if name.startswith("ref-subject") else G
classes = sorted(set(y)); X = F[idx, s, li].astype(np.float64); pred = np.empty(len(idx), dtype=object)
for gname in np.unique(grp):
    te = grp == gname; tr = ~te; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; A = (X[tr] - mu) / sd; B = (X[te] - mu) / sd
    Y = np.stack([(y[tr] == c) for c in classes], 1) * 2.0 - 1; K = A @ A.T
    pred[te] = np.array(classes)[np.argmax((B @ A.T) @ np.linalg.solve(K + np.trace(K) / len(K) * np.eye(len(K)), Y), 1)]
fl = T.flag.values[idx]; print("top-1 by attribute type:", {f: (round(float((pred[fl == f] == y[fl == f]).mean()), 2), int((fl == f).sum())) for f in sorted(set(fl))})

# (2) does this image need a correction? rare vs common view, leave-one-subject-out
yv, grp2 = np.where(T.rare.values, "rare", "common"), T.subject.values
print(f"\n(2) rare vs common view: {len(T)} cases | majority {max((yv == 'rare').mean(), (yv == 'common').mean()):.2f}")
for name, F in [("ref-subject h", HR), ("ref - target g", G)]:
    for s, li in [(0, 4), (0, 9), (0, 13), (0, 17), (1, 9), (1, 13)]:
        print(f"{name:15s} {s} {layers[li]:4s} acc {ridge_loso(F[:, s, li].astype(np.float64), yv, grp2, topk=(1,))[1]:.2f}")
