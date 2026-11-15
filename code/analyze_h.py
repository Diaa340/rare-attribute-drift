"""Where do attribute directions live? Per layer and step, using subject-pooled dh (collect_h.py):
- within: mean cosine between dh of the SAME rare attribute on DIFFERENT subjects
- across: mean cosine between dh of different rare attributes
- loso_acc: leave-one-subject-out nearest-centroid accuracy of the rare attribute (attributes with >= 2 subjects)
Velocity (pooled dv, 128-d) is included as layer "v" for comparison.
"""
import numpy as np
import pandas as pd
import torch

pd.set_option("display.width", 200)
d = pd.read_parquet("drift_table_dev.parquet"); d = d[d.fixed_detailed].reset_index(drop=True)
H = torch.stack([torch.load(f"dh/{r.id}_{r.edit_seed}.pt")["dh"].float() for r in d.itertuples()])      # (N, 2, 25, 3072)
V = torch.stack([torch.load(f"dv/{r.id}_{r.edit_seed}.pt")["pooled_mask"][:2].float() for r in d.itertuples()])  # (N, 2, 128)
layers = torch.load(f"dh/{d.id[0]}_{d.edit_seed[0]}.pt")["layers"]
attr, subj, flag = d.rare_attribute.values, d.subject.values, d.flag.values
multi = d.groupby("rare_attribute").subject.transform("nunique").values > 1
diff_subj = subj[:, None] != subj[None]; same_attr = attr[:, None] == attr[None]


def score(X):
    Xn = torch.nn.functional.normalize(X, dim=-1); C = (Xn @ Xn.T).numpy()
    within = C[same_attr & diff_subj].mean(); across = C[~same_attr].mean()
    hits = []
    for i in np.where(multi)[0]:
        tr = subj != subj[i]
        cents = {a: Xn[tr & (attr == a)].mean(0) for a in set(attr[tr & multi])}
        pred = max(cents, key=lambda a: float(Xn[i] @ cents[a]))
        hits.append(pred == attr[i])
    return within, across, np.mean(hits)


rows = []
for k in range(2):
    rows.append(dict(step=k, layer="v", **dict(zip(["within", "across", "loso_acc"], score(V[:, k])))))
    for li, l in enumerate(layers):
        rows.append(dict(step=k, layer=l, **dict(zip(["within", "across", "loso_acc"], score(H[:, k, li])))))
t = pd.DataFrame(rows); t["gap"] = t.within - t.across
print("cases", len(d), "| LOSO-eligible", multi.sum(), "| chance", round(1 / d[multi].rare_attribute.nunique(), 3))
print(t.round(3).to_string(index=False))
t.to_csv("layer_scores.csv", index=False)

# per flag: within-attribute cross-subject cosine at each layer (step 0), to see if semantic attributes peak elsewhere
out = {}
for f in ["color", "pattern", "age_activity", "place", "pose", "occupation_gender", "state", "material"]:
    m = flag == f
    if (same_attr & diff_subj)[m][:, m].sum() == 0:
        continue
    col = {}
    for li, l in enumerate(["v"] + layers):
        X = V[m, 0] if l == "v" else H[m, 0, li - 1]
        Xn = torch.nn.functional.normalize(X, dim=-1); C = (Xn @ Xn.T).numpy()
        col[l] = C[(same_attr & diff_subj)[m][:, m]].mean()
    out[f] = col
print("\nwithin-attribute, cross-subject cosine at step 0, by flag:")
print(pd.DataFrame(out).round(2).to_string())
