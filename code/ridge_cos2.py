"""CPU: is the ridge map's cosine more than the generic 'detailed prompt' shift? S8, step 0, leave-one-subject-out.
Reports cosine with the true dh for: global mean dh, label mean dh, ridge prediction; and the same after removing the global mean
(of the training subjects) from prediction and truth, which leaves only the attribute-specific part."""
import numpy as np, pandas as pd, torch
torch.set_num_threads(8)
d = pd.read_parquet("drift_table_dev.parquet"); T = d[d.fixed_detailed].reset_index(drop=True)
R = [torch.load(f"refdir/{r.id}_{r.edit_seed}.pt") for r in T.itertuples()]; L = R[0]["layers"]; i, l = 0, L.index("S8")
G = torch.stack([x["g"][i, l] for x in R]).double(); DH = torch.stack([x["dh"][i, l] for x in R]).double()
GC = torch.stack([torch.load(f"refdir/{r.pair_id}_standard_{r.edit_seed}.pt")["g"][i, l] for r in T.itertuples()]).double()
subj, attr, flag = T.subject.values, T.rare_attribute.values, T.flag.values
N = len(T); P = torch.zeros(N, 3072, dtype=torch.double); PC = torch.zeros_like(P); LB = torch.zeros_like(P); GM = torch.zeros_like(P); has = np.zeros(N, bool)
for s in np.unique(subj):
    te = torch.from_numpy(subj == s); tr = ~te
    X = torch.cat([G[tr], GC[tr]]); Y = torch.cat([DH[tr], torch.zeros_like(DH[tr])])
    mu, sd = X.mean(0), X.std(0) + 1e-6; A = (X - mu) / sd; ym = Y.mean(0); K = A @ A.T
    al = torch.linalg.solve(K + K.trace() / len(K) * torch.eye(len(K), dtype=K.dtype), Y - ym)
    P[te] = ((G[te] - mu) / sd) @ A.T @ al + ym; PC[te] = ((GC[te] - mu) / sd) @ A.T @ al + ym; GM[te] = DH[tr].mean(0)
    for k in np.where(subj == s)[0]:
        m = (attr == attr[k]) & (subj != s)
        if m.any():
            LB[k] = DH[torch.from_numpy(m)].mean(0); has[k] = True
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).numpy()
h = torch.from_numpy(has)
print("cosine with the true dh (all cases | cases with a label | cases without):")
for nm, V in [("global mean", GM), ("ridge", P)]:
    c = cos(V, DH); print(f"  {nm:12s} {c.mean():.3f} | {c[has].mean():.3f} | {c[~has].mean():.3f}")
print(f"  {'label mean':12s}   -   | {cos(LB, DH)[has].mean():.3f} |   -")
print("after removing the global mean (attribute-specific part only):")
c = cos(P - GM, DH - GM); print(f"  ridge        {c.mean():.3f} | {c[has].mean():.3f} | {c[~has].mean():.3f}")
print(f"  label mean     -   | {cos(LB - GM, DH - GM)[has].mean():.3f} |   -")
print("  ridge by type:", pd.Series(c).groupby(flag).mean().round(2).to_dict())
print("  label by type:", pd.Series(cos(LB - GM, DH - GM)[has]).groupby(flag[has]).mean().round(2).to_dict())
# does the map stay quiet on common views? projection of the prediction on the unit global-mean direction and residual norm
u = GM / GM.norm(dim=-1, keepdim=True); nd = DH.norm(dim=-1).mean()
print("prediction norm / mean |dh|: rare", round(float(P.norm(dim=-1).mean() / nd), 2), "| common", round(float(PC.norm(dim=-1).mean() / nd), 2),
      "| attribute-specific part: rare", round(float((P - GM).norm(dim=-1).mean() / nd), 2), "| common", round(float((PC - GM).norm(dim=-1).mean() / nd), 2))
pr, pc = (P * u).sum(-1), (PC * u).sum(-1); thr = (pr.median() + pc.median()) / 2
print("rare vs common by the projection of the prediction on the mean direction: rare above threshold", round(float((pr > thr).float().mean()), 2), "| common below", round(float((pc <= thr).float().mean()), 2))
