"""Summary of steer2.py: every method on the cases where it could run, next to the references on the same cases."""
import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
D = sys.argv[1] if len(sys.argv) > 1 else "steer2_dev"
r = pd.read_parquet(f"{D}/results.parquet")
ref = pd.read_parquet("sdedit_dev2/results.parquet"); ref = ref[ref.method.isin(["min_prompt", "detailed", "sdedit_k1", "rep1_full_subj_b3"])]
r = pd.concat([r, ref], ignore_index=True)
for c, col, thr in [("kept", "p_rare", 0.5), ("edited", "p_edit", 0.5), ("defect", "p_defect", 0.5)]:
    r[c] = r[col] > thr
r["clean"] = r.kept & r.edited & ~r.defect
key = lambda x: list(zip(x.id, x.edit_seed))
r["key"] = key(r); done = set(r[r.method == "baseline"].key); r = r[r.key.isin(done)]
print(len(done), "cases,", r.subject.nunique(), "subjects")


def boot(x, n=2000):
    rng = np.random.default_rng(0); s = x.groupby("subject").clean.agg(["sum", "count"]); i = rng.integers(0, len(s), (n, len(s)))
    v = s["sum"].values[i].sum(1) / s["count"].values[i].sum(1); return f"[{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]"


def table(sub, title):
    t = sub.groupby("method").agg(n=("clean", "size"), kept=("kept", "mean"), edited=("edited", "mean"), defect=("defect", "mean"), clean=("clean", "mean")).round(2)
    t["ci"] = [boot(sub[sub.method == m]) for m in t.index]
    print(f"\n== {title} =="); print(t.sort_values("clean", ascending=False).to_string())


table(r, "all finished cases (methods that need a label or the case's own dh run on fewer cases)")
lab = set(r[r.method == "label_a4"].key); table(r[r.key.isin(lab)], "cases with a label direction")
strict = set(r[(r.method == "label_a4") & (r.strict == True)].key); table(r[r.key.isin(strict)], "cases whose label direction comes from other subjects only")
orc = set(r[r.method == "oracle_a1"].key); table(r[r.key.isin(orc) & r.key.isin(lab)], "cases with both a label direction and an own-dh oracle")
print("\nclean by flag (all finished cases):")
print(r.pivot_table(index="method", columns="flag", values="clean", aggfunc="mean").round(2).to_string())
b = r[r.method == "route_a4"]
if len(b):
    print("\nrouting: predicted attribute == true attribute:", round(float((b.route == b.rare).mean()), 2), "| by flag:", b.assign(ok=b.route == b.rare).groupby("flag").ok.mean().round(2).to_dict())
