"""Summary of depth.py: block ranges for pooled steering, with references on the same cases."""
import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
D = sys.argv[1] if len(sys.argv) > 1 else "depth_dev"
r = pd.read_parquet(f"{D}/results.parquet")
ref = pd.read_parquet("sdedit_dev2/results.parquet"); ref = ref[ref.method.isin(["min_prompt", "detailed"])]
r = pd.concat([r, ref], ignore_index=True)
r["kept"] = r.p_rare > 0.5; r["edited"] = r.p_edit > 0.5; r["defect"] = r.p_defect > 0.5; r["clean"] = r.kept & r.edited & ~r.defect
r["key"] = list(zip(r.id, r.edit_seed)); r = r[r.key.isin(set(r[r.method == "baseline"].key))]


def boot(x, n=2000):
    rng = np.random.default_rng(0); s = x.groupby("subject").clean.agg(["sum", "count"]); i = rng.integers(0, len(s), (n, len(s)))
    v = s["sum"].values[i].sum(1) / s["count"].values[i].sum(1); return f"[{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]"


def table(sub, title):
    t = sub.groupby("method").agg(n=("clean", "size"), kept=("kept", "mean"), edited=("edited", "mean"), defect=("defect", "mean"), clean=("clean", "mean")).round(2)
    t["ci"] = [boot(sub[sub.method == m]) for m in t.index]
    print(f"\n== {title} =="); print(t.sort_values("clean", ascending=False).to_string())


lab = set(r[r.method == "label_A_a4"].key); orc = set(r[r.method == "oracle_A_a2"].key)
table(r[r.key.isin(lab) & ~r.method.str.startswith("oracle")], "label directions: cases with a label direction")
table(r[r.key.isin(orc) & ~r.method.str.startswith("label")], "own-dh oracle: cases fixed by the detailed prompt")
print("\nclean by flag:"); print(r.pivot_table(index="method", columns="flag", values="clean", aggfunc="mean").round(2).to_string())
