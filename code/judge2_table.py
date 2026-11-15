"""Test table under both judges (233 drifting test cases)."""
import numpy as np, pandas as pd
from analyze_baselines import load
pd.set_option("display.width", 220)
j = pd.read_parquet("judge2_test.parquet")
j["kept2"] = j.p_rare2 > 0.5; j["edited2"] = j.p_edit2 > 0.5; j["defect2"] = j.p_defect2 > 0.5; j["clean2"] = j.kept2 & j.edited2 & ~j.defect2


def boot(x, col, n=2000):
    rng = np.random.default_rng(0); s = x.groupby("subject")[col].agg(["sum", "count"]); i = rng.integers(0, len(s), (n, len(s)))
    v = s["sum"].values[i].sum(1) / s["count"].values[i].sum(1); return f"[{100 * np.percentile(v, 2.5):.0f}, {100 * np.percentile(v, 97.5):.0f}]"


t = j.groupby("method")[["kept2", "edited2", "defect2", "clean2"]].mean().mul(100).round(0)
t["ci"] = [boot(j[j.method == m], "clean2") for m in t.index]
print(t.sort_values("clean2", ascending=False).to_string())
print("\nclean (judge 2) by attribute type:")
f = j.pivot_table(index="method", columns="flag", values="clean2", aggfunc="mean").mul(100).round(0); f.loc["n"] = j[j.method == "baseline"].groupby("flag").size(); print(f.to_string())
# paired difference steering vs prompts under judge 2
w = j.pivot_table(index=["id", "edit_seed", "subject"], columns="method", values="clean2", aggfunc="first").astype(float).reset_index()
for a_, b_ in [("hid_attn25", "min_prompt"), ("hid_attn25", "detailed"), ("hid_attn25", "textguid_s2")]:
    dlt = w[a_] - w[b_]; rng = np.random.default_rng(0); s = pd.DataFrame({"d": dlt, "s": w.subject}).groupby("s").d.agg(["sum", "count"]); i = rng.integers(0, len(s), (2000, len(s)))
    v = s["sum"].values[i].sum(1) / s["count"].values[i].sum(1)
    print(f"{a_} - {b_}: {100 * dlt.mean():+.0f} points [{100 * np.percentile(v, 2.5):+.0f}, {100 * np.percentile(v, 97.5):+.0f}]")
