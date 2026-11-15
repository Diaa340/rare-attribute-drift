"""Our methods vs baselines on the held-out test drift cases, with subject-clustered bootstrap 95% CIs."""
import glob

import numpy as np
import pandas as pd

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
INFO = {  # method -> (label, needs attribute label?, extra forward passes per edit vs. 4)
    "baseline": ("no steering", "-", 0),
    "detailed": ("detailed prompt", "label (prompt)", 0),
    "min_prompt": ("minimal attribute prompt", "label (prompt)", 0),
    "textguid_s2": ("text-difference guidance, s=2", "label", 2),
    "negguid_w1": ("negative-attribute guidance, w=1", "label (typical)", 2),
    "refboost_g2": ("reference-attention boost x2", "none", 1),
    "refboost_g4": ("reference-attention boost x4", "none", 1),
    "sdedit_k1": ("SDEdit init, skip 1 step", "none", -1),
    "sdedit_k2": ("SDEdit init, skip 2 steps", "none", -2),
    "vel_attn10": ("ours: velocity steering, mask 10%", "label", 1),
    "vel_attn25": ("ours: velocity steering", "label", 1),
    "hid_full": ("ours: hidden S4-S12, whole image", "label", 0),
    "hid_attn10": ("ours: hidden S4-S12, mask 10%", "label", 1),
    "hid_attn25_a2": ("ours: hidden S4-S12, alpha 2", "label", 1),
    "hid_attn25": ("ours: hidden S4-S12, alpha 4", "label", 1),
}


def load():
    a = pd.concat([pd.read_parquet(f) for f in glob.glob("final_test/defects_*.parquet")], ignore_index=True)
    b = pd.concat([pd.read_parquet(f) for f in glob.glob("baselines_test/results_*.parquet")], ignore_index=True)
    b = b[b.method != "baseline"]
    r = pd.concat([a, b], ignore_index=True)
    base = r[r.method == "baseline"].set_index(["id", "edit_seed"]).p_rare
    r = r[[base.get((i, s), 1.0) < 0.2 for i, s in zip(r.id, r.edit_seed)]].copy()
    have = r.groupby(["id", "edit_seed"]).method.nunique(); full = have[have == have.max()].index      # cases with every method
    r = r[[k in set(full) for k in zip(r.id, r.edit_seed)]]
    r["success"] = (r.p_rare > 0.5) & (r.p_edit > 0.5); r["clean"] = r.success & (r.p_defect <= 0.5)
    return r


def boot_ci(r, col, n=2000, seed=0):
    rng = np.random.default_rng(seed); subj = r.subject.unique()
    per = r.groupby(["method", "subject"])[col].agg(["sum", "count"])
    out = {}
    for m in r.method.unique():
        p = per.loc[m].reindex(subj).fillna(0)
        idx = rng.integers(0, len(subj), (n, len(subj)))
        s = p["sum"].values[idx].sum(1) / p["count"].values[idx].sum(1)
        out[m] = (np.percentile(s, 2.5), np.percentile(s, 97.5))
    return out


if __name__ == "__main__":
    r = load()
    print("test drift cases with all methods:", r.groupby(["id", "edit_seed"]).ngroups, "| subjects:", r.subject.nunique())
    t = r.groupby("method").agg(success=("success", "mean"), clean=("clean", "mean"), kept=("p_rare", lambda x: (x > 0.5).mean()),
                                edited=("p_edit", lambda x: (x > 0.5).mean()), defect=("p_defect", lambda x: (x > 0.5).mean()),
                                bg_change=("dpix_background", "mean"))
    ci = boot_ci(r, "clean")
    t["clean_95CI"] = [f"[{ci[m][0]:.2f}, {ci[m][1]:.2f}]" for m in t.index]
    t["method_name"] = [INFO[m][0] for m in t.index]; t["uses"] = [INFO[m][1] for m in t.index]; t["extra_fwd"] = [INFO[m][2] for m in t.index]
    t = t.sort_values("clean", ascending=False)
    print(t[["method_name", "uses", "extra_fwd", "success", "clean", "clean_95CI", "kept", "edited", "defect", "bg_change"]].round(2).to_string())
    t.to_csv("baselines_summary.csv")
    f = r.pivot_table(index="method", columns="flag", values="clean", aggfunc="mean").round(2)
    f.loc["n"] = r[r.method == "baseline"].groupby("flag").size()
    print("\nclean success by attribute kind:"); print(f.loc[list(t.index) + ["n"]].to_string())
    f.to_csv("baselines_by_flag.csv")
