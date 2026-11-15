"""Summarize eval_lora.py outputs. Usage: python analyze_lora.py val_1000 val_2000 ... | test_<tag> | unseen_<tag>"""
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)


def load(name):
    r = pd.read_parquet(f"eval_lora/{name}.parquet")
    base = r[r.method == "baseline"].set_index(["id", "edit_seed"])
    r["base_p_rare"] = [base.p_rare.get((i, s), np.nan) for i, s in zip(r.id, r.edit_seed)]
    r["success"] = (r.p_rare > 0.5) & (r.p_edit > 0.5); r["clean"] = r.success & (r.p_defect <= 0.5)
    return r


def boot(x, subj, n=2000):
    rng = np.random.default_rng(0); s = pd.DataFrame({"x": x, "s": subj}).groupby("s").x.agg(["sum", "count"])
    idx = rng.integers(0, len(s), (n, len(s))); v = s["sum"].values[idx].sum(1) / s["count"].values[idx].sum(1)
    return f"[{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]"


for name in sys.argv[1:]:
    r = load(name)
    dr = r[(r["view"] == "rare") & (r.base_p_rare < 0.2)]
    print(f"\n=== {name}: {dr[dr.method == 'baseline'].shape[0]} drifting rare cases, {dr.subject.nunique()} subjects ===")
    t = dr.groupby("method").agg(success=("success", "mean"), clean=("clean", "mean"), edited=("p_edit", lambda x: (x > 0.5).mean()),
                                 defect=("p_defect", lambda x: (x > 0.5).mean()), dpix=("dpix", "mean")).round(3)
    t["clean_95CI"] = [boot(dr[dr.method == m].clean.values, dr[dr.method == m].subject.values) for m in t.index]
    print(t.sort_values("clean", ascending=False).to_string())
    f = dr.pivot_table(index="flag", columns="method", values="clean", aggfunc="mean").round(2); f["n"] = dr[dr.method == "baseline"].groupby("flag").size()
    print(f.to_string())
    c = r[r["view"] == "common"]
    if len(c):
        ok = c[c.base_p_rare < 0.5]                                   # common cases the unmodified model handles correctly
        g = ok.groupby("method").agg(n=("id", "size"), keeps_typical=("p_rare", lambda x: (x < 0.5).mean()), edited=("p_edit", lambda x: (x > 0.5).mean()),
                                     defect=("p_defect", lambda x: (x > 0.5).mean()), dpix=("dpix", "mean")).round(3)
        print("common views (should stay typical, edit applied):"); print(g.to_string())
