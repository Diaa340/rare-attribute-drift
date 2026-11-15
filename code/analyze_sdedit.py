import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
r = pd.read_parquet((sys.argv[1] if len(sys.argv) > 1 else "sdedit_dev") + "/results.parquet")
r["kept"] = r.p_rare > 0.5; r["edited"] = r.p_edit > 0.5; r["defect"] = r.p_defect > 0.5; r["clean"] = r.kept & r.edited & ~r.defect
print(r.groupby(["id", "edit_seed"]).ngroups, "cases,", r.subject.nunique(), "subjects | by flag:", r[r.method == "baseline"].flag.value_counts().to_dict())
t = r.groupby("method")[["kept", "edited", "defect", "clean"]].mean().round(2); t["dpix"] = r.groupby("method").dpix.mean().round(1)
t["clean_flagavg"] = r.groupby(["method", "flag"]).clean.mean().groupby("method").mean().round(2)
print(t.sort_values("clean", ascending=False).to_string())
for c in ["clean", "kept", "edited"]:
    print(f"\n{c} by flag:"); print(r.pivot_table(index="method", columns="flag", values=c, aggfunc="mean").round(2).loc[t.sort_values("clean", ascending=False).index].to_string())
print("\nclean by edit family:"); print(r.pivot_table(index="method", columns="edit_family", values="clean", aggfunc="mean").round(2).loc[t.sort_values("clean", ascending=False).index].to_string())
