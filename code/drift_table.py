"""Drift table: rare-view rows whose vanilla edit drifted to the typical attribute, with the edit(s) that corrected it.

drifted  = judge P(rare | vanilla edit) < DRIFT
corrected = P(rare | detailed or manual edit) > KEPT and P(edit applied) > EDIT
"""
import base64
import io
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent / "data"
DRIFT, KEPT, EDIT = 0.2, 0.8, 0.5


def build(judge_path="judge_dev.parquet"):
    meta = pd.read_json(ROOT / "metadata.jsonl", lines=True)
    df = pd.read_parquet(judge_path).merge(meta, on=["id", "edit_seed"])
    rare = df[df["view"] == "rare"].copy()
    rare["drifted"] = rare.p_rare_vanilla < DRIFT
    for k in ["detailed", "manual"]:
        rare[f"fixed_{k}"] = rare.drifted & (rare[f"p_rare_{k}"] > KEPT) & (rare[f"p_edit_{k}"] > EDIT)
    rare["fixed_by"] = [("detailed+manual" if d and m else "detailed" if d else "manual" if m else "none") if dr else "-"
                        for dr, d, m in zip(rare.drifted, rare.fixed_detailed, rare.fixed_manual)]
    return rare


def summary(rare):
    s = rare.groupby(["seed_kind"]).agg(n=("id", "size"), drift_rate=("drifted", "mean"),
                                        fixed_detailed=("fixed_detailed", "sum"), fixed_manual=("fixed_manual", "sum"))
    by_flag = rare[rare.drifted].groupby("flag").agg(drifted=("id", "size"), fixed_detailed=("fixed_detailed", "mean"),
                                                     fixed_manual=("fixed_manual", "mean")).round(2)
    return s, by_flag


def _thumb(path, size=160):
    im = Image.open(ROOT / path).convert("RGB"); im.thumbnail((size, size))
    b = io.BytesIO(); im.save(b, format="JPEG", quality=85)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}">'


def html(rows, size=160):
    """rows: drifted rows from build(). Columns: source | drifted (vanilla) | corrected (detailed / manual) + fixing prompt."""
    cells = []
    for _, r in rows.iterrows():
        fixes = []
        for k in ["detailed", "manual"]:
            if r[f"fixed_{k}"]:
                fixes.append(f'<td>{_thumb(r[f"{k}_image"], size)}<br><b>{k}</b> P(rare)={r[f"p_rare_{k}"]:.2f}</td>'
                             f'<td style="max-width:260px;font-size:11px">{r[f"{k}_prompt"]}</td>')
        if not fixes:
            fixes = ['<td colspan=2><i>no prompt corrected it</i></td>']
        cells.append(f'<tr><td style="font-size:11px"><b>{r.id}</b><br>seed {r.edit_seed} ({r.seed_kind})<br>{r.subject}<br>'
                     f'rare: <b>{r.rare_attribute}</b><br>typical: {r.typical_attribute}<br>{r.flag} / {r.edit_family}</td>'
                     f'<td>{_thumb(r.source_image, size)}</td>'
                     f'<td>{_thumb(r.edited_image, size)}<br>P(rare)={r.p_rare_vanilla:.2f}</td>'
                     f'<td style="max-width:160px;font-size:11px">{r.edit_prompt}</td>' + "".join(fixes) + "</tr>")
    head = ("<tr><th>row</th><th>source</th><th>drifted (vanilla)</th><th>vanilla prompt</th>"
            "<th>corrected</th><th>prompt that fixed it</th><th>corrected</th><th>prompt that fixed it</th></tr>")
    return f'<table style="border-collapse:collapse" border=1>{head}{"".join(cells)}</table>'


if __name__ == "__main__":
    rare = build()
    s, f = summary(rare); print(s, "\n", f)
    d = rare[rare.drifted]
    d.drop(columns=[c for c in d.columns if c.endswith("_image") and c not in ["source_image", "edited_image", "detailed_image", "manual_image"]]) \
     .to_parquet("drift_table_dev.parquet")
    Path("drift_table_dev.html").write_text("<html><body>" + html(d) + "</body></html>")
    print("wrote drift_table_dev.parquet / .html with", len(d), "drifted rows")
