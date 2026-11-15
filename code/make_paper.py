"""Figures and tables for the short paper, written to paper/figs and paper/tables. CPU only: python make_paper.py"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from analyze_baselines import boot_ci, load

figs, tabs = Path("paper/figs"), Path("paper/tables"); figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42})
pct = lambda v: f"{100 * v:.0f}"

# ---------------------------------------------------------------- Figure: where the attribute is decided
r = pd.read_parquet("step0/results.parquet"); r = r[r.p_rare_vanilla < 0.2]
names = {"vanilla": "none", "det_step0_only": "0", "det_steps1-3": "1-3", "det_steps0-1": "0-1", "detailed": "0-3"}
vals = [(r[f"p_rare_{k}"] > 0.5).mean() for k in names]
t = pd.read_csv("layer_scores.csv"); t0 = t[(t.step == 0) & (t.layer != "v")].reset_index(drop=True); vrow = t[(t.step == 0) & (t.layer == "v")].iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.05), gridspec_kw={"width_ratios": [1, 1.45]})
ax[0].bar(range(5), vals, width=0.62, color=[GREY] + [BLUE] * 4)
for i, v in enumerate(vals):
    ax[0].text(i, v + 0.025, f"{v:.0%}", ha="center", color=INK)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(list(names.values()), fontsize=7.5); ax[0].set_ylim(0, 1.12); ax[0].set_yticks([0, 0.5, 1]); ax[0].set_yticklabels(["0%", "50%", "100%"])
ax[0].set_ylabel("rare attribute kept"); ax[0].set_xlabel("steps given the detailed prompt", fontsize=7); ax[0].set_title(f"(a) Step swap (n = {len(r)})", fontsize=8, loc="left")
x = np.arange(len(t0)); s4, s12 = list(t0.layer).index("S4"), list(t0.layer).index("S12")
ax[1].axvspan(s4 - 0.5, s12 + 0.5, color=BLUE, alpha=0.08, lw=0); ax[1].text((s4 + s12) / 2, 1.0, "S4-S12", ha="center", va="top", color=MUTED, fontsize=7)
ax[1].plot(x, t0.loso_acc, "-o", color=BLUE, lw=1.6, ms=3); ax[1].text(x[-1] + 0.3, t0.loso_acc.iloc[-1], "hidden\nstate", color=INK, fontsize=7, va="center")
ax[1].axhline(vrow.loso_acc, color=MUTED, lw=1, ls="--"); ax[1].text(0.2, vrow.loso_acc + 0.02, "velocity", color=MUTED, fontsize=7)
ax[1].axhline(1 / 11, color=MUTED, lw=0.6, ls=":"); ax[1].text(0.2, 1 / 11 + 0.02, "chance", color=MUTED, fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels(t0.layer, rotation=90, fontsize=6); ax[1].set_ylim(0, 1.02); ax[1].set_xlim(-0.6, len(x) + 2.2)
ax[1].set_ylabel("attribute accuracy"); ax[1].set_title("(b) Attribute decodable from the correction", fontsize=8, loc="left")
plt.tight_layout(); plt.savefig(figs / "decision.pdf"); plt.close()

# ---------------------------------------------------------------- test table (same 233 cases for every method)
R = load()
L = pd.read_parquet("eval_lora/test_8000.parquet"); L = L[(L.method == "lora") & (L["view"] == "rare")].copy()
L["success"] = (L.p_rare > 0.5) & (L.p_edit > 0.5); L["clean"] = L.success & (L.p_defect <= 0.5)
keys = set(zip(R.id, R.edit_seed)); L = L[[k in keys for k in zip(L.id, L.edit_seed)]]
R = pd.concat([R, L], ignore_index=True)
ROWS = [("detailed", "Detailed prompt", "prompt", "0"), ("min_prompt", "Minimal attribute prompt", "prompt", "0"), ("textguid_s2", "Text-difference guidance", "yes", "2"),
        ("hid_attn25", "Hidden-state steering (S4--S12)", "yes", "1"), ("vel_attn25", "Velocity steering", "yes", "1"), None,
        ("lora", "Velocity-matching LoRA", "no", "0"), ("sdedit_k1", "SDEdit, skip 1 step", "no", "$-1$"), ("negguid_w1", "Negative guidance", "typical", "2"),
        ("refboost_g4", "Reference-attention boost", "no", "1"), ("baseline", "Vanilla prompt", "no", "0")]
ci = boot_ci(R, "clean"); g = R.groupby("method")
J2 = pd.read_parquet("judge2_test.parquet") if os.path.exists("judge2_test.parquet") else None
if J2 is not None:
    J2["clean2"] = (J2.p_rare2 > 0.5) & (J2.p_edit2 > 0.5) & (J2.p_defect2 <= 0.5); c2 = J2.groupby("method").clean2.mean()
head = "Method & Label & Passes & Kept & Edited & Defect & Clean [95\\% CI]" + (" & Judge 2" if J2 is not None else "") + " \\\\"
lines = ["\\begin{tabular}{@{}lccccc" + ("lc" if J2 is not None else "l") + "@{}}", "\\toprule", head, "\\midrule"]
for row in ROWS:
    if row is None:
        lines.append("\\midrule"); continue
    m, name, lab, extra = row; d = g.get_group(m)
    cells = [name, lab, extra, pct((d.p_rare > 0.5).mean()), pct((d.p_edit > 0.5).mean()), pct((d.p_defect > 0.5).mean()),
             f"\\textbf{{{pct(d.clean.mean())}}} [{pct(ci[m][0])}, {pct(ci[m][1])}]"]
    if J2 is not None:
        cells.append(pct(c2[m]) if m in c2.index else "--")
    lines.append(" & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]; (tabs / "test.tex").write_text("\n".join(lines))
n_cases, n_subj = R.groupby(["id", "edit_seed"]).ngroups, R.subject.nunique()

# by attribute type
FL = {"color": "Colour", "pattern": "Pattern", "material": "Material", "occupation_gender": "Gender", "age_activity": "Age", "state": "State"}
MM = [("min_prompt", "Min.\\ prompt"), ("hid_attn25", "Hidden steer."), ("vel_attn25", "Vel.\\ steer."), ("lora", "LoRA"), ("sdedit_k1", "SDEdit")]
pv = R.pivot_table(index="flag", columns="method", values="clean", aggfunc="mean"); nn = R[R.method == "baseline"].groupby("flag").size()
lines = ["\\begin{tabular}{@{}lr" + "c" * len(MM) + "@{}}", "\\toprule", "Type & $n$ & " + " & ".join(n for _, n in MM) + " \\\\", "\\midrule"]
for f, name in FL.items():
    lines.append(f"{name} & {nn[f]} & " + " & ".join(pct(pv.loc[f, m]) for m, _ in MM) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]; (tabs / "types.tex").write_text("\n".join(lines))

# ---------------------------------------------------------------- dev: source injection under two judges
D = pd.read_parquet("sdedit_dev2/judge2.parquet")
for j, (a, b, c) in {"1": ("p_rare", "p_edit", "p_defect"), "2": ("p_rare2", "p_edit2", "p_defect2")}.items():
    D["kept" + j] = D[a] > 0.5; D["edited" + j] = D[b] > 0.5; D["clean" + j] = D["kept" + j] & D["edited" + j] & (D[c] <= 0.5)
DR = [("min_prompt", "Minimal attribute prompt"), ("detailed", "Detailed prompt"), ("rep1_full_subj_b5_g1", "Source injection + edit guidance"),
      ("rep1_full_subj_b3", "Source injection"), ("sdedit_k1", "SDEdit, skip 1 step"), ("baseline", "Vanilla prompt")]
dg = D.groupby("method")[["kept1", "edited1", "clean1", "kept2", "edited2", "clean2"]].mean()
lines = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule", " & \\multicolumn{3}{c}{Judge 1} & \\multicolumn{3}{c}{Judge 2} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
         "Method & Kept & Edited & Clean & Kept & Edited & Clean \\\\", "\\midrule"]
for m, name in DR:
    v = dg.loc[m]; lines.append(f"{name} & {pct(v.kept1)} & {pct(v.edited1)} & \\textbf{{{pct(v.clean1)}}} & {pct(v.kept2)} & {pct(v.edited2)} & \\textbf{{{pct(v.clean2)}}} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]; (tabs / "dev_judges.tex").write_text("\n".join(lines))
agree = {c: (D[c + "1"] == D[c + "2"]).mean() for c in ["kept", "edited"]}

def examples():
    """Needs the dataset images and the generated test images (not shipped in the repo)."""
    # ---------------------------------------------------------------- qualitative grid: first test case (by id) of four attribute types, no selection on outcomes
    meta = pd.read_json("data/metadata.jsonl", lines=True).set_index(["id", "edit_seed"])
    COLS = [("source", "Source"), ("baseline", "Vanilla prompt"), ("hid_attn25", "Hidden steering"), ("lora", "LoRA"), ("sdedit_k1", "SDEdit"), ("min_prompt", "Minimal prompt")]
    PATH = {"baseline": "final_test/img/{i}_{s}_baseline.jpg", "hid_attn25": "final_test/img/{i}_{s}_hid_attn25.jpg", "min_prompt": "baselines_test/img/{i}_{s}_min_prompt.jpg",
            "sdedit_k1": "baselines_test/img/{i}_{s}_sdedit_k1.jpg", "lora": "eval_lora/img/test_8000_{i}_{s}_lora.jpg"}
    S = R.set_index(["id", "edit_seed", "method"])
    picks = [sorted(set(zip(*[R[R.flag == f][c] for c in ("id", "edit_seed")])))[0] for f in ["color", "pattern", "material", "occupation_gender"]]
    fig, ax = plt.subplots(len(picks), len(COLS), figsize=(7.0, 1.36 * len(picks)))
    for i, (cid, seed) in enumerate(picks):
        m = meta.loc[(cid, seed)]
        for j, (key, title) in enumerate(COLS):
            a = ax[i, j]; a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            im = Image.open("data/" + m.source_image).convert("RGB").resize((256, 256)) if key == "source" else Image.open(PATH[key].format(i=cid, s=seed))
            a.imshow(im)
            if i == 0:
                a.set_title(title, fontsize=7.5, color=INK, pad=3)
            if key == "source":
                a.set_xlabel(f"{m.subject}: {m.rare_attribute}\n(typical: {m.typical_attribute})", fontsize=6.5, color=INK, labelpad=2)
            else:
                s = S.loc[(cid, seed, key)]; kept, ed, de = s.p_rare > 0.5, s.p_edit > 0.5, s.p_defect > 0.5
                a.set_xlabel("✓ attribute kept" if kept and ed and not de else "✗ attribute lost" if not kept else "✗ edit lost" if not ed else "✗ defect", fontsize=6.5, color=INK, labelpad=2)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.07, wspace=0.04, hspace=0.42); plt.savefig(figs / "examples.pdf", dpi=200); plt.close()
    print("picks:", picks)


if os.path.exists("data/metadata.jsonl") and os.path.isdir("final_test/img"):
    examples()
else:
    print("examples figure skipped: images not available")

print(f"test cases {n_cases}, subjects {n_subj}; step-swap n {len(r)} values {[round(v, 2) for v in vals]}")
print("probe: best block", t0.loc[t0.loso_acc.idxmax(), "layer"], round(t0.loso_acc.max(), 2), "| velocity", round(vrow.loso_acc, 2))
print("dev judge agreement:", {k: round(v, 3) for k, v in agree.items()}, "| cases", D.groupby(["id", "edit_seed"]).ngroups)
print("judge2 on test:", J2 is not None)
for f in ["test", "types", "dev_judges"]:
    print(f"--- {f}.tex"); print((tabs / f"{f}.tex").read_text())

# ---------------------------------------------------------------- dev: the image-side ceiling (ladder.py, depth.py, steer2.py), 95 cases with an own-dh oracle
def dev(dirname, methods=None):
    x = pd.read_parquet(f"{dirname}/results.parquet")
    x = x if methods is None else x[x.method.isin(methods)]
    x = x.assign(clean=(x.p_rare > 0.5) & (x.p_edit > 0.5) & (x.p_defect <= 0.5))
    return x[["id", "edit_seed", "subject", "flag", "method", "clean"]].assign(src=dirname)


DV = pd.concat([dev("sdedit_dev2", ["min_prompt", "detailed", "baseline", "rep1_full_subj_b5_g1", "rep1_full_subj_b3", "sdedit_k1"]),
                dev("ladder_dev", ["tok_all_a1", "svd16_all_a1", "svd4_all_a1", "svd1_all_a1"]), dev("depth_dev", ["oracle_A_a2", "oracle_E_a2", "label_D_a4", "label_A_a4"]),
                dev("steer2_dev", ["oracle_a2", "ridge_a8", "route_a4"]), dev("text_dev", ["A_D_a1"]), dev("refguid_dev", ["noref_all_w2_k2"]), dev("txtref_dev", ["txt_tgt_g8"])])
orc = set(zip(*[DV[DV.method == "oracle_A_a2"][c] for c in ("id", "edit_seed")]))
SUB = DV[[k in orc for k in zip(DV.id, DV.edit_seed)]]
CEIL = [("detailed", "Detailed prompt", "text"), ("min_prompt", "Minimal attribute prompt", "text"),
        ("tok_all_a1", "Exact patch, every token", "image"), ("svd16_all_a1", "Patch, rank 16", "image"), ("svd4_all_a1", "Patch, rank 4", "image"),
        ("svd1_all_a1", "Patch, rank 1", "image"), ("oracle_A_a2", "Own pooled vector, S4-S12", "image"), ("oracle_E_a2", "Own pooled vector, all 25 blocks", "image"),
        ("label_D_a4", "Label vector, S0-S19", "image")]
cv = SUB.groupby("method").clean.mean(); cc = boot_ci(SUB, "clean"); n_sub = SUB.groupby(["id", "edit_seed"]).ngroups
fig, ax = plt.subplots(figsize=(3.4, 2.6))
y = np.arange(len(CEIL))[::-1]
for yi, (m, name, kind) in zip(y, CEIL):
    ax.barh(yi, cv[m], height=0.62, color=ORANGE if kind == "text" else BLUE)
    ax.plot([cc[m][0], cc[m][1]], [yi, yi], color=INK, lw=0.7)
    ax.text(cc[m][1] + 0.02, yi, pct(cv[m]), va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels([n for _, n, _ in CEIL], fontsize=7); ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0%", "50%", "100%"])
ax.set_xlabel(f"clean success, dev ({n_sub} cases)")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=ORANGE, label="changes the text tokens"), Patch(color=BLUE, label="changes image tokens only")], fontsize=6.5, frameon=False,
           loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=2, handlelength=1.2, columnspacing=1.0)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(figs / "ceiling.pdf"); plt.close()

# ---------------------------------------------------------------- dev: label-free attempts, all 121 cases (best setting of each family on these cases: optimistic)
LF = [("min_prompt", "Minimal attribute prompt", "label"), ("detailed", "Detailed prompt", "label"), ("label_A_a4", "Hidden-state steering, S4--S12", "label"), None,
      ("rep1_full_subj_b5_g1", "Source injection + edit guidance", "no"), ("rep1_full_subj_b3", "Source injection", "no"), ("sdedit_k1", "SDEdit, skip 1 step", "no"),
      ("ridge_a8", "Correction predicted from reference", "no"), ("A_D_a1", "Text-token steering (SHIFT-style)", "contrast pairs"),
      ("noref_all_w2_k2", "Reference guidance", "no"), ("route_a4", "Probe names attribute, then steering", "no"), ("txt_tgt_g8", "Text-to-reference attention boost", "no"),
      ("baseline", "Vanilla prompt", "no")]
ALL = DV.copy(); lv = ALL.groupby("method").clean.mean(); lc = boot_ci(ALL, "clean"); ln = ALL.groupby("method").size()
lines = ["\\begin{tabular}{@{}llrl@{}}", "\\toprule", "Method & Label & $n$ & Clean [95\\% CI] \\\\", "\\midrule"]
for row in LF:
    if row is None:
        lines.append("\\midrule"); continue
    m, name, lab = row
    lines.append(f"{name} & {lab} & {ln[m]} & \\textbf{{{pct(lv[m])}}} [{pct(lc[m][0])}, {pct(lc[m][1])}] \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]; (tabs / "labelfree.tex").write_text("\n".join(lines))
print("ceiling cases", n_sub, {m: pct(cv[m]) for m, _, _ in CEIL})
print("--- labelfree.tex"); print((tabs / "labelfree.tex").read_text())
