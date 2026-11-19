# Rare attributes are lost in the text stream

Code and per-case results for the short paper **"Rare Attributes Are Lost in the Text Stream: Prior-Driven Drift in
Reference-Based Editing with FLUX.2 [klein]"** ([`paper/main.pdf`](paper/main.pdf)).

When FLUX.2 [klein] 4B edits a photo of a subject with a rare attribute (a purple banana), the result often reverts to
the typical one (a yellow banana). We locate where this happens and test fixes that leave the prompt unchanged.

| Finding | Number | Where |
|---|---|---|
| Vanilla edits that lose the rare attribute (dev, dataset seed) | 84% | Sec. 2 |
| Detailed prompt at steps 0-1 only restores it | 97% | Fig. 1a |
| Model's own activations name the reference's rare attribute (unseen subjects, top-1 / top-3) | 69% / 92% | Sec. 3 |
| Held-out test, clean success: detailed / minimal prompt | 92% / 87% | Table 1 |
| Held-out test: hidden-state steering (no prompt change) / LoRA (no label) | 71% / 47% | Table 1 |
| Exact patch of the image tokens at S4-S12 (oracle) vs. detailed prompt | 61% vs. 99% | Fig. 3 |
| Best label-free, training-free method (dev) | 57% | Table 3 |

## Layout

```
code/      all scripts; run from this folder (they read data/ and write next to themselves)
results/   per-case judge scores (parquet/csv) behind every table and figure
paper/     LaTeX source, figures, tables and the compiled PDF
```

## Setup

One GPU with at least 40 GB (an A6000 was used), Python 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd code
hf download NinaK/shift_dataset --repo-type dataset --local-dir data   # gated: request access first
```

Models are downloaded on first use: `black-forest-labs/FLUX.2-klein-4B` (editor), `Qwen/Qwen2.5-VL-7B-Instruct`
(judge 1) and `Qwen/Qwen3-VL-8B-Instruct` (judge 2).

## Rebuilding the paper's tables and figures from the shipped results

```bash
cp -r results/* code/ && cd code && python make_paper.py   # writes code/paper/figs and code/paper/tables
```

This reproduces all four tables exactly and the figures except the example images, which need the dataset and the
generated images (not in the repo); without them that figure is skipped.

## Running the experiments

Scripts are listed in the order they depend on each other. Each one has a docstring with its arguments.

| Step | Script | Produces | Paper |
|---|---|---|---|
| Core | `velocity.py` | manual 4-step edit loop with access to the velocity | all |
| | `hidden.py`, `attn_mask.py` | hooks on the 25 blocks; subject mask from attention to the subject word | all |
| | `judge.py`, `judge2.py` | VLM judges (Qwen2.5-VL-7B, Qwen3-VL-8B) | all |
| 1. Drift | `judge.py --split dev --out judge_dev.parquet`, then `drift_table.py` | `judge_dev.parquet`, `drift_table_dev.parquet` | Sec. 2 |
| 2. Analysis | `exp_step0.py`, `analyze_step0.py` | step swap | Fig. 1a |
| | `collect_dv.py`, `collect_h.py`, `analyze_h.py` | corrections per step and block, `layer_scores.csv` | Fig. 1b |
| | `refdir_collect.py`, `refprobe.py`, `ridge_cos2.py` | attribute read from reference activations | Sec. 3 |
| 3. Test | `steer_final.py --mode test`, `rescore_defects.py` | hidden / velocity steering | Table 1 |
| | `baselines.py`, `analyze_baselines.py` | prompt, guidance, SDEdit, attention baselines | Table 1-2 |
| | `train_lora.py`, `eval_lora.py`, `analyze_lora.py` | velocity-matching LoRA | Table 1-2 |
| | `score2_test.py`, `judge2_table.py` | second-judge rescoring | Table 1 |
| 4. Ceiling (dev) | `ladder.py` | exact and low-rank patching oracles | Fig. 3 |
| | `steer2.py`, `depth.py` (+ `analyze_*.py`) | own-vector oracle, block ranges, label-free directions | Fig. 3, Table 3 |
| 5. Label-free (dev) | `sdedit_plus2.py`, `score2_sdedit.py` | SDEdit and source injection | Table 3, App. |
| | `text_steer.py`, `refguid.py`, `txtref.py` | text-token steering, reference guidance, attention boost | Table 3 |
| Paper | `make_paper.py` | figures and tables | all |

## KOALA: correction adapter + source pull

KOALA keeps the plain instruction and needs no label: a small trained adapter (`adapter.py`) adds a predicted teacher
correction to the velocity at every step, and after step 0 the source pull of `sdedit_plus2.py` (`rep1_full_subj_b2_g1`)
moves the subject tokens toward the reference. 7 editor forwards.

| Script | Produces |
|---|---|
| `collect_adapter.py` | teacher corrections at the states of plain rollouts of training subjects (`adapter_data/`) |
| `train_adapter.py` | the adapter (`adapter.pt`) |
| `koala.py --ckpt adapter.pt` | KOALA and vanilla edits of the val subjects with judge-1 scores (`koala_val/`) |

Subjects come from `koala_split.json` (a re-split of the released dev + test pairs by subject: 63 train / 39 val / 39
test), so the teacher choice needs `judge.py` scores of both splits. The trained adapter used in the report (s 2, b 2,
g 1) is `/home/nina.konovalova/shift/experiments/20261118-2236_correction_train/heads/full_notext.pt` (MBZUAI cluster,
md5 `7563d0b067bc64831ae1e25952c5d67c`); `koala.py --ckpt` loads it as is. Its teachers were chosen with the judge
wording of our own pipeline rather than `judge.py`'s, so rerunning `collect_adapter.py` can pick slightly different
pairs.

## Protocol

Every method choice was made on the dev split. The test split (70 subjects) was used for two evaluation rounds (steering
with baselines, then the LoRA) and for rescoring the same images with the second judge. Scores come from VLM judges; the
two judges agree on "attribute kept" (kappa 0.76) but less on "edit applied" (kappa 0.46).

## Related

This work follows [SHIFT: Steering Hidden Intermediates in Flow Transformers](https://arxiv.org/abs/2604.09213)
(Konovalova, Idrisov, Alanov, 2026), which steers FLUX.1 for concept control in text-to-image generation.
