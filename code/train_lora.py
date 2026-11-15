"""Velocity-matching LoRA: make v(x_k, vanilla prompt, ref) match the frozen model's v(x_k, detailed prompt, ref), k in {0, 1}.

Data: dev sources (rare and common views) whose detailed edit is judged good at both seeds; 6 dev subjects held out for
validation (lora_val_subjects.json). Fresh noise every iteration; the teacher is computed online with LoRA disabled.
Usage: python train_lora.py --iters 3000 --out lora_r16
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from lora import add_lora, load_state, set_enabled, state, vel
from velocity import Editor

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--out", default="lora_r16"); ap.add_argument("--n_val_subjects", type=int, default=6)
ap.add_argument("--init", help="checkpoint to resume from"); ap.add_argument("--start", type=int, default=0)
ap.add_argument("--ckpt_every", type=int, default=1000)
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)
random.seed(a.start); torch.manual_seed(a.start)

meta = pd.read_json("data/metadata.jsonl", lines=True)
dev = meta[meta.split == "dev"].merge(pd.read_parquet("judge_dev.parquet"), on=["id", "edit_seed"])
dev["teacher_ok"] = (dev.p_edit_detailed > 0.5) & ((dev["view"] == "rare") & (dev.p_rare_detailed > 0.8) | (dev["view"] == "common") & (dev.p_rare_detailed < 0.2))
ok = dev.groupby("id").teacher_ok.all()
src = dev.drop_duplicates("id").set_index("id").loc[ok[ok].index].reset_index()
drift = pd.read_parquet("drift_table_dev.parquet")
subjects = sorted(drift[drift.fixed_detailed].subject.unique())
val_subjects = random.Random(0).sample(subjects, a.n_val_subjects)
json.dump(val_subjects, open("lora_val_subjects.json", "w"))
train = src[~src.subject.isin(val_subjects)].reset_index(drop=True)
print(f"sources: {len(src)} with a good teacher of {dev.id.nunique()} | train {len(train)} ({(train['view'] == 'rare').sum()} rare) "
      f"| val subjects {val_subjects}", flush=True)

E = Editor(); mods = add_lora(E.pipe.transformer, r=a.rank, alpha=a.rank)
if a.init:
    load_state(mods, torch.load(a.init))
params = [p for m in mods.values() for p in (m.A, m.B)]
print("trainable params:", sum(p.numel() for p in params), flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))
cache = {}


def text(p):
    if p not in cache:
        with torch.no_grad():
            e, i = E.encode_text(p)
        cache[p] = (e.cpu(), i.cpu())
    e, i = cache[p]
    return e.to(E.device), i.to(E.device)


run, log = [], []
for it in range(a.start + 1, a.iters + 1):
    r = train.iloc[random.randrange(len(train))]
    st = E.prepare(Image.open("data/" + r.source_image), random.randrange(2 ** 31))
    c_van, c_det = text(r.edit_prompt), text(r.detailed_prompt)
    k = random.randrange(2); x = st["x"]
    with torch.no_grad():
        if k == 1:                                                   # x_1 from the teacher or from the student, 50/50
            student = random.random() < 0.5
            set_enabled(mods, student)
            v0 = vel(E, x, st["ts"][0], c_van if student else c_det, st)
            x = (x.float() + (st["sigmas"][1] - st["sigmas"][0]) * v0.float()).to(v0.dtype)
        set_enabled(mods, False)
        v_t = vel(E, x, st["ts"][k], c_det, st).float()
    set_enabled(mods, True)
    v_s = vel(E, x, st["ts"][k], c_van, st).float()
    loss = torch.nn.functional.mse_loss(v_s, v_t)
    (loss / a.accum).backward()
    if it % a.accum == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    run.append((k, r["view"], loss.item()))
    if it % 100 == 0:
        df = pd.DataFrame(run[-100:], columns=["k", "view", "loss"])
        rec = dict(it=it, loss=df.loss.mean(), **{f"{v}_k{k}": df[(df["view"] == v) & (df.k == k)].loss.mean() for v in ["rare", "common"] for k in [0, 1]})
        log.append(rec); print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)
        pd.DataFrame(log).to_csv(out / f"loss_{a.start}.csv", index=False)
    if it % a.ckpt_every == 0 or it == a.iters:
        torch.save(state(mods), out / f"ckpt_{it}.pt")
print("done")
