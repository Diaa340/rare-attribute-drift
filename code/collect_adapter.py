"""Teacher corrections for the KOALA adapter (adapter.py): states of plain-instruction rollouts of training subjects.

Training subjects: koala_split.json "train", a re-split of the released dev + test pairs by subject, so judge.py must
have scored both splits (also python judge.py --split test --out judge_test.parquet). Teacher per (pair, seed): the
detailed prompt if its edits pass on both views (rare view kept and applied: p_edit >= 0.5, p_rare >= p_rare(source) / 2;
common view applied and typical: p_edit >= 0.5, p_rare <= 0.3), else the manual prompt if it passes, else the pair is
dropped. For both views at both seeds: the plain
rollout and, at each of its 4 states x, the plain v and dv = v(x, teacher) - v(x, plain); the subject mask M (top 25%
of the attention to the subject word, S13-S19, step 0), the adapter's mask input. Saves adapter_data/{id}_{seed}.pt.
Usage: python collect_adapter.py --shard 0 --nshards 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from attn_mask import AttnProbe, subject_token_ids, top_mask
from velocity import Editor

ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
out = Path("adapter_data"); out.mkdir(exist_ok=True)
meta = pd.read_json("data/metadata.jsonl", lines=True)
judge = pd.concat([pd.read_parquet(f"judge_{s}.parquet") for s in ("dev", "test")])
d = meta[meta.subject.isin(json.load(open("koala_split.json"))["train"])].merge(judge, on=["id", "edit_seed"])
for t in ("detailed", "manual"):
    ok = (d[f"p_edit_{t}"] >= 0.5) & np.where(d["view"] == "rare", d[f"p_rare_{t}"] >= d.p_rare_source / 2, d[f"p_rare_{t}"] <= 0.3)
    d[f"ok_{t}"] = ok.groupby([d.pair_id, d.edit_seed]).transform("all")
d["koala_teacher"] = np.where(d.ok_detailed, d.detailed_prompt, np.where(d.ok_manual, d.manual_prompt, None))
d = d[d.koala_teacher.notna()].reset_index(drop=True)
print(f"{len(d)} items of {d.subject.nunique()} subjects, teacher detailed {d.ok_detailed.mean():.2f}", flush=True)
d = d.iloc[a.shard::a.nshards]

E = Editor(); P = AttnProbe(E.pipe.transformer, range(13, 20))
for n, r in enumerate(d.itertuples()):
    f = out / f"{r.id}_{r.edit_seed}.pt"
    if f.exists():
        continue
    st = E.prepare(Image.open("data/" + r.source_image), int(r.edit_seed)); tv = E.encode_text(r.edit_prompt)
    P.ids, P.maps = subject_token_ids(E.pipe.tokenizer, r.edit_prompt, r.subject), {}
    E.velocity(st["x"], st["ts"][0], tv, st); P.ids = None
    mask = top_mask(torch.stack(list(P.maps.values())).mean(0), 0.25) if P.maps else torch.ones(1024, dtype=torch.bool)
    _, rec = E.run(st, tv, probe_texts={"t": E.encode_text(r.koala_teacher)})
    torch.save(dict(x=torch.cat([q["x"] for q in rec]).bfloat16(), v=torch.cat([q["v"] for q in rec]).bfloat16(),
                    dv=torch.cat([q["v_t"] - q["v"] for q in rec]).half(), ref=st["ref"][0].cpu(), m=mask,
                    sigma=st["sigmas"][:4].float().cpu()), f)
    if n % 100 == 0:
        print(a.shard, n, len(d), flush=True)
print("done", a.shard)
