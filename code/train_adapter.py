"""Train the KOALA adapter (adapter.py) on the teacher corrections of collect_adapter.py; the editor is not loaded.

Only the adapter is trained. Each sample is a random item at a random step; loss = item mean over all 1024 tokens of
|adapter - dv|^2, divided by that step's mean squared teacher correction over the training items (so the late, small
steps count); M is an input, not a loss weight. Batch 16, AdamW lr 1e-3 / wd 1e-4, 100-step warm-up, then cosine.
3500 iterations = the best iteration of a 4000-iteration run with 13 of the 62 training subjects held out; the shipped
head is this fit on all of them.
Usage: python train_adapter.py --out adapter.pt
"""
import argparse
import math
from pathlib import Path

import torch

from adapter import Adapter, positions

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=3500); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--out", default="adapter.pt")
a = ap.parse_args()
torch.manual_seed(0)

items = [torch.load(f) for f in sorted(Path("adapter_data").glob("*.pt"))]
D = {k: torch.stack([it[k] for it in items]).cuda() for k in ("x", "v", "dv", "ref", "m")}
sig, pos, n = items[0]["sigma"].cuda(), positions().cuda(), len(items)
del items
step_ms = torch.cat([D["dv"][j:j + 64].float().square().sum(-1).mean(-1) for j in range(0, n, 64)]).mean(0)   # (4,)
print(n, "items, step mean |dv|^2", step_ms.tolist(), flush=True)


def batch(i, k):
    """Adapter inputs and target dv of items i at steps k."""
    x, v, s = D["x"][i, k].float(), D["v"][i, k].float(), sig[k]
    return (D["ref"][i].float(), x - s.view(-1, 1, 1) * v, v, D["m"][i].float()[..., None], s, pos), D["dv"][i, k].float()


A = Adapter().cuda()
f = A.feats(*batch(torch.randperm(n)[:128], torch.randint(4, (128,)))[0]).flatten(0, 1)      # input statistics
A.mu.copy_(f.mean(0)); A.sd.copy_(f.std(0, correction=0).clamp_min(1e-4)); del f
opt = torch.optim.AdamW(A.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 100) * 0.5 * (1 + math.cos(math.pi * t / a.iters)))
for it in range(1, a.iters + 1):
    k = torch.randint(4, (a.batch,))
    inp, dv = batch(torch.randint(n, (a.batch,)), k)
    loss = ((A(*inp) - dv).square().sum(-1).mean(1) / step_ms[k]).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 100 == 0:
        print(it, round(loss.item(), 4), flush=True)
torch.save({"state": A.state_dict()}, a.out)
print("done")
