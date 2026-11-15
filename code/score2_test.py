"""Re-score the main test-table methods with the second judge (Qwen3-VL-8B). Writes judge2_test.parquet."""
import pandas as pd
from PIL import Image

from analyze_baselines import load
from judge2 import Judge2

SRC = {"baseline": "final_test/img/{i}_{s}_baseline.jpg", "hid_attn25": "final_test/img/{i}_{s}_hid_attn25.jpg", "vel_attn25": "final_test/img/{i}_{s}_vel_attn25.jpg",
       "detailed": "final_test/img/{i}_{s}_detailed.jpg", "min_prompt": "baselines_test/img/{i}_{s}_min_prompt.jpg", "textguid_s2": "baselines_test/img/{i}_{s}_textguid_s2.jpg",
       "sdedit_k1": "baselines_test/img/{i}_{s}_sdedit_k1.jpg", "lora": "eval_lora/img/test_8000_{i}_{s}_lora.jpg"}
r = load(); keys = sorted(set(zip(r.id, r.edit_seed)))
meta = pd.read_json("data/metadata.jsonl", lines=True).set_index(["id", "edit_seed"])
J = Judge2(); rows = []
for n, (cid, seed) in enumerate(keys):
    m = meta.loc[(cid, seed)]; src = Image.open("data/" + m.source_image).convert("RGB"); src.thumbnail((448, 448))
    names = list(SRC); ims = [Image.open(SRC[k].format(i=cid, s=seed)).convert("RGB") for k in names]; k = len(ims)
    pr = J.attr(ims, [m.subject] * k, [m.rare_attribute] * k, [m.typical_attribute] * k); pe = J.edit([src] * k, ims, [m.edit_prompt] * k); pdf = J.defect(ims, [m.subject] * k)
    rows += [dict(id=cid, edit_seed=seed, subject=m.subject, flag=m.flag, method=nm, p_rare2=float(pr[j]), p_edit2=float(pe[j]), p_defect2=float(pdf[j])) for j, nm in enumerate(names)]
    if n % 20 == 0:
        pd.DataFrame(rows).to_parquet("judge2_test.parquet"); print(n, len(keys), flush=True)
pd.DataFrame(rows).to_parquet("judge2_test.parquet"); print("done")
