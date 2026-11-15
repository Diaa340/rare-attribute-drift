"""VLM judge: for each image, P(rare attribute) vs P(typical attribute), and P(edit applied).
Usage: python judge.py --split dev --out judge_dev.parquet [--limit N]"""
import argparse, torch, pandas as pd
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).parent / "data"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VARIANTS = {"source": "source_image", "vanilla": "edited_image", "detailed": "detailed_image", "manual": "manual_image"}

ATTR_Q = ("Look at the {subject} in this image. It may be shown as a toy, cartoon, figurine, drawing or photo. "
          "Ignoring the style, which description fits the {subject} better?\nA: {a}\nB: {b}\nAnswer with only the letter A or B.")
EDIT_Q = ("The first image is the original. The second image was supposed to be produced by this edit: \"{edit}\". "
          "Was the edit applied in the second image? Answer only yes or no.")

def load(p, size=448):
    p = p["path"] if isinstance(p, dict) else p
    im = Image.open(ROOT / p).convert("RGB"); im.thumbnail((size, size)); return im

class Judge:
    def __init__(self, device="cuda"):
        self.proc = AutoProcessor.from_pretrained(MODEL, min_pixels=128*28*28, max_pixels=320*28*28)
        self.proc.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=device).eval()
        tok = self.proc.tokenizer
        self.ids = {k: tok.encode(k, add_special_tokens=False)[0] for k in ["A", "B", "yes", "no", "Yes", "No"]}

    @torch.no_grad()
    def _logits(self, msgs_imgs):
        texts = [self.proc.apply_chat_template([{"role": "user", "content": [*([{"type": "image"}] * len(ims)), {"type": "text", "text": q}]}],
                                               tokenize=False, add_generation_prompt=True) for q, ims in msgs_imgs]
        imgs = [im for _, ims in msgs_imgs for im in ims]
        inp = self.proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(self.model.device)
        return self.model(**inp).logits[:, -1].float()

    def attr(self, images, subjects, rares, typicals):
        """P(rare), averaged over both answer orders."""
        q1 = [(ATTR_Q.format(subject=s, a=r, b=t), [im]) for im, s, r, t in zip(images, subjects, rares, typicals)]
        q2 = [(ATTR_Q.format(subject=s, a=t, b=r), [im]) for im, s, r, t in zip(images, subjects, rares, typicals)]
        A, B = self.ids["A"], self.ids["B"]
        l1, l2 = self._logits(q1), self._logits(q2)
        p1 = torch.softmax(l1[:, [A, B]], -1)[:, 0]   # rare is A
        p2 = torch.softmax(l2[:, [A, B]], -1)[:, 1]   # rare is B
        return ((p1 + p2) / 2).cpu().numpy()

    def edit(self, sources, edited, edits):
        q = [(EDIT_Q.format(edit=e), [s, x]) for s, x, e in zip(sources, edited, edits)]
        l = self._logits(q)
        y = torch.logsumexp(l[:, [self.ids["yes"], self.ids["Yes"]]], -1); n = torch.logsumexp(l[:, [self.ids["no"], self.ids["No"]]], -1)
        return torch.sigmoid(y - n).cpu().numpy()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="dev"); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int); ap.add_argument("--bs", type=int, default=16); a = ap.parse_args()
    df = pd.read_json(ROOT / "metadata.jsonl", lines=True)
    df = df[df.split == a.split].reset_index(drop=True)
    if a.limit: df = df.head(a.limit)
    J = Judge(); out = []
    for i in range(0, len(df), a.bs):
        b = df.iloc[i:i + a.bs]
        src = [load(v) for v in b.source_image]
        rec = {"id": b.id.values, "edit_seed": b.edit_seed.values}
        for name, col in VARIANTS.items():
            ims = src if name == "source" else [load(v) for v in b[col]]
            rec[f"p_rare_{name}"] = J.attr(ims, b.subject.tolist(), b.rare_attribute.tolist(), b.typical_attribute.tolist())
            if name != "source": rec[f"p_edit_{name}"] = J.edit(src, ims, b.edit_prompt.tolist())
        out.append(pd.DataFrame(rec)); print(f"{i + len(b)}/{len(df)}", flush=True)
        pd.concat(out).to_parquet(a.out)
