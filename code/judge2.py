"""Second, independent VLM judge (Qwen3-VL-8B, a different model generation from the Qwen2.5-VL-7B judge) with the same questions."""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from judge import ATTR_Q, EDIT_Q

MODEL2 = "Qwen/Qwen3-VL-8B-Instruct"
DEFECT_Q = ("Look at the {subject} in this image. Does it have visual defects, such as missing or erased facial features, "
            "smeared or washed-out regions, broken or fragmented parts that look like glitches, or a malformed shape? "
            "Answer only yes or no.")
SAME_Q = ("The first image is the original and shows a {subject}. The second image is an edited version, possibly in a different style or view. "
          "Ignoring the change of style, does the {subject} in the second image still have the same distinctive appearance as in the first image "
          "(same colour, pattern, material, age, gender, pose and condition)? Answer only yes or no.")


class Judge2:
    def __init__(self, device="cuda", bs=8):
        self.proc = AutoProcessor.from_pretrained(MODEL2); self.proc.tokenizer.padding_side = "left"; self.bs = bs
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL2, dtype=torch.bfloat16, device_map=device).eval()
        tok = self.proc.tokenizer
        self.ids = {k: tok.encode(k, add_special_tokens=False)[0] for k in ["A", "B", "yes", "no", "Yes", "No"]}

    @torch.no_grad()
    def _logits(self, msgs_imgs):
        out = []
        for i in range(0, len(msgs_imgs), self.bs):
            chunk = msgs_imgs[i:i + self.bs]
            texts = [self.proc.apply_chat_template([{"role": "user", "content": [*([{"type": "image"}] * len(ims)), {"type": "text", "text": q}]}],
                                                   tokenize=False, add_generation_prompt=True) for q, ims in chunk]
            imgs = [im for _, ims in chunk for im in ims]
            inp = self.proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(self.model.device)
            out.append(self.model(**inp).logits[:, -1].float())
        return torch.cat(out)

    def _yes(self, q):
        l = self._logits(q)
        return torch.sigmoid(torch.logsumexp(l[:, [self.ids["yes"], self.ids["Yes"]]], -1) - torch.logsumexp(l[:, [self.ids["no"], self.ids["No"]]], -1)).cpu().numpy()

    def attr(self, images, subjects, rares, typicals):
        A, B = self.ids["A"], self.ids["B"]
        l1 = self._logits([(ATTR_Q.format(subject=s, a=r, b=t), [im]) for im, s, r, t in zip(images, subjects, rares, typicals)])
        l2 = self._logits([(ATTR_Q.format(subject=s, a=t, b=r), [im]) for im, s, r, t in zip(images, subjects, rares, typicals)])
        return ((torch.softmax(l1[:, [A, B]], -1)[:, 0] + torch.softmax(l2[:, [A, B]], -1)[:, 1]) / 2).cpu().numpy()

    def edit(self, sources, edited, edits):
        return self._yes([(EDIT_Q.format(edit=e), [s, x]) for s, x, e in zip(sources, edited, edits)])

    def defect(self, images, subjects):
        return self._yes([(DEFECT_Q.format(subject=s), [im]) for im, s in zip(images, subjects)])

    def same(self, sources, edited, subjects):
        return self._yes([(SAME_Q.format(subject=sb), [s, x]) for s, x, sb in zip(sources, edited, subjects)])
