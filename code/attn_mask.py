"""Label-free subject mask: attention from target-image tokens to the subject word's tokens in the vanilla edit prompt.

Single-stream sequence = [text 512, target 1024, ref 1024]. For the wrapped blocks we recompute the attention
probabilities of the target queries over all keys and keep the mass that lands on the subject tokens: a (1024,) map.
"""
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_flux2 import apply_rotary_emb

N_TGT, N_TXT = 1024, 512


def subject_token_ids(tokenizer, prompt, subject, max_len=N_TXT):
    """Token positions of `subject` (last occurrence) in the chat-templated prompt, as the pipeline encodes it."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
    lo = text.lower().rfind(subject.lower())
    if lo < 0:                                           # e.g. edit prompt says "the person" for subject "ballet dancer"
        for w in sorted(subject.lower().split(), key=len, reverse=True):
            lo = text.lower().rfind(w)
            if lo >= 0:
                subject = w; break
    if lo < 0:                                           # "Turn the person into ...": the noun after "the"
        import re
        m = list(re.finditer(r"\b[Tt]he (\w+)", text))
        if m:
            lo, subject = m[0].start(1), m[0].group(1)
    if lo < 0:                                           # "Make it look like ...": the pronoun stands for the subject
        import re
        m = list(re.finditer(r"\bit\b", text))
        if not m:
            return []
        lo, subject = m[-1].start(), "it"
    hi = lo + len(subject)
    return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < hi and b > lo and b > a]


class AttnProbe:
    """Wraps the attention processors of single-stream blocks; when `ids` is set, records attention-to-subject maps."""

    def __init__(self, transformer, blocks):
        self.ids, self.maps = None, {}
        for b in blocks:
            attn = transformer.single_transformer_blocks[b].attn
            attn.set_processor(self._wrap(attn.processor, b))

    def _wrap(self, proc, b):
        probe = self

        class Wrapped:
            def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None, **kw):
                if probe.ids:
                    qkv = attn.to_qkv_mlp_proj(hidden_states)[..., : 3 * attn.inner_dim]
                    q, k, _ = qkv.chunk(3, dim=-1)
                    q = attn.norm_q(q.unflatten(-1, (-1, attn.head_dim))); k = attn.norm_k(k.unflatten(-1, (-1, attn.head_dim)))
                    q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1); k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
                    qt = q[:, N_TXT:N_TXT + N_TGT].transpose(1, 2)               # (1, H, 1024, d)
                    kt = k.transpose(1, 2)                                       # (1, H, L, d)
                    p = torch.softmax((qt @ kt.transpose(-1, -2)).float() / attn.head_dim ** 0.5, -1)
                    probe.maps[b] = p[0][:, :, probe.ids].sum(-1).mean(0).cpu()   # (1024,)
                return proc(attn, hidden_states, attention_mask=attention_mask, image_rotary_emb=image_rotary_emb, **kw)
        return Wrapped()


def top_mask(m, frac=0.1):
    return m >= m.quantile(1 - frac)
