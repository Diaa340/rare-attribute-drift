"""Manual FLUX.2 [klein] 4B edit loop with access to the velocity v(x_t, c, ref) at every step.

Flow matching: x_t = (1 - s) x0 + s * eps,  v = eps - x0,  so x0_hat = x_t - s * v and x_{next} = x_t + (s_next - s) v.
"""
import numpy as np, torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

MODEL = "black-forest-labs/FLUX.2-klein-4B"

class Editor:
    def __init__(self, device="cuda", size=512, steps=4):
        self.pipe = Flux2KleinPipeline.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device, self.size, self.steps = device, size, steps

    @torch.no_grad()
    def encode_text(self, prompt):
        return self.pipe.encode_prompt(prompt=prompt, device=self.device, num_images_per_prompt=1,
                                       max_sequence_length=512, text_encoder_out_layers=(9, 18, 27))

    @torch.no_grad()
    def prepare(self, source, seed):
        """Noise, reference-image latents and sigma schedule, identical to the pipeline."""
        p, s = self.pipe, self.size
        img = p.image_processor.preprocess(source.convert("RGB").resize((s, s), Image.LANCZOS), height=s, width=s, resize_mode="crop")
        ref, ref_ids = p.prepare_image_latents(images=[img], batch_size=1, generator=torch.Generator(self.device).manual_seed(seed),
                                               device=self.device, dtype=p.vae.dtype)
        x, x_ids = p.prepare_latents(batch_size=1, num_latents_channels=p.transformer.config.in_channels // 4, height=s, width=s,
                                     dtype=torch.bfloat16, device=self.device, generator=torch.Generator(self.device).manual_seed(seed), latents=None)
        sig = np.linspace(1.0, 1 / self.steps, self.steps)
        ts, _ = retrieve_timesteps(p.scheduler, self.steps, self.device, sigmas=sig, mu=compute_empirical_mu(image_seq_len=x.shape[1], num_steps=self.steps))
        sigmas = p.scheduler.sigmas.clone().to(self.device)          # len steps + 1, ends at 0
        return dict(x=x, x_ids=x_ids, ref=ref, ref_ids=ref_ids, ts=ts, sigmas=sigmas)

    @torch.no_grad()
    def velocity(self, x, t, text, st):
        emb, ids = text
        out = self.pipe.transformer(hidden_states=torch.cat([x, st["ref"]], 1).to(torch.bfloat16), timestep=t.expand(1).to(x.dtype) / 1000,
                                    guidance=None, encoder_hidden_states=emb, txt_ids=ids,
                                    img_ids=torch.cat([st["x_ids"], st["ref_ids"]], 1), return_dict=False)[0]
        return out[:, : x.shape[1]]                                     # bf16, as in the pipeline

    @torch.no_grad()
    def run(self, st, text, probe_texts=None, steer=None, text_per_step=None):
        """Integrate with `text` (or text_per_step[i]). At every step also evaluate v for probe_texts at the same x_t.
        steer(i, x, v) -> v lets you modify the velocity. Returns final latents and a per-step record."""
        x = st["x"]; rec = []                                           # bf16 latents, as in the pipeline
        for i, t in enumerate(st["ts"]):
            s, s_next = st["sigmas"][i], st["sigmas"][i + 1]
            c = text_per_step[i] if text_per_step else text
            v = self.velocity(x, t, c, st)
            r = dict(step=i, sigma=float(s), x=x.float().cpu(), v=v.float().cpu(), x0=(x.float() - s * v.float()).cpu())
            for k, pt in (probe_texts or {}).items():
                vp = self.velocity(x, t, pt, st); r[f"v_{k}"] = vp.float().cpu(); r[f"x0_{k}"] = (x.float() - s * vp.float()).cpu()
            if steer is not None: v = steer(i, x, v)
            r["v_used"] = v.float().cpu(); rec.append(r)
            x = (x.float() + (s_next - s) * v.float()).to(v.dtype)        # FlowMatchEulerDiscreteScheduler.step
        return x, rec

    @torch.no_grad()
    def decode(self, x, st):
        p = self.pipe; x = x.to(self.device, torch.bfloat16); s = self.size
        lh = 2 * (s // (p.vae_scale_factor * 2))
        x = p._unpack_latents_with_ids(x, st["x_ids"], lh // 2, lh // 2)
        mean = p.vae.bn.running_mean.view(1, -1, 1, 1).to(x); std = torch.sqrt(p.vae.bn.running_var.view(1, -1, 1, 1) + p.vae.config.batch_norm_eps).to(x)
        img = p.vae.decode(p._unpatchify_latents(x * std + mean), return_dict=False)[0]
        return p.image_processor.postprocess(img, output_type="pil")[0]

def token_map(t, side=32):
    """(1, N, C) token tensor -> (side, side) L2-norm map."""
    return t[0].norm(dim=-1).reshape(side, side).numpy()
