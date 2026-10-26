"""SmolVLA for the closed loop (SPEC_part2.md §6, P2.2): keyed noise per decision, the whole 10-action chunk of a decision at once, the Euler iterates of the
executed chunk (for accel) and K batched resamples that share one VLM prefix (for STAC).

Built on `second_look.policy_rollout` (same checkpoint `HuggingFaceVLA/smolvla_libero`, same processors in the same order as lerobot's LIBERO eval, same deterministic
2x-supersampled renderer). Differences, all deliberate:
  * the flow-matching noise is passed in (`second_look.seeding.decision_noise`), never drawn from a global RNG;
  * `plan` returns the 10 actions of one decision (`n_action_steps = 10`) in one call and leaves the policy's queues empty, so no policy state lives between
    decisions (nothing to snapshot except the detectors' own memory);
  * the denoise path of the executed chunk is recorded: `iterates` (T+1, chunk, 7) = noise, then every Euler iterate, in the policy's normalised action space,
    padded dimensions dropped (the same x_t that fmaccel's recorder stores);
  * `resample` runs K noises in one batch over ONE prefix pass (the prefix KV cache is expanded over the batch).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from second_look import policy_rollout as PR
from second_look.seeding import set_determinism

N_EXEC = 10
ACTION_DIM = 7
CHUNK = 50
MAX_DIM = 32


@dataclass
class Plan:
    actions: np.ndarray          # (N_EXEC, 7) environment actions (post-processed), in execution order
    iterates: np.ndarray | None  # (T+1, CHUNK, 7) float32, normalised space; None if not recorded
    chunk: np.ndarray            # (CHUNK, 7) the full normalised chunk (== iterates[-1] if recorded)


class SmolVLA:
    def __init__(self, device: str = "cuda", n_action_steps: int = N_EXEC):
        PR.patch_renderer()                               # MSAA off + 2x supersampling inside lerobot's LiberoEnv (idempotent)
        set_determinism()
        self.p = PR.Policy(device=device, n_action_steps=n_action_steps)
        set_determinism()                                 # AFTER the build: nothing may flip the flags back
        self.policy, self.device = self.p.policy, device
        self.model = self.policy.model
        self.n_action_steps = n_action_steps
        self.num_steps = int(self.policy.config.num_steps)

    # ------------------------------------------------------------------ observations
    def batch(self, raw_obs: dict, task: str) -> dict:
        """lerobot's chain: raw obs -> preprocess_observation -> +task -> env preprocessor -> policy preprocessor (B = 1)."""
        from lerobot.envs.utils import preprocess_observation
        add = lambda o: {k: add(v) for k, v in o.items()} if isinstance(o, dict) else (o[None] if isinstance(o, np.ndarray) else o)
        obs = preprocess_observation(add(raw_obs))
        obs["task"] = [task]
        return self.p.pre(self.p.env_pre(obs))

    def _post(self, a: torch.Tensor) -> np.ndarray:
        a = self.p.env_post({"action": self.p.post(a)})["action"]
        return a.to("cpu").numpy()[0]

    # ------------------------------------------------------------------ the executed chunk
    def plan(self, raw_obs: dict, task: str, noise: torch.Tensor, *, record: bool = True) -> tuple[Plan, dict]:
        """One decision: 10 actions from `noise` (1, CHUNK, MAX_DIM). Returns (Plan, batch); the batch is what `resample` needs for the same observation."""
        assert noise.shape == (1, CHUNK, MAX_DIM), noise.shape
        batch = self.batch(raw_obs, task)
        self.policy.reset()                               # empty queues: a decision never depends on an earlier one
        xs: list[torch.Tensor] = []
        if record:
            orig_step, orig_sample = self.model.denoise_step, self.model.sample_actions

            def step(prefix_pad_masks, past_key_values, x_t, timestep):
                xs.append(x_t.detach().float().cpu())
                return orig_step(prefix_pad_masks, past_key_values, x_t, timestep)

            def sample(*a, **kw):
                out = orig_sample(*a, **kw)
                xs.append(out.detach().float().cpu())
                return out

            self.model.denoise_step, self.model.sample_actions = step, sample
        try:
            with torch.inference_mode():
                first = self.policy.select_action(batch, noise=noise)
                rest = [self.policy._queues["action"].popleft() for _ in range(self.n_action_steps - 1)]
        finally:
            if record:
                self.model.__dict__.pop("denoise_step", None)
                self.model.__dict__.pop("sample_actions", None)
        assert len(self.policy._queues["action"]) == 0
        actions = np.stack([self._post(a) for a in [first] + rest])
        iterates = chunk = None
        if record:
            assert len(xs) == self.num_steps + 1, len(xs)
            iterates = torch.cat(xs, 0)[:, :, :ACTION_DIM].numpy()                 # (T+1, CHUNK, 7)
            chunk = iterates[-1].copy()
        return Plan(actions, iterates, chunk), batch

    # ------------------------------------------------------------------ STAC resamples
    @torch.inference_mode()
    def resample(self, batch: dict, noises: torch.Tensor) -> np.ndarray:
        """K chunks for the SAME observation from K noises (K, CHUNK, MAX_DIM): one prefix pass at B = 1, KV cache expanded over the batch, `num_steps`
        Euler steps at B = K. Returns (K, CHUNK, 7) float32 in the normalised space. `batch` is the dict `plan` returned (already queue-stacked)."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        K = noises.shape[0]
        m, p = self.model, self.policy
        images, img_masks = p.prepare_images(batch)
        state = p.prepare_state(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
        att2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, pkv = m.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None, inputs_embeds=[prefix_embs, None],
                                           use_cache=m.config.use_cache, fill_kv_cache=True)
        pkv = {l: {k: v.expand(K, *v.shape[1:]).contiguous() for k, v in d.items()} for l, d in pkv.items()}
        ppm = prefix_pad_masks.expand(K, -1).contiguous()
        dt = -1.0 / self.num_steps
        x = noises.to(self.device, torch.float32)
        for i in range(self.num_steps):
            t = torch.tensor(1.0 + i * dt, dtype=torch.float32, device=self.device).expand(K)
            x = x + dt * m.denoise_step(x_t=x, prefix_pad_masks=ppm, past_key_values=pkv, timestep=t)
        return x[:, :, :ACTION_DIM].float().cpu().numpy()
