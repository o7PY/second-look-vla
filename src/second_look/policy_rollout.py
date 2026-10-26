"""SmolVLA rollouts on LIBERO-Goal with exact state recording (SPEC §4.4, pass 1 of S3).

Mirrors lerobot's eval rollout (same processors, same order) and the old project's deterministic renderer
(MSAA off + 2x supersampling). We record the full MuJoCo state, the executed action and the gripper
integrator at EVERY control step, plus the episode's model XML, so pass 2 can restore any moment in our own
Scene exactly like a human demo.
"""

from __future__ import annotations

import numpy as np
import torch

SMOLVLA = "HuggingFaceVLA/smolvla_libero"
SUITE = "libero_goal"
SIZE = 360
SUPERSAMPLE = 2


def patch_renderer() -> None:
    """MSAA off + deterministic 2x supersampling of every camera inside lerobot's LiberoEnv."""
    import cv2
    import lerobot.envs.libero as L
    from second_look import render
    render.patch()
    if getattr(L, "_sl_ss", False):
        return
    cv2.setNumThreads(1)
    orig_off, orig_fmt = L.OffScreenRenderEnv, L.LiberoEnv._format_raw_obs

    def scaled_env(*a, **kw):
        for k in ("camera_heights", "camera_widths"):
            if k in kw:
                kw[k] = int(kw[k]) * SUPERSAMPLE
        return orig_off(*a, **kw)

    def fmt(self, raw_obs):
        out = orig_fmt(self, raw_obs)
        h, w = self.observation_height, self.observation_width
        for k, img in list(out["pixels"].items()):
            if img.shape[0] != h or img.shape[1] != w:
                out["pixels"][k] = cv2.resize(np.ascontiguousarray(img), (w, h), interpolation=cv2.INTER_AREA)
        return out

    L.OffScreenRenderEnv, L.LiberoEnv._format_raw_obs = scaled_env, fmt
    L._sl_ss = True


class Policy:
    def __init__(self, device: str = "cuda", n_action_steps: int = 10):
        from lerobot.configs import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import EnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors
        env_cfg = EnvConfig.get_choice_class("libero")(task=SUITE, task_ids=None, observation_height=SIZE, observation_width=SIZE)
        cfg = PreTrainedConfig.from_pretrained(SMOLVLA)
        cfg.pretrained_path, cfg.device, cfg.n_action_steps = SMOLVLA, device, n_action_steps
        if hasattr(cfg, "compile_model"):
            cfg.compile_model = False
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=SMOLVLA,
            preprocessor_overrides={"device_processor": {"device": device}, "rename_observations_processor": {"rename_map": {}}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)

    def act(self, raw_obs: dict, task: str) -> np.ndarray:
        from lerobot.envs.utils import preprocess_observation
        add = lambda o: {k: add(v) for k, v in o.items()} if isinstance(o, dict) else (o[None] if isinstance(o, np.ndarray) else o)
        obs = preprocess_observation(add(raw_obs))
        obs["task"] = [task]
        batch = self.pre(self.env_pre(obs))
        with torch.inference_mode():
            a = self.policy.select_action(batch)
        a = self.env_post({"action": self.post(a)})["action"]
        return a.to("cpu").numpy()[0]

    def reset(self):
        self.policy.reset()


def make_env(task_id: int, episode_index: int):
    from lerobot.envs.libero import LiberoEnv, _get_suite
    return LiberoEnv(_get_suite(SUITE), task_id=task_id, task_suite_name=SUITE, obs_type="pixels_agent_pos",
                     observation_width=SIZE, observation_height=SIZE, episode_index=episode_index)


def rollout(env, pol: Policy, seed: int = 0) -> dict:
    """One episode. Returns states (T+1,79), actions (T,7), grip (T+1,2), xml, success."""
    obs, info = env.reset(seed=seed)
    pol.reset()
    renv = _renv(env)
    sim, grip = renv.sim, renv.robots[0].gripper
    xml = sim.model.get_xml()
    flat = lambda: sim.get_state().flatten().copy()
    states, grips, actions = [flat()], [grip.current_action.copy()], []
    success, max_steps = False, int(env._max_episode_steps)
    for _ in range(max_steps):
        a = pol.act(obs, env.task_description)
        obs, _, terminated, truncated, info = env.step(a)
        actions.append(a.astype(np.float64))
        success |= bool(info.get("is_success", False))
        if terminated or truncated:     # LiberoEnv auto-resets on success: do not record the post-reset state
            break
        states.append(flat()), grips.append(grip.current_action.copy())
    n = len(actions)
    states, grips = np.stack(states[: n + 1]), np.stack(grips[: n + 1])
    if len(states) < n + 1:             # final state after success is lost to the auto-reset: drop the last action
        actions = actions[: len(states) - 1]
    return {"states": states, "actions": np.stack(actions), "grip": grips, "xml": xml, "success": success}


def _renv(env):
    e = env
    while not (hasattr(e, "robots") and hasattr(e, "timestep")):
        e = getattr(e, "_env", None) or getattr(e, "env")
    return e
