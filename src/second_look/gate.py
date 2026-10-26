"""The second look: frames and DeltaDINO (model C) inside the loop (SPEC_part2.md §2.2, §6).

`Frames` owns the ONE render Scene of a worker (separate from the policy's own LIBERO env, so rendering can never touch the policy's simulation, G4):
a frame is `Scene.set_flat(flat)` + `render.render_rgb(scene)` of a stored/live flat sim state, the same two entry points Part 1 used (restore only through
`Scene.set_flat`, render only through `render.render_rgb`). The simulator truth between two states (`oracle`) is read from the same Scene's registry.

`DeltaGate` scores a (reference, settled) pair with model C of a given seed at batch 1 (bf16 autocast, as second_look.eval): C masks the static region only and
ignores the arm, so no arm silhouette is needed (the arm-patch input is zeros).
"""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from second_look import imgcheck, render
from second_look.oracle import oracle_label
from second_look.scene import ROOT, Scene

CKPT = "results/part1/deltadino_v2_static/seed{}/best.pt"


class Frames:
    def __init__(self):
        self.scene: Scene | None = None
        self._xml_hash: str | None = None

    def bind(self, task_idx: int, xml: str) -> Scene:
        """Make the render Scene match the live policy env (task and episode XML). Reloads the model only when the XML differs."""
        if self.scene is not None and self.scene.task_idx != task_idx:
            self.close()                                     # destroys its GL context NOW (gc), not in the middle of someone else's render
            self._xml_hash = None
        if self.scene is None:
            self.scene = Scene(task_idx)
        h = hashlib.sha256(xml.encode()).hexdigest()
        if h != self._xml_hash:
            self.scene.load_demo_model(xml)
            self._xml_hash = h
        return self.scene

    def render_flat(self, flat: np.ndarray) -> np.ndarray:
        self.scene.set_flat(flat)
        img = render.render_rgb(self.scene)
        imgcheck.check_image(img, f"Frames.render_flat (task {self.scene.task_idx})")
        return img

    def nq(self) -> int:
        return int(self.scene.m.nq)

    def oracle(self, flat_a: np.ndarray, flat_b: np.ndarray) -> dict:
        """`label_v2` between two flat sim states ([time, qpos, qvel])."""
        nq = self.nq()
        return oracle_label(self.scene.registry, np.asarray(flat_a)[1:1 + nq], np.asarray(flat_b)[1:1 + nq])

    def close(self) -> None:
        if self.scene is not None:
            render.activate(self.scene)                      # freeing a render context frees GL objects in whatever context is CURRENT: make it this scene's own
            self.scene.env.close()                           # (freed in the policy env's context, it deleted that env's textures: found 2026-10-10, tests/p2 G1)
            self.scene = None
            gc.collect()


class DeltaGate:
    def __init__(self, device: str = "cuda", root: Path = ROOT):
        self.device, self.root = device, root
        self._models: dict[int, torch.nn.Module] = {}
        self._zero_arm = torch.zeros(1, 16, 16, dtype=torch.bool, device=device)

    def model(self, seed: int):
        if seed not in self._models:
            from second_look.eval import load_model
            m, _ = load_model(str(self.root / CKPT.format(seed)), self.device)
            assert m.mask_mode == "static", m.mask_mode
            self._models[seed] = m.eval()
        return self._models[seed]

    @torch.no_grad()
    def score(self, ref_u8: np.ndarray, set_u8: np.ndarray, seed: int = 0) -> tuple[float, np.ndarray]:
        """(image logit s, 16x16 change-map logits z) for (224,224,3) uint8 frames."""
        m = self.model(seed)
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].to(self.device).float() / 255.0
        with torch.autocast(self.device, dtype=torch.bfloat16):
            s, z = m(f(ref_u8), f(set_u8), m.apply_static(self._zero_arm))
        return float(s.float().item()), z.float()[0].cpu().numpy()
