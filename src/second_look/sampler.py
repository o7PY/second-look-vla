"""Pair construction and per-episode balanced sampling (SPEC §4)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from second_look import render, settle
from second_look.generators import GENERATORS, HARD_NEGATIVES, REGISTRY
from second_look.scene import DEMO_DIR, ROOT, Scene, label_pair, task_names

LABELER = label_pair            # swapped to second_look.labels_v2.label_pair_v2 by the pair builders of Part 1.5 / before-Part-2 (new data carries v2 labels directly)
DELTAS = (5, 10, 20, 40)
EDGE = 10                        # exclude first / last 10 steps of an episode
FLOOR_Z = -0.05                  # objects below this have left the world
MAX_PEN = 0.02                   # interpenetration > 2 cm -> unstable


@dataclass(frozen=True)
class Spec:
    source: str                  # "S1" | "S2" | "S3"
    task: int
    demo: int
    t: int
    delta: int
    k: int
    gen: str                     # "S1" | G1..G5 | H1..H4
    variant: int = 0

    @property
    def seed(self) -> int:
        key = "|".join(str(x) for x in (self.source, self.task, self.demo, self.t, self.delta, self.k, self.gen, self.variant))
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & (2**63 - 1)

    @property
    def pair_id(self) -> str:
        return f"{self.source}-{self.task}-{self.demo}-{self.t}-{self.delta}-{self.gen}-{self.variant}"


class Demo:
    """One episode (human demo or policy rollout): states, actions, optional recorded gripper state, and
    per-step 'is something held' flags."""

    def __init__(self, scene: Scene, h5: h5py.File | None, demo_idx: int, *, xml: str | None = None,
                 states: np.ndarray | None = None, actions: np.ndarray | None = None, grip: np.ndarray | None = None):
        self.idx = demo_idx
        if h5 is not None:
            g = h5[f"data/demo_{demo_idx}"]
            xml, states, actions = g.attrs["model_file"], g["states"][:], g["actions"][:]
        scene.load_demo_model(xml)
        self.states, self.actions, self.grip = states, actions, grip
        self.T = len(self.actions)
        self.held = np.zeros(self.T, dtype=bool)
        for t in range(self.T):
            scene.set_flat(self.states[t])
            self.held[t] = bool(settle.holding(scene))


def _unstable(scene: Scene) -> bool:
    if not scene.finite():
        return True
    d = scene.d
    if d.ncon and d.contact[: d.ncon].dist.min() < -MAX_PEN:
        return True
    return any(scene.d.qpos[it.qadr + 2] < FLOOR_Z for it in scene.registry if it.kind == "free")


def _visible(item_img: np.ndarray, idx: int) -> int:
    return int((item_img == idx).sum())


def build_pair(scene: Scene, masks: render.Masks, demo: Demo, spec: Spec, ks: tuple[int, ...] | None = None) -> dict | None:
    """Build one (ref, settled) pair, or several settled frames (one per k in ``ks``, same disturbance).

    Returns None if the generator rejects, the sim goes unstable, or a post-hoc requirement fails.
    The result for k is ``out["frames"][k]``; ``out["ref"]`` is shared."""
    ks = tuple(sorted(ks or (spec.k,)))
    t0 = spec.t - spec.delta
    assert t0 >= 0, spec
    rng = np.random.default_rng(spec.seed)

    settle.restore(scene, demo.states, demo.actions, t0, demo.grip)
    r_ref = scene.read()
    f_ref = render.render_frame(scene, masks)

    g = settle.restore(scene, demo.states, demo.actions, spec.t, demo.grip)
    res = REGISTRY[spec.gen](scene, demo, rng, spec.t, g)
    if res is None or _unstable(scene):
        return None

    frames, done = {}, 0
    for k in ks:
        settle.pause(scene, k - done, res["g"])
        done = k
        if _unstable(scene):
            return None
        r_set = scene.read()
        f_set = render.render_frame(scene, masks)
        lab = LABELER(scene, r_ref, r_set)
        chg_idx = [i for i, it in enumerate(scene.registry) if it.name in lab["changed_items"]]
        chg_ref = np.isin(f_ref["item"], chg_idx)
        chg_set = np.isin(f_set["item"], chg_idx)
        if res.get("check") and res["check"][0] == "occlusion":
            ii = next(i for i, it in enumerate(scene.registry) if it.name == res["check"][1])
            a, b = _visible(f_ref["item"], ii), _visible(f_set["item"], ii)
            if max(a, b) == 0 or abs(a - b) / max(a, b) < 0.3:
                return None
        frames[k] = {"rgb": f_set["rgb"], "robot": f_set["robot"], "chg": chg_set, "label": lab}
        frames[k]["chg_ref"] = chg_ref
    return {"spec": spec, "ref": {"rgb": f_ref["rgb"], "robot": f_ref["robot"]}, "frames": frames}


def pair_record(out: dict, k: int) -> tuple[dict, dict]:
    """(arrays, meta-row) for one k of a build_pair() result."""
    spec: Spec = out["spec"]
    fr = out["frames"][k]
    lab = fr["label"]
    arrays = {"ref": out["ref"]["rgb"], "set": fr["rgb"], "arm_ref": out["ref"]["robot"], "arm_set": fr["robot"],
              "chg_ref": fr["chg_ref"], "chg_set": fr["chg"]}
    meta = {**asdict(spec), "k": k, "pair_id": spec.pair_id, "seed": spec.seed, "label": lab["label"],
            "gray": lab["gray"], "max_trans_cm": 100 * lab["max_trans"], "max_rot_deg": lab["max_rot"],
            "changed_items": ",".join(lab["changed_items"])}
    return arrays, meta


# ---- per-episode quota plan ---------------------------------------------------------------------
@dataclass
class Quota:
    s1_pos: int = 30
    s1_neg: int = 30
    g_each: int = 6                      # G1..G5
    h_counts: tuple = (7, 8, 7, 8)       # H1..H4 (sum 30)
    max_gray: int = 4                    # gray pairs kept per episode (test splits only)
    max_draws: int = 40                  # draw attempts per wanted pair before giving up


def _candidate_t(rng: np.random.Generator, demo: Demo, delta: int, need_held: bool | None = None) -> int | None:
    lo, hi = max(EDGE, delta), demo.T - EDGE
    if hi <= lo:
        return None
    ts = np.arange(lo, hi)
    if need_held is True:
        ts = ts[demo.held[ts]]
    elif need_held is False:
        ts = ts[~demo.held[ts]]
    return int(rng.choice(ts)) if len(ts) else None


def sample_episode(scene: Scene, masks: render.Masks, demo: Demo, task: int, keep_gray: bool,
                   quota: Quota = Quota(), k: int = 10, s1_source: str = "S1") -> tuple[list, list, dict]:
    """All pairs for one episode. Returns (arrays list, meta list, stats)."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"ep|{task}|{demo.idx}".encode()).digest()[:8], "big"))
    arrays, metas = [], []
    stats = {g: {"tried": 0, "ok": 0, "pos": 0, "neg": 0, "gray": 0, "rejected": 0, "relabel_drop": 0}
             for g in ("S1",) + GENERATORS + HARD_NEGATIVES}

    def attempt(gen: str, source: str, variant: int, need_held=None):
        delta = int(rng.choice(DELTAS))
        t = _candidate_t(rng, demo, delta, need_held)
        st = stats[gen]
        st["tried"] += 1
        if t is None:
            st["rejected"] += 1
            return None
        spec = Spec(source, task, demo.idx, t, delta, k, gen, variant)
        out = build_pair(scene, masks, demo, spec)
        if out is None:
            st["rejected"] += 1
            return None
        return out

    def take(out, gen: str, want: int | None, grays: list) -> bool:
        """Accept a built pair if its MEASURED label matches ``want`` (None = either). Returns accepted."""
        a, m = pair_record(out, k)
        st = stats[gen]
        if m["gray"]:
            st["gray"] += 1
            if keep_gray and len(grays) < quota.max_gray:
                grays.append((a, m))
            return False
        if want is not None and m["label"] != want:
            st["relabel_drop"] += 1
            return False
        arrays.append(a), metas.append(m)
        st["ok"] += 1
        st["pos" if m["label"] else "neg"] += 1
        return True

    grays: list = []
    # S1: balanced by measured label
    need = {1: quota.s1_pos, 0: quota.s1_neg}
    draws, cap = 0, quota.max_draws * (quota.s1_pos + quota.s1_neg)
    while (need[0] > 0 or need[1] > 0) and draws < cap:
        draws += 1
        out = attempt("S1", s1_source, draws)
        if out is None:
            continue
        lab = out["frames"][k]["label"]
        if not lab["gray"] and need[lab["label"]] <= 0:
            stats["S1"]["relabel_drop"] += 1
            continue
        if take(out, "S1", None, grays):
            need[metas[-1]["label"]] -= 1

    # S2 positives (G1..G5) and hard negatives (H1..H4) with equal shares
    def fill(gen: str, n: int, want: int, need_held=None):
        got, tries = 0, 0
        while got < n and tries < quota.max_draws * n:
            tries += 1
            out = attempt(gen, "S2", tries, need_held)
            if out is not None and take(out, gen, want, grays):
                got += 1
        return got

    short = 0
    for gen in GENERATORS:
        got = fill(gen, quota.g_each, 1, need_held=True if gen == "G3" else None)
        short += quota.g_each - got
    # tasks without a carry phase can't make G3: give the shortfall to the other generators in turn
    order = ["G1", "G4", "G2", "G5"]
    i = 0
    while short > 0 and i < 4 * quota.g_each:
        short -= fill(order[i % 4], 1, 1)
        i += 1
    for gen, n in zip(HARD_NEGATIVES, quota.h_counts):
        fill(gen, n, 0, need_held=False if gen in ("H1", "H2") else None)

    for a, m in grays:
        arrays.append(a), metas.append(m)
    return arrays, metas, stats


# ---- splits (written once, never regenerated) ---------------------------------------------------
HELDOUT_TASK_NAMES = ("put_the_bowl_on_the_plate", "put_the_cream_cheese_in_the_bowl")
SPLITS_PATH = ROOT / "data" / "splits.json"


def make_splits(seed: int = 0) -> dict:
    """Split by demo episode. Held-out tasks go whole to ``test_heldout_tasks``; for the other tasks
    each task's 50 demos split 35 / 8 / 7 (70% / 16% / 14%) into train / val / test."""
    names = task_names()
    held = [names.index(n) for n in HELDOUT_TASK_NAMES]
    rng = np.random.default_rng(seed)
    out = {"seed": seed, "task_names": names, "heldout_tasks": held, "episodes": {}}
    for ti, name in enumerate(names):
        with h5py.File(DEMO_DIR / f"{name}_demo.hdf5", "r") as h:
            idxs = sorted(int(k.split("_")[1]) for k in h["data"].keys())
        if ti in held:
            for i in idxs:
                out["episodes"][f"{ti}:{i}"] = "test_heldout_tasks"
            continue
        perm = rng.permutation(idxs)
        for j, i in enumerate(perm):
            out["episodes"][f"{ti}:{int(i)}"] = "train" if j < 35 else ("val" if j < 43 else "test")
    return out


def load_or_make_splits() -> dict:
    if SPLITS_PATH.exists():
        return json.loads(SPLITS_PATH.read_text())
    s = make_splits()
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(s, indent=1, sort_keys=True))
    return s


def splits_sha256() -> str:
    return hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest()
