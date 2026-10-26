"""Train DeltaDINO (SPEC §5)."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from second_look import metrics as M
from second_look.dataset import PairDataset, augment, make_loader, patch_masks, pixel_arm
from second_look.eval import DEVICE, git_commit, score_model
from second_look.model import DeltaDINO, paint_arm_pixels
from second_look.sampler import SPLITS_PATH, splits_sha256
from second_look.scene import ROOT

DEFAULTS = {
    "model": {"lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05, "mask_mode": "feature", "use_static": False},
    "train": {"batch": 32, "epochs": 15, "patience": 3, "lr_head": 1e-3, "lr_lora": 2e-4, "wd": 0.01,
              "warmup": 500, "clip": 1.0, "lam": 0.5, "workers": 4, "epoch_draws": 0, "labels": "v1", "static_file": None},
    "out_dir": "results/part1/deltadino",
}


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(open(path)) if path else {}
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in cfg.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) else v
    for kv in overrides or []:                     # e.g. train.lam=0 model.lora_rank=16
        key, val = kv.split("=", 1)
        sec, _, name = key.partition(".")
        val = yaml.safe_load(val)
        if name:
            out[sec][name] = val
        else:
            out[sec] = val
    return out


def param_groups(model: DeltaDINO, cfg: dict):
    t = cfg["train"]
    groups = {"head_decay": [], "head_nodecay": [], "lora_decay": [], "lora_nodecay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = "lora" if "lora_" in n else "head"
        nodecay = n.endswith(".bias") or "lora_B" in n          # no weight decay on biases / LoRA B
        groups[f"{kind}_{'nodecay' if nodecay else 'decay'}"].append(p)
    lr = {"head": t["lr_head"], "lora": t["lr_lora"]}
    return [{"params": ps, "lr": lr[k.split("_")[0]], "weight_decay": 0.0 if "nodecay" in k else t["wd"], "name": k}
            for k, ps in groups.items() if ps]


def lr_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


def val_metrics(model, ds_val) -> dict:
    sc = score_model(model, "val", ds=ds_val, include_gray=False)
    meta = ds_val.meta.iloc[sc["row"]].reset_index(drop=True)
    y, s = meta["label"].to_numpy(), sc["score"]
    out = {"val_auroc": M.auroc(y, s), **{f"val_{k}": v for k, v in M.image_metrics(y, s, 0.0).items() if k in ("ap", "f1")}}
    far = M.hard_negative_far(meta, s, 0.0)           # logit 0 == probability 0.5
    out["val_h1h2_far"] = far["H1+H2"]["false_alarm"]
    return out


def run(cfg: dict, seed: int, out: Path | None = None, max_steps: int | None = None, quiet: bool = False) -> dict:
    t = cfg["train"]
    out = Path(out or ROOT / cfg["out_dir"] / f"seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed), np.random.seed(seed)
    dev = DEVICE
    model = DeltaDINO(**cfg["model"]).to(dev)
    counts = model.param_counts()
    print(f"[params] trainable {counts['trainable']:,} (lora {counts['lora']:,} + head {counts['head']:,}); total {counts['total']/1e6:.2f}M", flush=True)
    if cfg["model"]["lora_rank"] == 8:
        assert abs(counts["trainable"] / 0.89e6 - 1) <= 0.05, "trainable count outside ±5% of 0.89M"

    ds_tr, ds_va = PairDataset("train", labels=t["labels"]), PairDataset("val", labels=t["labels"])
    if cfg["model"].get("use_static"):
        from second_look.static_mask import compute_static, load_static_file
        sm = load_static_file(t["static_file"]) if t.get("static_file") else compute_static()
        model.static.copy_(torch.from_numpy(sm).to(dev))
        import hashlib
        cfg["static_sha256"] = hashlib.sha256(np.packbits(sm).tobytes()).hexdigest()
        print(f"[static] {int(model.static.sum())} of 256 patches masked ({'file ' + str(t['static_file']) if t.get('static_file') else 'never contain a changed object in train'}); mask mode {model.mask_mode}; sha256 {cfg['static_sha256'][:12]}", flush=True)
    draws = t["epoch_draws"] or len(ds_tr)
    steps_per_epoch = draws // t["batch"]
    total = steps_per_epoch * t["epochs"]
    dl = make_loader(ds_tr, t["batch"], True, t["workers"], seed, n_draws=draws)
    y_all = torch.from_numpy(ds_tr.meta["label"].to_numpy()).float()

    opt = torch.optim.AdamW(param_groups(model, cfg), betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(t["warmup"], total))
    meta_ck = {"config": cfg, "git": git_commit(), "splits_sha256": splits_sha256(), "seed": seed, "params": counts}

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(str(out / "tb"))
    log_path = out / "log.csv"
    best, bad, step = -1.0, 0, 0
    fields = ["epoch", "step", "train_loss", "train_loss_img", "train_loss_patch", "val_auroc", "val_ap", "val_f1",
              "val_h1h2_far", "lr_head", "lr_lora", "vram_gb", "img_per_s", "secs"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0, n_img = time.time(), 0
        li = lp = lt = 0.0
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for ref, st, mk, row in dl:
            ref, st, mk = ref.to(dev, non_blocking=True), st.to(dev, non_blocking=True), mk.to(dev, non_blocking=True)
            if model.static is not None:        # warp the static region together with the images
                sp = F.interpolate(model.static.float()[None, None], scale_factor=14, mode="nearest").expand(mk.shape[0], 1, 224, 224)
                mk = torch.cat([mk, sp.to(mk.dtype)], 1)
            ref, st, mk = augment(ref, st, mk)
            arm, chg = patch_masks(mk[:, :4])
            if model.static is not None:
                st_w = F.avg_pool2d(mk[:, 4:5], 14).squeeze(1) >= 0.5          # static region after the same warp as the images
                arm = st_w if model.mask_mode == "static" else arm | st_w
            if model.mask_mode == "pixel":
                a_r, a_s = pixel_arm(mk)
                ref, st = paint_arm_pixels(ref, a_r), paint_arm_pixels(st, a_s)
            y = y_all[row].to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16):
                s, z = model(ref, st, arm)
            loss_img = F.binary_cross_entropy_with_logits(s.float(), y)
            keep = torch.ones_like(arm) if model.mask_mode == "none" else ~arm
            bce = F.binary_cross_entropy_with_logits(z.float(), chg.float(), reduction="none")
            per_img = (bce * keep).flatten(1).sum(1) / keep.flatten(1).sum(1).clamp(min=1)
            loss_patch = per_img.mean()
            loss = loss_img + t["lam"] * loss_patch
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], t["clip"])
            opt.step(), sched.step()
            step += 1
            n_img += 2 * len(y)
            li, lp, lt = li + loss_img.item(), lp + loss_patch.item(), lt + loss.item()
            if step % 200 == 0 and not quiet:
                print(f"  ep{epoch} step {step}  loss {loss.item():.4f}  ({n_img / (time.time() - t0):.0f} img/s)", flush=True)
            if max_steps and step >= max_steps:
                break
        n_it = max(1, step - (epoch - 1) * steps_per_epoch) if not max_steps else step
        secs = time.time() - t0
        vm = val_metrics(model, ds_va)
        row_log = {"epoch": epoch, "step": step, "train_loss": lt / n_it, "train_loss_img": li / n_it, "train_loss_patch": lp / n_it,
                   **vm, "lr_head": opt.param_groups[0]["lr"], "lr_lora": opt.param_groups[-1]["lr"],
                   "vram_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0, "img_per_s": n_img / secs, "secs": secs}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row_log)
        for k, v in row_log.items():
            if k not in ("epoch", "step"):
                tb.add_scalar(k, v, epoch)
        print("[epoch {epoch}] loss {train_loss:.4f} | val AUROC {val_auroc:.4f} AP {val_ap:.4f} F1 {val_f1:.4f} H1+H2 FAR@0.5 {val_h1h2_far:.3f} | {img_per_s:.0f} img/s, {vram_gb:.1f} GB".format(**row_log), flush=True)
        ck = {**meta_ck, "state": model.trainable_state(), "epoch": epoch, "val_auroc": vm["val_auroc"]}
        torch.save(ck, out / "last.pt")
        if vm["val_auroc"] > best:
            best, bad = vm["val_auroc"], 0
            torch.save(ck, out / "best.pt")
        else:
            bad += 1
            if bad >= t["patience"]:
                print(f"[early stop] no val AUROC gain for {t['patience']} epochs", flush=True)
                break
        if max_steps and step >= max_steps:
            break
    tb.close()
    return {"best_val_auroc": best, "out": str(out), "params": counts}
