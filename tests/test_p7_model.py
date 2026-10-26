"""Model / augmentation / mask / metric unit tests (CPU only)."""
import numpy as np
import pandas as pd
import pytest
import torch

from second_look import metrics as M
from second_look.dataset import augment, patch_masks
from second_look.model import DeltaDINO, smooth_max


@pytest.fixture(scope="module")
def model():
    return DeltaDINO(lora_rank=8).eval()


def test_param_count_within_5pct(model):
    c = model.param_counts()
    assert c["lora"] == 12 * 4 * 8 * (384 + 384)
    assert abs(c["trainable"] / 0.89e6 - 1) <= 0.05
    assert c["head"] == 1153 * 256 + 256 + 256 * 128 * 9 + 128 + 128 + 1


def test_only_lora_and_head_trainable(model):
    for n, p in model.named_parameters():
        assert p.requires_grad == (n.startswith("head.") or "lora_" in n), n


def test_smooth_max_properties():
    z = torch.full((1, 16, 16), 2.0)
    z[0, 0, 0] = 99.0
    keep = torch.ones(1, 16, 16, dtype=torch.bool)
    keep[0, 0, 0] = False
    assert float(smooth_max(z, keep)) == pytest.approx(2.0, abs=1e-5)       # masked patch ignored
    z2 = torch.zeros(1, 16, 16)
    z2[0, 5, 5] = 10.0
    ones = torch.ones_like(keep)
    s = float(smooth_max(z2, ones))
    assert 0.0 < s < 10.0 and s > 1.0                                        # between mean and max, pulled to the peak
    assert float(smooth_max(z2, torch.zeros_like(keep))) == -10.0            # nothing to look at


def test_arm_patches_cannot_influence_output(model, monkeypatch):
    """Head input is zeroed on arm patches, so arm-patch features must not change the image logit."""
    torch.manual_seed(0)
    B = 2
    f = torch.randn(2 * B, 384, 16, 16)
    arm = torch.zeros(B, 16, 16, dtype=torch.bool)
    arm[:, 3:7, 4:9] = True
    monkeypatch.setattr(model, "features", lambda x: f.clone())
    img = torch.zeros(B, 3, 224, 224)
    s0, _ = model(img, img, arm)
    f2 = f.clone()
    for half in (0, B):
        f2[half:half + B, :, 3:7, 4:9] = torch.randn(B, 384, 4, 5) * 5
    monkeypatch.setattr(model, "features", lambda x: f2.clone())
    s1, _ = model(img, img, arm)
    # 3x3 conv leaks arm-patch *zeros* only; the logit pools over non-arm patches whose receptive field touches the
    # (zeroed) arm region, so the outputs must be identical.
    assert torch.allclose(s0, s1, atol=1e-5)


def test_mask_channel_distinguishes_arm_from_empty(model, monkeypatch):
    f = torch.zeros(4, 384, 16, 16)
    monkeypatch.setattr(model, "features", lambda x: f.clone())
    img = torch.zeros(2, 3, 224, 224)
    arm = torch.zeros(2, 16, 16, dtype=torch.bool)
    arm[:, 8, 8] = True
    _, z = model(img, img, arm)
    assert not torch.allclose(z[:, 8, 8], z[:, 0, 0])       # same (zero) features, different mask channel


def test_trainable_state_roundtrip(model):
    sd = model.trainable_state()
    assert all(k.startswith("head.") or "lora_" in k for k in sd)
    other = DeltaDINO(lora_rank=8)
    other.load_trainable(sd)
    for k, v in other.trainable_state().items():
        assert torch.equal(v, sd[k])


def test_patch_masks_threshold_and_dilation():
    mk = torch.zeros(1, 4, 224, 224)
    mk[0, 0, 70:84, 70:84] = 1                              # exactly one patch (cells 5,5) of robot in ref
    mk[0, 2, 14:15, 14:28] = 1                              # 14 px of a patch = 1/14 = 7% < 10% -> not changed
    mk[0, 3, 0:14, 0:2] = 1                                 # 2/14 columns = 14% >= 10% -> changed
    arm, chg = patch_masks(mk)
    assert arm[0, 5, 5] and arm[0, 4:7, 4:7].all() and not arm[0, 3, 3]      # dilated by one patch
    assert not chg[0, 1, 1] and not chg[0, 1, 2]
    assert chg[0, 0, 0]


def test_augment_moves_images_and_masks_together():
    torch.manual_seed(1)
    B = 8
    m = torch.zeros(B, 4, 224, 224)
    for i in range(B):
        m[i, :, 40 + 5 * i:100 + 5 * i, 60:150] = 1
    ref = (m[:, 0:1].repeat(1, 3, 1, 1) * 255).to(torch.uint8)
    st = (m[:, 2:3].repeat(1, 3, 1, 1) * 255).to(torch.uint8)
    a, b, mk = augment(ref, st, m.to(torch.uint8))
    for i in range(B):
        for img, k in ((a, 0), (b, 2)):
            pred, gt = img[i].mean(0) > 0.5, mk[i, k] > 0.5
            iou = (pred & gt).sum() / max(1, (pred | gt).sum())
            assert iou > 0.95, (i, float(iou))              # same warp for image and mask
    assert not torch.equal(mk, m), "augmentation did nothing"


def test_augment_noise_is_independent_per_frame():
    torch.manual_seed(2)
    x = torch.full((4, 3, 224, 224), 128, dtype=torch.uint8)
    m = torch.zeros(4, 4, 224, 224, dtype=torch.uint8)
    a, b, _ = augment(x, x.clone(), m)
    assert (a - b).abs().mean() > 1e-3                       # identical inputs, independent noise -> they differ a little
    assert (a - b).abs().mean() < 0.02                       # ... but only by sensor-noise amounts


def test_best_threshold_and_bootstrap():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    s = np.r_[rng.normal(0, 1, 200), rng.normal(2, 1, 200)]
    thr = M.best_f1_threshold(y, s)
    assert 0.3 < thr < 1.7
    meta = pd.DataFrame({"task": np.repeat(np.arange(8), 50), "demo": np.tile(np.arange(5), 80)})
    lo, hi = M.bootstrap_ci(meta, y, s, thr, "auroc", n_boot=100)
    assert lo < M.auroc(y, s) < hi


def test_magnitude_bins_and_slices():
    meta = pd.DataFrame({"max_trans_cm": [0.2, 1.5, 3.0, 8.0, 0.0], "max_rot_deg": [0, 0, 0, 0, 20.0],
                         "gray": False, "label": [0, 1, 1, 1, 1], "gen": "G1", "source": "S2", "task": 0, "delta": 5})
    b = M.magnitude_bin(meta).tolist()
    assert b == ["none", "trans 1-2cm", "trans 2-5cm", "trans >5cm", "rot >15deg"]
    t = M.slice_table(meta, np.array([0.1, 0.9, 0.9, 0.9, 0.2]), 0.5, "magnitude")
    assert set(t["magnitude"]) == set(b)
