"""Part 1.5 tests: change-removal and arm-swap edits, pointing metrics (CPU only)."""
import numpy as np
import torch

from second_look import p15


def _batch(B=2):
    g = torch.Generator().manual_seed(0)
    ref = torch.rand(B, 3, 224, 224, generator=g)
    st = torch.rand(B, 3, 224, 224, generator=g)
    z = torch.zeros(B, 224, 224, dtype=torch.bool)
    return ref, st, z


def test_removal_with_empty_mask_is_identity():
    ref, st, z = _batch()
    out, m = p15.remove_change(ref, st, z, z, z, z)
    assert torch.equal(out, st) and int(m.sum()) == 0


def test_removal_copies_ref_inside_dilated_mask_outside_arm():
    ref, st, z = _batch(1)
    chg = z.clone(); chg[0, 100:110, 100:110] = True
    arm = z.clone(); arm[0, :, 120:] = True
    out, m = p15.remove_change(ref, st, chg, z, arm, z, dil=3)
    assert m[0, 98:112, 98:112].all() and not m[0, 95, 95]                      # dilated by 3 px
    assert torch.equal(out[0][:, 105, 105], ref[0][:, 105, 105])               # pasted from ref
    assert torch.equal(out[0][:, 20, 20], st[0][:, 20, 20])                    # elsewhere untouched
    arm_only = z.clone(); arm_only[0, 100:110, 100:110] = True                 # change fully under the arm: nothing to paste
    out2, m2 = p15.remove_change(ref, st, chg, z, arm_only, z, dil=0)
    assert int(m2.sum()) == 0 and torch.equal(out2, st)


def test_swap_arm_only_touches_new_arm_pixels():
    ref, st, z = _batch(1)
    arm_set = z.clone(); arm_set[0, 50:80, 50:80] = True
    arm_ref = z.clone(); arm_ref[0, 60:70, 60:70] = True
    out, m = p15.swap_arm(ref, st, arm_ref, arm_set)
    assert not m[0, 65, 65] and m[0, 55, 55]
    assert torch.equal(out[0][:, 55, 55], ref[0][:, 55, 55]) and torch.equal(out[0][:, 65, 65], st[0][:, 65, 65])


def test_cheb_distance_and_pointing_row():
    m = np.zeros((16, 16), bool); m[5, 5] = True
    assert p15.cheb_dist_to_set((5, 5), m) == 0 and p15.cheb_dist_to_set((6, 7), m) == 2 and p15.cheb_dist_to_set((1, 1), np.zeros((16, 16), bool)) is None

    class Fake:                                    # minimal Split stand-in
        arm = np.zeros((1, 16, 16), bool); chg = np.zeros((1, 16, 16), bool); pmap = np.full((1, 16, 16), -5.0)
        def argmax_patch(self, i):
            z = np.where(self.arm[i], -np.inf, self.pmap[i]); return np.unravel_index(int(np.argmax(z)), z.shape)
    f = Fake(); f.chg[0, 5, 5] = f.chg[0, 5, 6] = True; f.pmap[0, 5, 7] = 3.0     # argmax one patch beside the changed patches
    r = p15.pointing_row(f, 0)
    assert r["P_strict"] is False and r["P_adj"] is True and r["dist_argmax"] == 1 and r["P_visible"] is False and r["n_changed_visible"] == 2
    f.arm[0, 5, 5] = True                                                         # a changed patch under the arm is not "visible"
    assert p15.pointing_row(f, 0)["n_changed_visible"] == 1


def test_tilt_of_pure_up_axis_spin_is_zero_and_toppling_counts():
    from second_look.labels_v2 import compare_v2, tilt_deg
    from second_look.scene import Item, quat_angle_deg
    q0 = np.array([1.0, 0, 0, 0])
    for deg in (5.0, 30.0, 90.0, 170.0):                                       # spin about local z (= world up for an upright object)
        h = np.radians(deg) / 2
        q = np.array([np.cos(h), 0, 0, np.sin(h)])
        assert tilt_deg(q0, q) < 1e-3 and quat_angle_deg(q0, q) == __import__("pytest").approx(deg, abs=1e-6)
    for deg in (5.0, 30.0, 90.0):                                              # tipping about a horizontal axis
        h = np.radians(deg) / 2
        q = np.array([np.cos(h), np.sin(h), 0, 0])
        assert tilt_deg(q0, q) == __import__("pytest").approx(deg, abs=1e-6)
    # v2 labels: a 30deg spin of the bowl is unchanged, of the cream cheese changed, tipping the bottle 30deg changed
    items = [Item(n, "free", i, i, 7 * i, None) for i, n in enumerate(["akita_black_bowl_1_main", "cream_cheese_1_main", "wine_bottle_1_main"])]
    pos = np.array([0.0, 0, 0.9])
    spin = lambda deg: np.concatenate([pos, [np.cos(np.radians(deg) / 2), 0, 0, np.sin(np.radians(deg) / 2)]])
    tip = lambda deg: np.concatenate([pos, [np.cos(np.radians(deg) / 2), np.sin(np.radians(deg) / 2), 0, 0]])
    r0 = {n.name: np.concatenate([pos, q0]) for n in items}
    r1 = {"akita_black_bowl_1_main": spin(30), "cream_cheese_1_main": spin(30), "wine_bottle_1_main": np.concatenate([pos, q0])}
    st = {d.name: d.status for d in compare_v2(items, r0, r1)}
    assert st == {"akita_black_bowl_1_main": "same", "cream_cheese_1_main": "changed", "wine_bottle_1_main": "same"}
    r1["wine_bottle_1_main"] = tip(30)
    assert {d.name: d.status for d in compare_v2(items, r0, r1)}["wine_bottle_1_main"] == "changed"


def test_static_buffer_roundtrip_and_apply():
    import torch
    from second_look.model import DeltaDINO
    m = DeltaDINO(lora_rank=8, use_static=True)
    m.static[0, 11] = True; m.static[3, 3] = True
    sd = m.trainable_state()
    assert "static" in sd and all(k.startswith("head.") or "lora_" in k or k == "static" for k in sd)
    other = DeltaDINO(lora_rank=8, use_static=True); other.load_trainable(sd)
    assert torch.equal(other.static, m.static)
    arm = torch.zeros(2, 16, 16, dtype=torch.bool); arm[:, 8, 8] = True
    out = other.apply_static(arm)
    assert out[0, 0, 11] and out[1, 3, 3] and out[0, 8, 8] and int(out[0].sum()) == 3
    plain = DeltaDINO(lora_rank=8)
    assert plain.static is None and torch.equal(plain.apply_static(arm), arm) and "static" not in plain.trainable_state()


def test_static_region_is_warped_with_the_images():
    import torch
    import torch.nn.functional as F
    from second_look.dataset import augment
    torch.manual_seed(3)
    B = 8
    static = torch.zeros(16, 16); static[2:5, 9:13] = 1
    sp = F.interpolate(static[None, None], scale_factor=14, mode="nearest").expand(B, 1, 224, 224)
    img = (sp.repeat(1, 3, 1, 1) * 255).to(torch.uint8)                      # image that IS the static region
    m = torch.cat([torch.zeros(B, 4, 224, 224), sp], 1).to(torch.uint8)
    a, b, mk = augment(img, img.clone(), m)
    for i in range(B):
        pred, gt = a[i].mean(0) > 0.5, mk[i, 4] > 0.5
        assert (pred & gt).sum() / max(1, (pred | gt).sum()) > 0.95
    # the patch rule used in training (>= 50% of the patch) keeps the static patches static after a warp
    st_patch = F.avg_pool2d(mk[:, 4:5].float(), 14).squeeze(1) >= 0.5
    assert st_patch.sum() > 0


def test_v2_dataset_labels_align_with_labels_parquet():
    import numpy as np
    from second_look.dataset import PairDataset
    d1, d2 = PairDataset("val", labels="v1"), PairDataset("val", labels="v2")
    assert len(d1.meta) == len(d2.meta) and (d1.meta.pair_id.values == d2.meta.pair_id.values).all()
    ch = d1.meta.label.values != d2.meta.label.values
    assert 0 < ch.sum() < 0.02 * len(ch)                       # only a small share of val labels flip
    assert (d2.meta.label.values[ch] == 0).all()               # flips are changed -> unchanged / gray, never the other way
    assert len(d2.rows) <= len(d1.rows)                        # pairs that became gray leave the trainable set
    import pytest
    with pytest.raises(ValueError):
        PairDataset("val", labels="v3")
