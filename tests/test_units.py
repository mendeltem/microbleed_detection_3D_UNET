"""Unit tests that need no weight files: FRST, U-Net, checkpoint key conversion, resampling, components, classifier plumbing."""
import numpy as np
import pytest
import torch

from mb_segment.classifier import CROP, HALF, CandidateClassifier, build_classifier, candidate_cubes, crop_block
from mb_segment.frst import frst_slice, frst_volume
from mb_segment.infer import PATCH, components, sliding_window
from mb_segment.preprocess import SPACING, from_grid, invert, target_grid, to_grid
from mb_segment.unet import PAD_MULTIPLE, build_unet, convert_legacy_keys, count_parameters, load_unet


def test_frst_peaks_at_disc_centre_and_ignores_lines():
    s = np.ones((120, 120))
    x, y = np.meshgrid(np.arange(120), np.arange(120), indexing="ij")
    s[(x - 60) ** 2 + (y - 60) ** 2 <= 4 ** 2] = 3.0            # bright disc, radius 4 voxels = 2 mm
    line = np.ones((120, 120))
    line[:, 58:60] = 3.0                                          # a bright line (vessel-like)
    disc = frst_slice(s)
    peak = np.unravel_index(disc.argmax(), disc.shape)
    assert max(abs(peak[0] - 60), abs(peak[1] - 60)) <= 1
    assert frst_slice(line).max() < 0.2 * disc.max()


def test_frst_volume_scaled_to_unit_and_zero_outside_brain():
    vol = np.zeros((60, 60, 8), np.float32)
    vol[10:50, 10:50, :] = 0.5
    vol[30, 30, 4] = 1.0
    out, info = frst_volume(vol)
    assert out.shape == vol.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out[0, 0, 0] == 0.0 and info["radii"] == [2, 3, 4, 6]


def test_unet_shape_parameters_and_padding():
    model = build_unet().eval()
    assert abs(count_parameters(model) / 1e6 - 22.5) < 0.3
    x = torch.zeros(1, 2, *PATCH)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, *PATCH)                              # 38 is not a multiple of 8: PadToMultiple crops back
    odd = torch.zeros(1, 2, 13, 21, 35)
    with torch.no_grad():
        assert model(odd).shape == (1, 1, 13, 21, 35)
    assert PAD_MULTIPLE == (8, 16, 16)


def test_legacy_keys_round_trip(tmp_path):
    model = build_unet()
    state = model.state_dict()
    legacy = {}
    for k, v in state.items():                                    # fabricate the research code's names
        lk = k.replace("net.", "netz.", 1)
        for new, old in (("bottleneck.", "mitte.net."), ("head.", "aus.")):
            lk = lk.replace(new, old)
        for i in range(4):
            lk = lk.replace(f"enc{i}.", f"e{i}.net.").replace(f"dec{i}.", f"d{i}.net.").replace(f"up{i}.", f"u{i}.")
        legacy[lk] = v
    assert any(k.startswith("netz.e0.net.0") for k in legacy)
    converted = convert_legacy_keys(legacy)
    assert set(converted) == set(state)
    torch.save(legacy, tmp_path / "modell.pt")
    loaded = load_unet(str(tmp_path / "modell.pt"))
    x = torch.randn(1, 2, 16, 32, 32)
    with torch.no_grad():
        assert torch.allclose(loaded(x), model.eval()(x))


def test_target_grid_keeps_the_centre_and_round_trips():
    affine = np.diag([0.45, 0.45, 4.0, 1.0])
    shape = (200, 220, 30)
    s, M, r, off = target_grid(affine, shape)
    assert np.allclose(s, [0.45, 0.45, 4.0]) and np.allclose(r, SPACING / s)
    centre_source = (np.array(shape) - 1) / 2.0
    centre_target_in_source = off + r * (M - 1) / 2.0
    assert np.allclose(centre_source, centre_target_in_source, atol=1e-9)
    vol = np.zeros(shape, np.float32)
    vol[90:110, 100:120, 12:18] = 1.0
    grid = to_grid(vol, r, off, M, order=1)
    back = from_grid((grid > 0.5).astype(np.uint8), r, off, shape, order=0)
    inter = np.logical_and(back > 0, vol > 0).sum()
    dice = 2 * inter / (back.sum() + vol.sum())
    assert dice > 0.9


def test_invert_reserves_zero_for_outside():
    img = np.array([[[0.0, 10.0, 20.0, 40.0]]], np.float32)
    mask = np.array([[[True, True, True, True]]])
    inv, brain = invert(img, mask)
    assert inv[0, 0, 0] == 0.0 and not brain[0, 0, 0]              # image 0 inside the mask = outside
    assert inv[0, 0, 3] == 1e-6                                   # the maximum maps to 0 -> reserved value
    assert np.isclose(inv[0, 0, 1], 0.75)


def test_components_size_limits_and_centres():
    prob = np.zeros((40, 40, 20), np.float32)
    prob[10:13, 10:13, 5:7] = 0.9                                 # 18 voxels: kept
    prob[30, 30, 10] = 0.9                                        # 1 voxel: below the 8-voxel minimum
    prob[0:20, 20:40, 0:20] = 0.5                                 # 8000 voxels: giant, dropped
    labels, rows, giant = components(prob, 0.3, 8, 2100)
    assert len(rows) == 1 and giant == 1 and rows[0]["voxels"] == 18
    assert rows[0]["centre_voxel"] == [11, 11, 5] or rows[0]["centre_voxel"] == [11, 11, 6]
    assert labels.max() == 1
    labels, rows, giant = components(np.zeros((5, 5, 5), np.float32), 0.3, 8, 2100)
    assert rows == [] and giant == 0


def test_sliding_window_covers_small_and_large_volumes():
    model = build_unet().eval()
    small = np.random.RandomState(0).rand(30, 40, 20).astype(np.float32)        # smaller than a window
    frst = np.zeros_like(small)
    p = sliding_window([model], small, frst)
    assert p.shape == small.shape and 0.0 <= p.min() and p.max() <= 1.0
    large = np.random.RandomState(1).rand(100, 70, 45).astype(np.float32)       # several windows, odd remainder
    p = sliding_window([model, model], large, np.zeros_like(large), batch=3)
    assert p.shape == large.shape


def test_candidate_cubes_and_classifier_scores():
    img = np.random.RandomState(0).rand(50, 60, 30).astype(np.float32)
    rows = [dict(id=1, centre_voxel=[25, 30, 15]), dict(id=2, centre_voxel=[1, 1, 1])]      # second one at the border
    cubes = candidate_cubes(img, img, img, rows)
    assert cubes.shape == (2, 3, 2 * HALF[2], 2 * HALF[1], 2 * HALF[0])
    assert crop_block(img, [1, 1, 1]).shape == (40, 40, 20)
    clf = CandidateClassifier([build_classifier().eval(), build_classifier().eval()], 0.5, dict(crop=list(CROP)))
    scores = clf.score(cubes)
    assert scores.shape == (2,) and np.all((scores >= 0) & (scores <= 1))
    assert clf.score(cubes[:0]).shape == (0,)
