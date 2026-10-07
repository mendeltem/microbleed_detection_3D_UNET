"""End-to-end tests: the phantom through the whole chain (random weights), the CLI, and -- when the release weights and the
research data are present on this machine -- the regression against the research pipeline's own probability maps."""
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

from mb_segment.classifier import CandidateClassifier, build_classifier
from mb_segment.cli import main, segment
from mb_segment.synthetic import phantom
from mb_segment.unet import build_unet

WEIGHTS = os.environ.get("MB_SEGMENT_WEIGHTS")
LEGACY = os.environ.get("MB_SEGMENT_LEGACY_ROOT")          # e.g. /home/uchralt/data/work/valdo-t2s


@pytest.fixture(scope="module")
def phantom_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("phantom")
    img, truth, centres = phantom()
    nib.save(img, d / "phantom.nii.gz")
    nib.save(truth, d / "truth.nii.gz")
    return str(d / "phantom.nii.gz"), str(d / "truth.nii.gz"), centres


def test_phantom_end_to_end_random_weights(phantom_file, tmp_path):
    path, truth, centres = phantom_file
    torch.manual_seed(0)
    rep = segment(path, "test", str(tmp_path), bias="none", models=[build_unet().eval()],
                  classifier=CandidateClassifier([build_classifier().eval()], 0.5, dict(crop=[16, 32, 32])), save_prob=True)
    out = nib.load(rep["mask_file"])
    src = nib.load(path)
    assert out.shape == src.shape and np.allclose(out.affine, src.affine)
    assert set(np.unique(np.asarray(out.dataobj))) <= {0, 1}
    records = json.load(open(rep["lesions_file"]))
    assert rep["n_microbleeds"] + rep["n_rejected"] == len(records)
    assert os.path.exists(rep["mask_file"].replace("_cmb_mask.nii.gz", "_cmb_prob_grid.nii.gz"))
    assert rep["preprocessing"]["grid_shape"][2] == pytest.approx(rep["preprocessing"]["source_shape"][2] * 2, abs=20)


def test_phantom_stage1_only_and_oriented_input(phantom_file, tmp_path):
    """A LAS-oriented copy of the phantom must give the same mask after reorientation back."""
    path, truth, centres = phantom_file
    src = nib.load(path)
    flipped = nib.Nifti1Image(np.asarray(src.dataobj)[::-1], src.affine @ np.diag([-1, 1, 1, 1]) @ np.array(
        [[1, 0, 0, -(src.shape[0] - 1)], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    nib.save(flipped, tmp_path / "las.nii.gz")
    torch.manual_seed(0)
    model = build_unet().eval()
    a = segment(path, "test", str(tmp_path / "a"), bias="none", models=[model], stage1_only=True)
    b = segment(str(tmp_path / "las.nii.gz"), "test", str(tmp_path / "b"), bias="none", models=[model], stage1_only=True)
    ma = np.asarray(nib.load(a["mask_file"]).dataobj)
    mb = np.asarray(nib.load(b["mask_file"]).dataobj)[::-1]
    assert ma.shape == mb.shape
    assert (ma != mb).sum() <= 0.02 * max(ma.sum(), 1) + 2


def test_cli_list_models_and_selftest(capsys):
    assert main(["--list-models"]) == 0
    out = capsys.readouterr().out
    assert "valdo_t2" in out and "charite_swi" in out and "classifier" in out
    assert main(["--selftest"]) == 0
    assert "SELFTEST PASSED" in capsys.readouterr().out


def test_cli_requires_model_and_input():
    with pytest.raises(SystemExit):
        main([])


@pytest.mark.skipif(not WEIGHTS or not os.path.exists(os.path.join(WEIGHTS or "", "valdo_t2_seed42.pt")), reason="release weights not present")
def test_release_weights_load_and_run_on_phantom(phantom_file, tmp_path):
    path, truth, centres = phantom_file
    rep = segment(path, "valdo_t2", str(tmp_path), bias="none", weights_dir=WEIGHTS, download=False)
    assert len(rep["members"]) == 2 and os.path.exists(rep["mask_file"])


@pytest.mark.skipif(not (LEGACY and os.path.isdir(os.path.join(LEGACY or "", "ergebnisse/turnier/a03-aniso-e60-gd/fold0"))),
                    reason="research results not present")
def test_regression_against_research_fold_prediction():
    """Feed the research grid (image + FRST of one fold-0 test case) to the fold-0 research checkpoint through OUR sliding
    window and compare with the probability map the research code stored for that case."""
    import glob
    from mb_segment.infer import sliding_window
    from mb_segment.unet import load_unet
    fold = os.path.join(LEGACY, "ergebnisse/turnier/a03-aniso-e60-gd/fold0")
    maps = sorted(glob.glob(os.path.join(fold, "*_pred_proba.nii.gz")))
    assert maps
    case = os.path.basename(maps[0])[:-len("_pred_proba.nii.gz")]
    grid = os.path.join(LEGACY, "dev/gitter_d", case)
    image = np.asarray(nib.load(f"{grid}/{case}_image.nii.gz").dataobj, dtype=np.float32)
    frst = np.asarray(nib.load(f"{grid}/{case}_frst.nii.gz").dataobj, dtype=np.float32)
    model = load_unet(os.path.join(fold, "modell.pt"))
    ours = sliding_window([model], image, frst, batch=8)
    theirs = np.asarray(nib.load(maps[0]).dataobj).astype(np.float32) / 255.0
    assert ours.shape == theirs.shape
    diff = np.abs(ours - theirs)
    # the research map was computed on the GPU in bfloat16 and stored as uint8: expect small differences, same lesions
    assert diff.mean() < 0.01 and np.mean(diff > 0.1) < 0.002
    assert ((ours > 0.3) == (theirs > 0.3)).mean() > 0.999


@pytest.mark.skipif(not (LEGACY and os.path.isdir(os.path.join(LEGACY or "", "dev/preproc"))), reason="research data not present")
def test_regression_preprocessing_matches_research_grid():
    """Our preprocessing from the research's bias-corrected, masked VALDO image must reproduce the stored grid image and FRST."""
    import glob
    from mb_segment.preprocess import preprocess
    cases = sorted(glob.glob(os.path.join(LEGACY, "dev/gitter_d/sub-*")))
    case = os.path.basename(cases[0])
    src = os.path.join(LEGACY, "dev/preproc", case, "fast_restore.nii.gz")     # FSL FAST output, masked, not yet inverted
    if not os.path.exists(src):
        pytest.skip("fast_restore.nii.gz missing")
    g = preprocess(src, bias="none", log=lambda m: None)
    # the research grid keeps the source orientation, we work in RAS: compare in RAS (world coordinates are exact in both)
    ref_img_nii = nib.as_closest_canonical(nib.load(f"{cases[0]}/{case}_image.nii.gz"))
    ref_img = np.asarray(ref_img_nii.dataobj, dtype=np.float32)
    ref_frst = np.asarray(nib.as_closest_canonical(nib.load(f"{cases[0]}/{case}_frst.nii.gz")).dataobj, dtype=np.float32)
    assert g.image.shape == ref_img.shape, (g.image.shape, ref_img.shape)
    assert np.allclose(ref_img_nii.affine, g.affine, atol=1e-3), (ref_img_nii.affine, g.affine)
    # the research mask came from the VALDO file, ours from fast_restore > 0: a handful of border voxels differ (measured on
    # sub-101: 42 of 15.4 M voxels), the FRST follows at the same spots. Tolerances are on the fraction, not the maximum.
    assert (np.abs(g.image - ref_img) > 1e-3).mean() < 1e-4
    assert (np.abs(g.frst - ref_frst) > 1e-2).mean() < 1e-3
