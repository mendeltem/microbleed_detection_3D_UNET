"""Every command-line option, exercised through ``mb_segment.cli.main`` with the RELEASE weights (``MB_SEGMENT_WEIGHTS``).

Fast cases run on the synthetic phantom (a few seconds each); the ones that need a brain mask, a tissue map or FSL use the
research data when ``MB_SEGMENT_LEGACY_ROOT`` points at it. Each case checks the outputs and the report, not the accuracy.
"""
import json
import os
import shutil

import nibabel as nib
import numpy as np
import pytest
import torch

from mb_segment.cli import main
from mb_segment.synthetic import phantom

WEIGHTS = os.environ.get("MB_SEGMENT_WEIGHTS")
LEGACY = os.environ.get("MB_SEGMENT_LEGACY_ROOT")
HAVE_WEIGHTS = bool(WEIGHTS) and os.path.exists(os.path.join(WEIGHTS or "", "valdo_t2_seed1.pt"))
pytestmark = pytest.mark.skipif(not HAVE_WEIGHTS, reason="release weights not present (set MB_SEGMENT_WEIGHTS)")


@pytest.fixture(scope="module")
def phantom_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    img, truth, centres = phantom()
    nib.save(img, d / "phantom.nii.gz")
    mask = (np.asarray(img.dataobj) > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, img.affine), d / "mask.nii.gz")
    tissue = np.where(mask > 0, 2, 0).astype(np.int32)              # 2 = cerebral white matter
    tissue[:, :, : img.shape[2] // 3][mask[:, :, : img.shape[2] // 3] > 0] = 24   # lower third "CSF" (label 24)
    nib.save(nib.Nifti1Image(tissue, img.affine), d / "tissue.nii.gz")
    return str(d / "phantom.nii.gz"), str(d / "mask.nii.gz"), str(d / "tissue.nii.gz")


def run(args, out):
    rc = main(args + ["-o", str(out), "--weights-dir", WEIGHTS, "--no-download"])
    assert rc == 0
    rep = json.load(open(os.path.join(out, "phantom_report.json")))
    assert os.path.exists(rep["mask_file"]) and os.path.exists(rep["lesions_file"])
    return rep


@pytest.mark.parametrize("model", ["valdo_t2", "charite_t2", "charite_swi"])
def test_every_model_runs(phantom_files, tmp_path, model):
    img, _, _ = phantom_files
    rep = run(["-m", model, "-i", img, "--bias", "none"], tmp_path)
    assert rep["model"] == model and len(rep["members"]) == 2 and rep["decision"].startswith("stage 2")


def test_mask_option(phantom_files, tmp_path):
    img, mask, _ = phantom_files
    rep = run(["-m", "valdo_t2", "-i", img, "--mask", mask, "--bias", "none"], tmp_path)
    assert rep["preprocessing"]["mask"].startswith("given:")


def test_mask_shape_mismatch_is_an_error(phantom_files, tmp_path):
    img, _, _ = phantom_files
    bad = tmp_path / "bad_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((10, 10, 5), np.uint8), np.eye(4)), bad)
    with pytest.raises(ValueError):
        main(["-m", "valdo_t2", "-i", img, "--mask", str(bad), "--bias", "none", "-o", str(tmp_path), "--weights-dir", WEIGHTS, "--no-download"])


def test_bias_none_and_auto(phantom_files, tmp_path):
    img, _, _ = phantom_files
    a = run(["-m", "valdo_t2", "-i", img, "--bias", "none"], tmp_path / "none")
    assert a["preprocessing"]["bias"] == "none"
    b = run(["-m", "valdo_t2", "-i", img, "--bias", "auto"], tmp_path / "auto")
    assert b["preprocessing"]["bias"] in ("none",) or b["preprocessing"]["bias"].startswith(("FSL", "N4"))


def test_bias_n4(phantom_files, tmp_path):
    pytest.importorskip("SimpleITK")
    img, _, _ = phantom_files
    rep = run(["-m", "valdo_t2", "-i", img, "--bias", "n4"], tmp_path)
    assert rep["preprocessing"]["bias"].startswith("N4")


def test_bias_fsl(phantom_files, tmp_path):
    from mb_segment.preprocess import _fsl_fast
    if not _fsl_fast()[0]:
        pytest.skip("FSL not installed")
    img, _, _ = phantom_files
    rep = run(["-m", "valdo_t2", "-i", img, "--bias", "fsl"], tmp_path)
    assert rep["preprocessing"]["bias"].startswith("FSL")


def test_stage1_only_with_tissue_map_and_csf_rule(phantom_files, tmp_path):
    img, _, tissue = phantom_files
    with_rule = run(["-m", "valdo_t2", "-i", img, "--bias", "none", "--stage1-only", "--tissue-map", tissue], tmp_path / "rule")
    assert with_rule["decision"].startswith("stage 1") and with_rule["settings"]["csf_rule"] is True
    no_rule = run(["-m", "valdo_t2", "-i", img, "--bias", "none", "--stage1-only", "--tissue-map", tissue, "--no-csf-rule"], tmp_path / "norule")
    assert no_rule["settings"]["csf_rule"] is False
    assert no_rule["n_microbleeds"] >= with_rule["n_microbleeds"]           # the rule can only remove
    lesions = json.load(open(with_rule["lesions_file"]))
    assert all(l["tissue_label"] in (2, 24, 0, -1) for l in lesions)
    assert all((not l["accepted"]) for l in lesions if l["in_csf"])          # everything in CSF was rejected by the rule


def test_tissue_map_with_stage2_keeps_label_in_table(phantom_files, tmp_path):
    img, _, tissue = phantom_files
    rep = run(["-m", "valdo_t2", "-i", img, "--bias", "none", "--tissue-map", tissue], tmp_path)
    assert rep["decision"].startswith("stage 2")
    lesions = json.load(open(rep["lesions_file"]))
    assert all("tissue_label" in l and l["rejected_by"] in (None, "classifier") for l in lesions)


def test_save_prob_batch_threads(phantom_files, tmp_path):
    img, _, _ = phantom_files
    rep = run(["-m", "valdo_t2", "-i", img, "--bias", "none", "--save-prob", "--batch", "2", "--threads", "2"], tmp_path)
    prob = nib.load(rep["mask_file"].replace("_cmb_mask.nii.gz", "_cmb_prob_grid.nii.gz"))
    assert prob.shape == tuple(rep["preprocessing"]["grid_shape"]) and np.asarray(prob.dataobj).max() <= 255


def test_device_cuda_or_fallback(phantom_files, tmp_path):
    img, _, _ = phantom_files
    rep = run(["-m", "valdo_t2", "-i", img, "--bias", "none", "--device", "cuda"], tmp_path)
    assert rep["device"] == ("cuda" if torch.cuda.is_available() else "cpu")


def test_no_download_with_missing_weights_fails_clearly(phantom_files, tmp_path):
    img, _, _ = phantom_files
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        main(["-m", "valdo_t2", "-i", img, "--bias", "none", "-o", str(tmp_path), "--weights-dir", str(empty), "--no-download"])


def test_hd_bet(phantom_files, tmp_path):
    if not shutil.which("hd-bet"):
        pytest.skip("hd-bet not on the PATH")
    img, _, _ = phantom_files
    rep = run(["-m", "valdo_t2", "-i", img, "--bias", "none", "--hd-bet"], tmp_path)
    assert rep["preprocessing"]["mask"].startswith("HD-BET")


@pytest.mark.skipif(not (LEGACY and os.path.isdir(os.path.join(LEGACY or "", "dev/synthseg"))), reason="research data not present")
def test_real_image_with_mask_tissue_map_and_fsl(tmp_path):
    """A VALDO case with its own mask, the SynthSeg map of the research grid and FSL: the full option set on real data."""
    from mb_segment.preprocess import _fsl_fast
    case = "sub-107"
    image = f"/home/uchralt/data/extern/valdo2021/Task2/{case}/{case}_space-T2S_desc-masked_T2S.nii.gz"
    if not os.path.exists(image):
        pytest.skip("VALDO data not present")
    src = nib.load(image)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image((np.asarray(src.dataobj) > 0).astype(np.uint8), src.affine), mask_path)
    tissue = os.path.join(LEGACY, "dev/synthseg", f"{case}_synthseg.nii.gz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bias = "fsl" if _fsl_fast()[0] else "none"
    rc = main(["-m", "valdo_t2", "-i", image, "--mask", str(mask_path), "--tissue-map", tissue, "--bias", bias, "--stage1-only",
               "--device", device, "-o", str(tmp_path), "--weights-dir", WEIGHTS, "--no-download"])
    assert rc == 0
    rep = json.load(open(tmp_path / f"{case}_space-T2S_desc-masked_T2S_report.json"))
    assert rep["preprocessing"]["mask"].startswith("given:") and rep["settings"]["csf_rule"] is True
    out = nib.load(rep["mask_file"])
    assert out.shape == src.shape and np.allclose(out.affine, src.affine)
    reference = np.asarray(nib.load(image.replace("desc-masked_T2S", "CMB")).dataobj) > 0
    predicted = np.asarray(out.dataobj) > 0
    assert predicted.sum() > 0 and (predicted & reference).sum() > 0        # it finds something, and something real


def test_threshold_option_stage1(phantom_files, tmp_path):
    img, _, _ = phantom_files
    low = run(["-m", "valdo_t2", "-i", img, "--bias", "none", "--stage1-only", "--threshold", "0.2"], tmp_path / "low")
    high = run(["-m", "valdo_t2", "-i", img, "--bias", "none", "--stage1-only", "--threshold", "0.8"], tmp_path / "high")
    assert low["settings"]["stage1"]["threshold"] == 0.2 and high["settings"]["stage1"]["threshold"] == 0.8
    assert low["mask_voxels_native"] >= high["mask_voxels_native"]
