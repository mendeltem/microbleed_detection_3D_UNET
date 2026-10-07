"""``mb_segment``: cerebral microbleeds from one T2*-GRE or SWI volume, on the CPU.

    mb_segment -m charite_swi -i sub-01_swi.nii.gz -o out/ --mask sub-01_brain_mask.nii.gz
    mb_segment -m valdo_t2   -i sub-02_T2star.nii.gz -o out/
    mb_segment --list-models
    mb_segment --selftest

Outputs (``<out>/<input stem>_...``): ``cmb_mask.nii.gz`` in the space and orientation of the input, ``lesions.csv`` and
``lesions.json`` (one row per lesion: volume, centre in mm, probability, classifier score, decision), ``report.json``
(model, settings, preprocessing, timings, versions) and, with ``--save-prob``, the probability map on the 0.5 x 0.5 x 1 mm grid.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np
import torch

from . import __version__
from .classifier import CandidateClassifier, candidate_cubes
from .infer import CANDIDATES, STAGE1, components, majority_tissue, sliding_window
from .postprocess import grid_to_native, lesion_table, save_grid, save_native, write_tables
from .preprocess import preprocess
from .unet import build_unet, load_unet
from .weights import model_names, registry, resolve


def _stem(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]


def tissue_on_grid(tissue_path: str, case):
    """Nearest-neighbour resampling of a SynthSeg map onto the cropped grid of the case."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    seg = nib.as_closest_canonical(nib.load(tissue_path))
    out = resample_from_to(seg, (case.image.shape, case.affine), order=0)
    return np.asarray(out.dataobj).astype(np.int32)


def segment(input_path: str, model: str, out_dir: str, mask: str | None = None, bias: str = "auto", hd_bet: bool = False,
            tissue_map: str | None = None, stage1_only: bool = False, csf_rule: bool = True, device: str = "cpu",
            batch: int = 4, threads: int | None = None, weights_dir: str | None = None, download: bool = True,
            save_prob: bool = False, models=None, classifier=None, log=None, threshold: float | None = None) -> dict:
    """The whole chain for one image. ``models`` / ``classifier`` may be passed for tests (no weight files needed)."""
    t0 = time.time()
    times = {}
    log = log or (lambda m: print(f"[mb_segment {time.time() - t0:6.1f} s] {m}", file=sys.stderr, flush=True))
    if threads:
        torch.set_num_threads(threads)
    if device.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA requested but not available, using the CPU")
        device = "cpu"
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, _stem(input_path))
    workdir = tempfile.mkdtemp(prefix="mb_segment_")

    if models is None:
        w = resolve(model, weights_dir, download=download, with_classifier=not stage1_only)
        models = [load_unet(p, device) for p in w["members"]]
        member_files = [os.path.basename(p) for p in w["members"]]
        if not stage1_only and w["classifier"]:
            classifier = CandidateClassifier.load(w["classifier"], device)
    else:
        member_files = ["in-memory"] * len(models)
    log(f"model {model}: {len(models)} members on {device}" + ("" if stage1_only or classifier is None else f", classifier threshold {classifier.threshold:.2f}"))

    case = preprocess(input_path, mask, bias, hd_bet, workdir, log)
    times["preprocess_s"] = round(time.time() - t0, 1)
    t1 = time.time()
    prob = sliding_window(models, case.image, case.frst, device, batch, log)
    times["inference_s"] = round(time.time() - t1, 1)
    log(f"probability map: max {prob.max():.2f}")

    tissue = tissue_on_grid(tissue_map, case) if tissue_map else None
    stage1 = dict(STAGE1, threshold=threshold if threshold is not None else STAGE1["threshold"])
    labels1, rows1, giant1 = components(prob, **stage1)
    majority_tissue(labels1, rows1, tissue)
    report = dict(version=__version__, model=model, members=member_files, device=device, input=os.path.abspath(input_path),
                  preprocessing=case.info, settings=dict(stage1=stage1, candidates=CANDIDATES, stage1_only=stage1_only,
                  csf_rule=bool(tissue is not None and csf_rule), tissue_map=tissue_map),
                  giant_components_dropped=giant1)
    if stage1_only or classifier is None:
        for r in rows1:
            r["accepted"] = not (csf_rule and tissue is not None and r["in_csf"])
            r["rejected_by"] = None if r["accepted"] else "csf rule"
        labels, rows = labels1, rows1
        report["decision"] = f"stage 1: threshold {stage1['threshold']}, 2 mm3 to 2100 voxels" + (", CSF rule" if tissue is not None and csf_rule else "")
    else:
        labels, rows, _ = components(prob, **CANDIDATES)
        majority_tissue(labels, rows, tissue)
        cubes = candidate_cubes(case.image, case.frst, prob, rows)
        scores = classifier.score(cubes, device)
        for r, s in zip(rows, scores):
            r["score"] = float(s)
            r["accepted"] = bool(s >= classifier.threshold)
            r["rejected_by"] = None if r["accepted"] else "classifier"
        report["decision"] = f"stage 2: {len(rows)} candidates at p > 0.15, classifier threshold {classifier.threshold:.2f}"
        report["classifier"] = {k: v for k, v in classifier.meta.items() if k not in ("members",)}
    accepted_ids = [r["id"] for r in rows if r["accepted"]]
    mask_grid = np.isin(labels, accepted_ids).astype(np.uint8)
    mask_native = grid_to_native(case, mask_grid, order=0)
    save_native(case, mask_native, stem + "_cmb_mask.nii.gz", np.uint8)
    if save_prob:
        save_grid(case, np.round(prob * 255).astype(np.uint8), stem + "_cmb_prob_grid.nii.gz", np.uint8)
    records = lesion_table(case, rows)
    write_tables(records, stem)
    n_acc = len(accepted_ids)
    report.update(n_microbleeds=n_acc, n_rejected=len(rows) - n_acc, lesions_file=stem + "_lesions.json",
                  mask_file=stem + "_cmb_mask.nii.gz", mask_voxels_native=int(mask_native.sum()),
                  timings=dict(**times, total_s=round(time.time() - t0, 1)),
                  environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, platform=platform.platform()))
    json.dump(report, open(stem + "_report.json", "w"), indent=1)
    log(f"{n_acc} microbleed(s) accepted, {len(rows) - n_acc} rejected -> {stem}_cmb_mask.nii.gz ({report['timings']['total_s']} s)")
    return report


def selftest(device: str = "cpu") -> bool:
    """End-to-end run on a synthetic phantom with randomly initialised networks: tests the plumbing, not the accuracy."""
    import nibabel as nib
    from .classifier import build_classifier
    from .synthetic import phantom
    tmp = tempfile.mkdtemp(prefix="mb_segment_selftest_")
    img, truth, centres = phantom()
    path = os.path.join(tmp, "phantom.nii.gz")
    nib.save(img, path)
    torch.manual_seed(0)
    models = [build_unet().eval()]
    clf = CandidateClassifier([build_classifier().eval()], 0.5, dict(crop=[16, 32, 32]))
    rep = segment(path, "selftest", tmp, bias="none", models=models, classifier=clf, device=device, save_prob=True)
    out = nib.load(rep["mask_file"])
    ok = out.shape == img.shape and np.allclose(out.affine, img.affine) and os.path.exists(rep["lesions_file"])
    print("SELFTEST " + ("PASSED" if ok else "FAILED") + f": phantom {img.shape}, {rep['n_microbleeds']} accepted / {rep['n_rejected']} rejected "
          f"(random weights, so the counts mean nothing), outputs in {tmp}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mb_segment", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", choices=model_names(), help="which pre-trained model to use")
    ap.add_argument("-i", "--input", help="T2*-GRE or SWI volume (NIfTI)")
    ap.add_argument("-o", "--out", default=".", help="output folder (default: current folder)")
    ap.add_argument("--mask", help="brain mask in the space of the input (recommended; otherwise image > 0 is taken as the brain)")
    ap.add_argument("--hd-bet", action="store_true", help="compute the brain mask with HD-BET (must be installed; CPU)")
    ap.add_argument("--bias", choices=["auto", "fsl", "n4", "none"], default="auto",
                    help="bias-field correction: FSL fast -B (what the models were trained with), N4 via SimpleITK, or none; auto = first available")
    ap.add_argument("--tissue-map", help="SynthSeg segmentation of the same scan; enables the CSF rule (stage 1) and the tissue label in the tables")
    ap.add_argument("--stage1-only", action="store_true", help="skip the candidate classifier: threshold 0.3 plus the CSF rule if a tissue map is given")
    ap.add_argument("--no-csf-rule", action="store_true", help="with --tissue-map and --stage1-only: keep components in CSF")
    ap.add_argument("--threshold", type=float, default=None,
                    help="stage-1 probability threshold (default 0.3). The released full-data models give higher probabilities than the "
                         "research fold models: with --stage1-only on another scanner 0.6 scored best (docs/CROSS-TESTS.md); stage 2 does not need this")
    ap.add_argument("--save-prob", action="store_true", help="also write the probability map on the 0.5 x 0.5 x 1 mm grid")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--threads", type=int, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--batch", type=int, default=4, help="windows per forward pass")
    ap.add_argument("--weights-dir", help="folder with the weight files (default: $MB_SEGMENT_WEIGHTS, ./weights, ~/.cache/mb_segment)")
    ap.add_argument("--no-download", action="store_true", help="never download weights; fail if they are missing")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run the chain on a synthetic phantom with random weights (no download)")
    ap.add_argument("--version", action="version", version=f"mb_segment {__version__}")
    a = ap.parse_args(argv)
    if a.list_models:
        reg = registry()
        for name, e in reg["models"].items():
            print(f"{name:12s} {e['sequence']:20s} {e['trained_on']}  [{', '.join(m['file'] for m in e['members'])}]")
        print(f"classifier   {reg['classifier']['file']}: {reg['classifier']['trained_on']}")
        return 0
    if a.selftest:
        return 0 if selftest(a.device) else 1
    if not a.model or not a.input:
        ap.error("-m/--model and -i/--input are required (or --list-models / --selftest)")
    segment(a.input, a.model, a.out, a.mask, a.bias, a.hd_bet, a.tissue_map, a.stage1_only, not a.no_csf_rule, a.device,
            a.batch, a.threads, a.weights_dir, not a.no_download, a.save_prob, threshold=a.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
