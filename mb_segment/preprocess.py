"""From the native image to the training grid: brain mask, bias correction, inverted scale, 0.5 x 0.5 x 1 mm, FRST.

The chain is the one the models were trained with (cmb-valdo-chain: ``code/preprocess.py``, ``code/gitter_c_bauen.py``,
``cmb/transfer/build_private_grids.py``), ported function by function:

  1. reorient to RAS (``nibabel.as_closest_canonical``);
  2. brain mask: the mask you pass (recommended; HD-BET or the scanner's), or HD-BET if installed and requested,
     or ``image > 0`` with holes filled (the convention of skull-stripped data such as VALDO);
  3. bias-field correction inside the mask: FSL ``fast -B`` (what the models saw), else N4 (SimpleITK), else none
     (measured on VALDO: training without the correction -0.022 [-0.097; +0.052], within noise);
  4. inverted scale ``1 - I / max(I)`` inside the mask, so microbleeds are the brightest objects; 0 = outside;
  5. resampling to 0.5 x 0.5 x 1.0 mm with the centre of the field of view preserved (``i = off + r * j``), linear,
     edge slices held (no zero slices), cropped to the brain bounding box plus 8 voxels;
  6. the fast radial symmetry map on that grid as the second channel.

Nothing is painted over (the vessel inpainting of MicrobleedNet removed 109 of 236 VALDO microbleeds).
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

from .frst import brain_mask_from_image, frst_volume

SPACING = np.array([0.5, 0.5, 1.0])     # mm, (x, y, z) of the RAS image
MARGIN = 8                              # voxels of the target grid around the brain bounding box
VOXEL_MM3 = float(np.prod(SPACING))     # 0.25 mm3


@dataclass
class GridCase:
    """Everything the inference and the back-projection need."""
    image: np.ndarray                   # (x, y, z) float32 on the grid, inverted, 0 outside the brain, cropped
    frst: np.ndarray                    # same grid, float32 in [0, 1]
    brain: np.ndarray                   # same grid, bool
    affine: np.ndarray                  # 4 x 4 of the cropped grid (world coordinates exact)
    full_shape: np.ndarray              # M: shape of the uncropped grid
    lo: np.ndarray                      # crop start on the uncropped grid
    r: np.ndarray                       # source voxels per target voxel (x, y, z)
    off: np.ndarray                     # source index of target index 0
    canonical: nib.Nifti1Image          # the RAS image the grid was built from
    original: nib.Nifti1Image           # the file as given (orientation of the outputs)
    native_mask: np.ndarray             # brain mask on the canonical grid
    info: dict = field(default_factory=dict)


def _squeeze(a: np.ndarray) -> np.ndarray:
    return a[..., 0] if a.ndim == 4 else a


def load_canonical(path: str):
    original = nib.load(path)
    return original, nib.as_closest_canonical(original)


def _run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (rc {proc.returncode}):\n{proc.stdout[-2000:]}")
    return proc.stdout


def _fsl_fast() -> Optional[str]:
    """FSL's ``fast``: on the PATH, or below $FSLDIR, or in the usual home install."""
    exe = shutil.which("fast")
    if exe:
        return exe
    for root in (os.environ.get("FSLDIR"), os.path.expanduser("~/fsl"), "/usr/local/fsl"):
        if root and os.path.exists(os.path.join(root, "bin", "fast")):
            return os.path.join(root, "bin", "fast")
    return None


def brain_mask(canonical: nib.Nifti1Image, image: np.ndarray, mask_path: Optional[str] = None,
               hd_bet: bool = False, workdir: Optional[str] = None) -> tuple[np.ndarray, str]:
    """Returns ``(mask on the canonical grid, how it was obtained)``."""
    if mask_path:
        m = _squeeze(np.asarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj)) > 0
        if m.shape != image.shape:
            raise ValueError(f"mask shape {m.shape} does not match the image {image.shape}")
        return m, f"given: {mask_path}"
    if hd_bet:
        exe = shutil.which("hd-bet")
        if not exe:
            raise RuntimeError("--hd-bet requested but the 'hd-bet' command is not installed (pip install hd-bet)")
        workdir = workdir or tempfile.mkdtemp(prefix="mb_segment_")
        src = os.path.join(workdir, "input.nii.gz")
        nib.save(nib.Nifti1Image(image.astype(np.float32), canonical.affine), src)
        out = os.path.join(workdir, "hdbet")
        try:                                                     # HD-BET 1.x
            _run([exe, "-i", src, "-o", out + ".nii.gz", "-device", "cpu", "-mode", "fast", "-tta", "0"])
        except RuntimeError:                                     # HD-BET 2.x
            _run([exe, "-i", src, "-o", out + ".nii.gz", "-device", "cpu", "--save_bet_mask", "--disable_tta"])
        masks = glob.glob(out + "*mask*.nii.gz") or glob.glob(out + "*bet*.nii.gz")
        if not masks:
            raise RuntimeError(f"hd-bet produced no mask in {workdir}")
        m = _squeeze(np.asarray(nib.as_closest_canonical(nib.load(masks[0])).dataobj)) > 0
        return m, f"HD-BET ({os.path.basename(masks[0])})"
    return brain_mask_from_image(image), "image > 0, holes filled (skull-stripped input assumed)"


def bias_correct(canonical: nib.Nifti1Image, image: np.ndarray, mask: np.ndarray, method: str = "auto",
                 workdir: Optional[str] = None) -> tuple[np.ndarray, str]:
    """``method``: auto | fsl | n4 | none. Returns ``(corrected image, method used)``."""
    if method == "auto":
        if _fsl_fast():
            method = "fsl"
        else:
            try:
                import SimpleITK  # noqa: F401
                method = "n4"
            except ImportError:
                method = "none"
    if method == "none":
        return image, "none"
    masked = np.where(mask, image, 0).astype(np.float32)
    if method == "fsl":
        exe = _fsl_fast()
        if not exe:
            raise RuntimeError("--bias fsl: FSL 'fast' is neither on the PATH nor in $FSLDIR/bin")
        workdir = workdir or tempfile.mkdtemp(prefix="mb_segment_")
        src = os.path.join(workdir, "brain.nii.gz")
        nib.save(nib.Nifti1Image(masked, canonical.affine), src)
        _run([exe, "-t", "2", "-n", "3", "-B", "--nopve", "-o", os.path.join(workdir, "fast"), src])
        restored = glob.glob(os.path.join(workdir, "fast_restore.nii*"))
        if not restored:
            raise RuntimeError(f"FSL fast wrote no *_restore image in {workdir}")
        out = _squeeze(np.asarray(nib.load(restored[0]).dataobj, dtype=np.float32))
        return out, "FSL fast -t 2 -n 3 -B --nopve"
    if method == "n4":
        import SimpleITK as sitk
        img = sitk.GetImageFromArray(np.ascontiguousarray(masked.transpose(2, 1, 0)))
        msk = sitk.GetImageFromArray(np.ascontiguousarray(mask.transpose(2, 1, 0)).astype(np.uint8))
        img.SetSpacing(tuple(float(v) for v in canonical.header.get_zooms()[:3]))
        msk.CopyInformation(img)
        shrink = 4
        small = sitk.Shrink(img, [shrink] * 3)
        small_mask = sitk.Shrink(msk, [shrink] * 3)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50] * 4)
        corrector.Execute(small, small_mask)
        log_field = corrector.GetLogBiasFieldAsImage(img)
        corrected = img / sitk.Exp(log_field)
        out = sitk.GetArrayFromImage(corrected).transpose(2, 1, 0).astype(np.float32)
        out[~mask] = 0.0
        return out, "N4 (SimpleITK, shrink 4)"
    raise ValueError(f"unknown bias method {method!r}")


def invert(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``1 - I / max(I)`` inside the brain, 0 outside; values <= 0 inside the brain become 1e-6 (0 is reserved)."""
    brain = mask & (image > 0)
    if not brain.any():
        raise ValueError("the brain mask is empty (or the image is zero inside it)")
    inverted = np.where(brain, 1.0 - image / image[brain].max(), 0.0).astype(np.float32)
    inverted[brain & (inverted <= 0)] = 1e-6
    return inverted, brain


def target_grid(affine: np.ndarray, shape, spacing: np.ndarray = SPACING):
    """Target size M, step r (source voxels per target voxel) and offset so that the centre of the field of view stays."""
    s = np.sqrt((affine[:3, :3] ** 2).sum(0))
    N = np.array(shape[:3], dtype=float)
    M = np.rint(N * s / spacing).astype(int)
    r = spacing / s
    off = ((N - 1) - r * (M - 1)) / 2.0
    return s, M, r, off


def to_grid(volume: np.ndarray, r: np.ndarray, off: np.ndarray, M: np.ndarray, order: int = 1) -> np.ndarray:
    out = ndimage.affine_transform(volume, r, offset=off, output_shape=tuple(int(v) for v in M), order=order,
                                   mode="nearest", output=np.float32)
    return out


def from_grid(volume_full: np.ndarray, r: np.ndarray, off: np.ndarray, N, order: int = 0) -> np.ndarray:
    """Inverse of ``to_grid``: target index j = (i - off) / r for every source voxel i (nearest by default)."""
    return ndimage.affine_transform(volume_full, 1.0 / r, offset=-off / r, output_shape=tuple(int(v) for v in N),
                                    order=order, mode="constant", cval=0.0, output=volume_full.dtype)


def preprocess(image_path: str, mask_path: Optional[str] = None, bias: str = "auto", hd_bet: bool = False,
               workdir: Optional[str] = None, log=print) -> GridCase:
    original, canonical = load_canonical(image_path)
    image = _squeeze(np.asarray(canonical.dataobj, dtype=np.float32))
    if image.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {image.shape}")
    mask, mask_how = brain_mask(canonical, image, mask_path, hd_bet, workdir)
    log(f"brain mask: {mask_how}; {int(mask.sum())} voxels")
    corrected, bias_how = bias_correct(canonical, image, mask, bias, workdir)
    log(f"bias field: {bias_how}")
    inverted, brain = invert(corrected, mask)
    s, M, r, off = target_grid(canonical.affine, image.shape)
    full = to_grid(inverted, r, off, M, order=1)
    nz = np.nonzero(full)
    if nz[0].size == 0:
        raise ValueError("nothing left after resampling; check the mask")
    lo = np.maximum(np.array([a.min() for a in nz]) - MARGIN, 0)
    hi = np.minimum(np.array([a.max() for a in nz]) + MARGIN + 1, M)
    grid_image = np.ascontiguousarray(full[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))])
    T = np.eye(4)
    T[:3, :3] = np.diag(r)
    T[:3, 3] = off + r * lo
    grid_affine = canonical.affine @ T
    grid_brain = brain_mask_from_image(grid_image)
    frst, frst_info = frst_volume(grid_image, grid_brain)
    log(f"grid: {list(image.shape)} at {np.round(s, 3).tolist()} mm -> {grid_image.shape} at {SPACING.tolist()} mm "
        f"(uncropped {M.tolist()}); FRST brain median {frst_info['brain_median']}")
    info = dict(image=os.path.abspath(image_path), mask=mask_how, bias=bias_how, source_shape=list(image.shape),
                source_spacing=[round(float(v), 5) for v in s], grid_shape=list(grid_image.shape),
                grid_full_shape=M.tolist(), crop_lo=lo.tolist(), crop_hi=hi.tolist(), frst=frst_info,
                brain_voxels_grid=int(grid_brain.sum()))
    return GridCase(image=grid_image, frst=frst, brain=grid_brain, affine=grid_affine, full_shape=M, lo=lo, r=r,
                    off=off, canonical=canonical, original=original, native_mask=mask, info=info)
