"""Fast radial symmetry transform (Loy & Zelinsky 2003), computed per axial slice.

Second input channel of the networks: every pixel votes along its gradient direction at a distance n,
votes accumulate at the centres of small round bright blobs (the image is inverted, so microbleeds are
bright) and the map stays flat on edges and long vessels. Float arithmetic, four radii in voxels on the
0.5 mm grid (= 1, 1.5, 2, 3 mm), one fixed scale per volume (99.9th percentile inside the brain).

Volumes are ``(x, y, z)`` arrays in nibabel order; slices are taken along axis 2.
This is a verbatim port of ``code/frst.py`` of the research repository (cmb-valdo-chain).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

RADII = (2, 3, 4, 6)
ALPHA = 2.0
PERCENTILE = 99.9


def _k_n(n: int) -> float:
    return 9.9 if n == 1 else 8.0


def _gradient_2d(s: np.ndarray):
    """Sobel 3x3 in the plane, divided by 8: derivative in intensity per voxel."""
    d = np.array([-1.0, 0.0, 1.0]) / 2.0
    w = np.array([1.0, 2.0, 1.0]) / 4.0
    gx = ndimage.correlate1d(ndimage.correlate1d(s, d, axis=0, mode="nearest"), w, axis=1, mode="nearest")
    gy = ndimage.correlate1d(ndimage.correlate1d(s, w, axis=0, mode="nearest"), d, axis=1, mode="nearest")
    return gx, gy


def frst_slice(s: np.ndarray, radii=RADII, alpha: float = ALPHA, beta: float = 0.0) -> np.ndarray:
    """FRST of one 2D slice for bright blobs, without normalisation (mean over the radii)."""
    s = np.asarray(s, dtype=np.float64)
    X, Y = s.shape
    gx, gy = _gradient_2d(s)
    mag = np.hypot(gx, gy)
    ok = mag > max(beta, 1e-12)
    px, py = np.nonzero(ok)
    g = mag[ok]
    ux, uy = gx[ok] / g, gy[ok] / g
    S = np.zeros((X, Y))
    for n in radii:
        kn = _k_n(n)
        dx = np.rint(n * ux).astype(np.int64)
        dy = np.rint(n * uy).astype(np.int64)
        qx, qy = px + dx, py + dy
        inside = (qx >= 0) & (qx < X) & (qy >= 0) & (qy < Y)
        flat = qx[inside] * Y + qy[inside]
        O = np.bincount(flat, minlength=X * Y).astype(np.float64)
        M = np.bincount(flat, weights=g[inside], minlength=X * Y)
        O = np.clip(O, -kn, kn).reshape(X, Y)
        M = M.reshape(X, Y)
        F = (np.abs(M) / kn) * (np.abs(O) / kn) ** alpha
        S += ndimage.gaussian_filter(F, 0.25 * n)
    return S / len(radii)


def brain_mask_from_image(img: np.ndarray) -> np.ndarray:
    """Brain = image > 0 with holes filled per slice (the convention of the inverted images)."""
    m = img > 0
    for z in range(m.shape[2]):
        m[:, :, z] = ndimage.binary_fill_holes(m[:, :, z])
    return m


def frst_volume(img: np.ndarray, brain: np.ndarray | None = None, radii=RADII, alpha: float = ALPHA,
                percentile: float = PERCENTILE):
    """FRST per axial slice (axis 2) of an ``(x, y, z)`` volume. Returns ``(frst float32 in [0, 1], info)``."""
    img = np.asarray(img, dtype=np.float32)
    if brain is None:
        brain = brain_mask_from_image(img)
    inner = np.zeros_like(brain)
    for z in range(brain.shape[2]):
        inner[:, :, z] = ndimage.binary_erosion(brain[:, :, z])
    ref = float(np.median(img[inner])) if inner.any() else 1.0        # tissue level of the case
    out = np.zeros(img.shape, dtype=np.float32)
    for z in range(img.shape[2]):
        if not inner[:, :, z].any():
            continue
        s = img[:, :, z].astype(np.float64) / ref
        med = np.median(s[inner[:, :, z]])
        s[~inner[:, :, z]] = med                                         # the brain border must not be an edge
        out[:, :, z] = frst_slice(s, radii, alpha)
    out[~brain] = 0.0
    p = float(np.percentile(out[brain], percentile)) if brain.any() else 0.0
    if p > 0:
        out = np.clip(out / p, 0.0, 1.0)
    return out.astype(np.float32), {"brain_median": round(ref, 4), "p999_raw": p, "radii": list(radii), "alpha": alpha}
