"""A synthetic phantom for self-tests: an ellipsoid 'brain' with a few small dark spheres (microbleed-like)."""
from __future__ import annotations

import nibabel as nib
import numpy as np


def phantom(shape=(96, 112, 40), spacing=(1.0, 1.0, 2.0), n_spots: int = 4, seed: int = 0, bias: bool = True):
    """Returns ``(image nifti, truth mask nifti, spot centres in voxels)``. Tissue 100 with noise, spots 15, background 0."""
    rng = np.random.RandomState(seed)
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    centre = np.array(shape) / 2.0
    radii = np.array(shape) / 2.0 - np.array([6, 6, 3])
    brain = (((x - centre[0]) / radii[0]) ** 2 + ((y - centre[1]) / radii[1]) ** 2 + ((z - centre[2]) / radii[2]) ** 2) <= 1.0
    image = np.where(brain, 100.0 + rng.normal(0, 4, shape), 0.0)
    if bias:                                                                # slow brightness drift across the head
        image *= np.where(brain, 1.0 + 0.25 * (x - centre[0]) / shape[0], 1.0)
    truth = np.zeros(shape, bool)
    centres = []
    tries = 0
    while len(centres) < n_spots and tries < 1000:
        tries += 1
        c = np.array([rng.randint(12, shape[0] - 12), rng.randint(12, shape[1] - 12), rng.randint(4, shape[2] - 4)])
        if not brain[tuple(c)] or any(np.abs(c - o).max() < 12 for o in centres):
            continue
        r_mm = 2.0
        sphere = (((x - c[0]) * spacing[0]) ** 2 + ((y - c[1]) * spacing[1]) ** 2 + ((z - c[2]) * spacing[2]) ** 2) <= r_mm ** 2
        image[sphere] = 15.0
        truth |= sphere
        centres.append(c)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    affine[:3, 3] = -np.array(shape) * np.array(spacing) / 2.0
    return (nib.Nifti1Image(image.astype(np.float32), affine), nib.Nifti1Image(truth.astype(np.uint8), affine),
            [c.tolist() for c in centres])
