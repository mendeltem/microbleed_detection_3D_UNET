"""Inference on the grid: sliding window over the U-Net ensemble, operating point, CSF rule, candidate classifier.

All numbers here are the frozen settings of cmb-valdo-chain, never re-tuned per image:
  * windows of the training size (38 x 62 x 94 voxels in z, y, x), half-window stride, overlapping predictions
    averaged, windows outside the brain skipped; the ensemble is the mean of the member probabilities;
  * stage 1 operating point: p > 0.3, 26-connected components, 2 mm3 (8 voxels) <= size <= 2100 voxels;
  * CSF rule (optional, needs a tissue map): drop a component whose majority SynthSeg label is ventricle or
    extracerebral CSF (labels 4, 5, 14, 15, 24, 43, 44);
  * stage 2: candidates are the components of p > 0.15 with >= 1 mm3; a small 3D CNN sees image, FRST and
    probability in a 16 mm cube around each and answers yes / no at the threshold stored with its weights.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch
from scipy import ndimage

PATCH = (38, 62, 94)                 # (z, y, x) voxels = 38 x 31 x 47 mm
N26 = np.ones((3, 3, 3), bool)
STAGE1 = dict(threshold=0.3, min_voxels=8, max_voxels=2100)
CANDIDATES = dict(threshold=0.15, min_voxels=4, max_voxels=2100)
CSF_LABELS = (4, 5, 14, 15, 24, 43, 44)


def _tile(a: np.ndarray, start) -> np.ndarray:
    z, y, x = start
    p = a[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]]
    if p.shape != PATCH:
        p = np.pad(p, [(0, PATCH[i] - p.shape[i]) for i in range(3)])
    return p


def sliding_window(models: Iterable[torch.nn.Module], image_xyz: np.ndarray, frst_xyz: np.ndarray,
                   device: str = "cpu", batch: int = 4, log=None) -> np.ndarray:
    """Mean probability map of the models, ``(x, y, z)`` float32 on the grid of the inputs."""
    models = list(models)
    img = np.ascontiguousarray(image_xyz.transpose(2, 1, 0), dtype=np.float32)      # (z, y, x) as in training
    frs = np.ascontiguousarray(frst_xyz.transpose(2, 1, 0), dtype=np.float32)
    shp = img.shape
    padded = tuple(max(shp[i], PATCH[i]) for i in range(3))
    starts = []
    for i in range(3):
        st = list(range(0, padded[i] - PATCH[i] + 1, PATCH[i] // 2))
        if st[-1] + PATCH[i] < padded[i]:
            st.append(padded[i] - PATCH[i])
        starts.append(st)
    tiles = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]
    tiles = [s for s in tiles if _tile(img, s).any()]
    pred = np.zeros(padded, np.float32)
    cnt = np.zeros(padded, np.float32)
    use_amp = device.startswith("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, len(tiles), batch):
            b = tiles[i:i + batch]
            xb = np.stack([np.stack([_tile(img, s), _tile(frs, s)], 0) for s in b])
            x = torch.from_numpy(xb).to(device)
            out = np.mean([torch.sigmoid(m(x)).float().cpu().numpy() for m in models], axis=0)
            for k, (z, y, xx) in enumerate(b):
                pred[z:z + PATCH[0], y:y + PATCH[1], xx:xx + PATCH[2]] += out[k, 0]
                cnt[z:z + PATCH[0], y:y + PATCH[1], xx:xx + PATCH[2]] += 1.0
            if log and (i // batch) % 10 == 0:
                log(f"  window {min(i + batch, len(tiles))}/{len(tiles)}")
    prob = (pred / np.maximum(cnt, 1))[:shp[0], :shp[1], :shp[2]]
    return np.ascontiguousarray(prob.transpose(2, 1, 0))


def components(prob_xyz: np.ndarray, threshold: float, min_voxels: int, max_voxels: int):
    """26-connected components of ``prob > threshold`` within the size limits.
    Returns ``(labels int32 with the kept components numbered 1..n, rows, n_giant)``; a row holds id, voxels,
    max probability and the centre (grid voxel, x y z); ``n_giant`` counts the components above ``max_voxels``."""
    mask = prob_xyz > threshold
    labels = np.zeros(prob_xyz.shape, np.int32)
    rows: List[dict] = []
    if not mask.any():
        return labels, rows, 0
    raw, n = ndimage.label(mask, structure=N26)
    sizes = np.bincount(raw.ravel())
    keep = (sizes >= min_voxels) & (sizes <= max_voxels)
    keep[0] = False
    next_id = 1
    for index, box in enumerate(ndimage.find_objects(raw), 1):
        if box is None or not keep[index]:
            continue
        local = raw[box] == index
        labels[box][local] = next_id
        centre = [int(round(c)) + s.start for c, s in zip(ndimage.center_of_mass(local), box)]
        rows.append(dict(id=next_id, voxels=int(sizes[index]), max_prob=round(float(prob_xyz[box][local].max()), 4),
                         centre_voxel=centre))
        next_id += 1
    n_giant = int((sizes[1:] > max_voxels).sum())
    return labels, rows, n_giant


def majority_tissue(labels: np.ndarray, rows: List[dict], tissue_grid: Optional[np.ndarray]) -> None:
    """Adds ``tissue`` (majority SynthSeg label, -1 without map) and ``in_csf`` to every row."""
    for r in rows:
        if tissue_grid is None:
            r["tissue"], r["in_csf"] = -1, False
            continue
        t = tissue_grid[labels == r["id"]]
        t = t[t > 0]
        majority = int(np.bincount(t).argmax()) if len(t) else 0
        r["tissue"], r["in_csf"] = majority, majority in CSF_LABELS
