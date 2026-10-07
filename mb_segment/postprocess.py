"""Back to the patient's space: grid -> uncropped grid -> canonical (RAS) grid -> the orientation of the input file."""
from __future__ import annotations

import csv
import json
from typing import List

import nibabel as nib
import numpy as np

from .preprocess import VOXEL_MM3, GridCase, from_grid


def grid_to_native(case: GridCase, volume_grid: np.ndarray, order: int = 0) -> np.ndarray:
    """Cropped grid volume -> array in the orientation of the ORIGINAL file (nearest neighbour by default)."""
    full = np.zeros(tuple(int(v) for v in case.full_shape), volume_grid.dtype)
    sl = tuple(slice(int(a), int(a) + n) for a, n in zip(case.lo, volume_grid.shape))
    full[sl] = volume_grid
    canonical = from_grid(full, case.r, case.off, case.canonical.shape[:3], order=order)
    ornt_can = nib.io_orientation(case.canonical.affine)
    ornt_orig = nib.io_orientation(case.original.affine)
    transform = nib.orientations.ornt_transform(ornt_can, ornt_orig)
    return nib.orientations.apply_orientation(canonical, transform)


def save_native(case: GridCase, array, path: str, dtype) -> None:
    arr = np.asarray(array, dtype=dtype)
    if case.original.ndim == 4:
        arr = arr[..., None]
    header = case.original.header.copy()
    header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(arr, case.original.affine, header), path)


def save_grid(case: GridCase, array, path: str, dtype) -> None:
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=dtype), case.affine), path)


def lesion_table(case: GridCase, rows: List[dict]) -> List[dict]:
    """One record per lesion: voxels, volume in mm3, centre in grid voxels and in world mm, scores, decision."""
    out = []
    for r in rows:
        c = r["centre_voxel"]
        world = (case.affine @ np.array([c[0], c[1], c[2], 1.0]))[:3]
        rec = dict(id=int(r["id"]), accepted=bool(r.get("accepted", True)), voxels=int(r["voxels"]),
                   volume_mm3=round(r["voxels"] * VOXEL_MM3, 2), max_probability=r["max_prob"],
                   classifier_score=(round(float(r["score"]), 3) if "score" in r else None),
                   tissue_label=r.get("tissue", -1), in_csf=bool(r.get("in_csf", False)),
                   centre_grid_voxel=[int(v) for v in c], centre_world_mm=[round(float(v), 2) for v in world],
                   rejected_by=r.get("rejected_by"))
        out.append(rec)
    return out


def write_tables(records: List[dict], stem: str) -> None:
    json.dump(records, open(stem + "_lesions.json", "w"), indent=1)
    keys = ["id", "accepted", "voxels", "volume_mm3", "max_probability", "classifier_score", "tissue_label", "in_csf",
            "centre_grid_voxel", "centre_world_mm", "rejected_by"]
    with open(stem + "_lesions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for rec in records:
            w.writerow([";".join(str(v) for v in rec[k]) if isinstance(rec[k], list) else rec[k] for k in keys])
