"""Stage 2: the shared 3D-CNN candidate classifier (trained on the pooled candidates of VALDO and both Charité cohorts).

Input per candidate: a 20 x 40 x 40 voxel block (z, y, x; 20 mm cube) of image, FRST and stage-1 probability around
the component centre, centre-cropped to 16 x 32 x 32 (16 mm). Three members with different seeds are averaged.
Architecture, crop and channel order must match ``cmb/stage2/classifier.py`` of cmb-valdo-chain, where the weights
were trained (``cmb.stage2.final_classifier``).
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

HALF = (20, 20, 10)          # stored block, half sizes in (x, y, z) grid voxels -> 40 x 40 x 20
CROP = (16, 32, 32)          # (z, y, x) fed to the network


def build_classifier() -> nn.Sequential:
    def block(c_in, c_out):
        return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True),
                             nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True))
    return nn.Sequential(block(3, 16), nn.MaxPool3d(2), block(16, 32), nn.MaxPool3d(2), block(32, 64),
                         nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))


class CandidateClassifier:
    def __init__(self, members: List[nn.Module], threshold: float, meta: dict):
        self.members, self.threshold, self.meta = members, float(threshold), meta

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CandidateClassifier":
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        members = []
        for state in bundle["members"]:
            net = build_classifier()
            net.load_state_dict(state, strict=True)
            members.append(net.to(device).eval())
        meta = {k: v for k, v in bundle.items() if k != "members"}
        if tuple(meta.get("crop", CROP)) != CROP:
            raise ValueError(f"classifier crop {meta.get('crop')} differs from {CROP}")
        return cls(members, bundle["threshold"], meta)

    def score(self, cubes: np.ndarray, device: str = "cpu") -> np.ndarray:
        """``cubes``: (n, 3, 20, 40, 40) -> mean member probability per candidate."""
        if len(cubes) == 0:
            return np.zeros(0, np.float32)
        z0, y0, x0 = ((s - c) // 2 for s, c in zip(cubes.shape[-3:], CROP))
        x = torch.from_numpy(np.ascontiguousarray(cubes[..., z0:z0 + CROP[0], y0:y0 + CROP[1], x0:x0 + CROP[2]],
                                                  dtype=np.float32)).to(device)
        with torch.no_grad():
            scores = np.mean([torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy() for m in self.members], axis=0)
        return scores.astype(np.float32)


def crop_block(volume: np.ndarray, centre, half=HALF) -> np.ndarray:
    """Block of ``2 * half`` around ``centre`` (x, y, z), zero-padded at the volume border."""
    out = np.zeros([2 * h for h in half], volume.dtype)
    src, dst = [], []
    for c, h, n in zip(centre, half, volume.shape):
        lo, hi = c - h, c + h
        start, stop = max(lo, 0), min(hi, n)
        if start >= stop:
            return out
        src.append(slice(start, stop))
        dst.append(slice(start - lo, stop - lo))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def candidate_cubes(image: np.ndarray, frst: np.ndarray, prob: np.ndarray, rows) -> np.ndarray:
    """(n, 3, z, y, x) float32 blocks for the candidate rows (centres in grid voxels, x y z)."""
    cubes = []
    for r in rows:
        block = np.stack([crop_block(v, r["centre_voxel"]) for v in (image, frst, prob)])      # (3, x, y, z)
        cubes.append(np.transpose(block, (0, 3, 2, 1)).astype(np.float32))
    return np.stack(cubes) if cubes else np.zeros((0, 3, 2 * HALF[2], 2 * HALF[1], 2 * HALF[0]), np.float32)
