"""Turn research checkpoints into the release files and write ``mb_segment/weights.json`` with their SHA-256.

    python tools/export_weights.py --out weights \
        --unet valdo_t2    42 /path/valdo_t2/seed42/modell.pt  --unet valdo_t2    1 /path/valdo_t2/seed1/modell.pt \
        --unet charite_t2  42 ...                              --unet charite_t2  1 ... \
        --unet charite_swi 42 ...                              --unet charite_swi 1 ... \
        --classifier /path/final_classifier/classifier.pt [--release-url URL]

Each U-Net file holds ``{"state_dict": <keys of mb_segment.unet>, "meta": <meta.json of the training>}``; the classifier
bundle is copied as it is (its keys already match ``mb_segment.classifier.build_classifier``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mb_segment.unet import build_unet, convert_legacy_keys  # noqa: E402
from mb_segment.weights import REGISTRY_PATH, sha256          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unet", nargs=3, action="append", metavar=("NAME", "SEED", "MODELL_PT"), default=[])
    ap.add_argument("--classifier")
    ap.add_argument("--release-url")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = json.load(open(REGISTRY_PATH))
    if a.release_url:
        reg["release_url"] = a.release_url
    for name, seed, path in a.unet:
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = convert_legacy_keys(state) if any(k.startswith("netz.") for k in state) else state
        model = build_unet()
        model.load_state_dict(state, strict=True)                       # proves the keys and shapes match
        meta_path = os.path.join(os.path.dirname(path), "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        meta = {k: v for k, v in meta.items() if k not in ("cases",)}   # no case identifiers in the release
        target = os.path.join(a.out, f"{name}_seed{seed}.pt")
        torch.save(dict(state_dict=model.state_dict(), meta=dict(meta, name=name, seed=int(seed), source=os.path.basename(path))), target)
        entry = next(m for m in reg["models"][name]["members"] if m["file"] == os.path.basename(target))
        entry["sha256"] = sha256(target)
        print(f"{target}: {os.path.getsize(target) / 1e6:.1f} MB  sha256 {entry['sha256'][:12]}...")
    if a.classifier:
        bundle = torch.load(a.classifier, map_location="cpu", weights_only=False)
        target = os.path.join(a.out, "classifier.pt")
        torch.save(bundle, target)
        reg["classifier"]["sha256"] = sha256(target)
        reg["classifier"]["threshold"] = bundle["threshold"]
        print(f"{target}: {os.path.getsize(target) / 1e6:.1f} MB, threshold {bundle['threshold']:.2f}, {len(bundle['members'])} members")
    json.dump(reg, open(REGISTRY_PATH, "w"), indent=2)
    print(f"registry written: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
