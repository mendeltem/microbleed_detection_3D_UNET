#!/bin/bash
# Publish the weight files as release assets and tag the version (run after tools/export_weights.py filled weights/ and weights.json).
#   bash scripts/release.sh v0.1.0
set -eu
TAG=${1:?tag, e.g. v0.1.0}
cd "$(dirname "$0")/.."
for f in weights/*.pt; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
python3 - <<'PY'
import json; r=json.load(open("mb_segment/weights.json"))
missing=[m["file"] for e in r["models"].values() for m in e["members"] if not m["sha256"]]+([] if r["classifier"]["sha256"] else ["classifier.pt"])
assert not missing, f"no SHA-256 in weights.json for {missing}: run tools/export_weights.py first"
print("registry complete:", sum(len(e["members"]) for e in r["models"].values()), "members + classifier")
PY
git add mb_segment/weights.json && git commit -q -m "weights.json: SHA-256 of the $TAG release files" || true
git tag -f "$TAG" && git push -q origin main --tags
gh release create "$TAG" weights/*.pt --title "$TAG" --notes "Weight files for mb_segment $TAG: three two-seed anisotropic 3D U-Net ensembles (valdo_t2, charite_t2, charite_swi) and the shared candidate classifier. SHA-256 of every file in mb_segment/weights.json; licences in WEIGHTS-LICENSE.md." 2>&1 | tail -2
