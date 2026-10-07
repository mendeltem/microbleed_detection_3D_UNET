"""Parse the scoring section of cmb-valdo-chain's ``scripts/final_models.sh`` log into the table of docs/CROSS-TESTS.md.

    python tools/cross_tests_table.py /path/to/log_final_models_*.txt > docs/CROSS-TESTS.md
"""
from __future__ import annotations

import re
import sys

LABEL = {
    "valdo_t2-on-charite_t2": ("valdo_t2", "Charite T2* (75 cases, 828 CMB)", "cross test, nothing of this cohort in training"),
    "valdo_t2-on-charite_swi": ("valdo_t2", "Charite SWI (61 cases, 652 CMB)", "cross test, other sequence"),
    "charite_t2-on-valdo": ("charite_t2", "VALDO (72 cases, 236 CMB)", "cross test, nothing of VALDO in training"),
    "charite_swi-on-valdo": ("charite_swi", "VALDO (72 cases, 236 CMB)", "cross test, other sequence"),
    "valdo_t2-on-valdo": ("valdo_t2", "VALDO (its own training data)", "in-sample, optimistic by construction"),
    "charite_t2-on-charite_t2": ("charite_t2", "Charite T2* (its own training data)", "in-sample, optimistic by construction"),
    "charite_swi-on-charite_swi": ("charite_swi", "Charite SWI (its own training data)", "in-sample, optimistic by construction"),
}


def main() -> None:
    text = open(sys.argv[1]).read()
    blocks = re.split(r"^### \S+ \S+ score (\S+)$", text, flags=re.M)
    rows = {}
    for i in range(1, len(blocks), 2):
        name, body = blocks[i], blocks[i + 1]
        raw = re.search(r"F1 ([0-9.]+)\s+P ([0-9.]+)\s+S ([0-9.]+)\s+FP/case ([0-9.]+)", body)
        fixed = re.search(r"fixed cell \([^)]*\): ([0-9.]+)", body)
        cross = re.search(r"CROSS-SELECTED: F1 ([0-9.]+)\s+P ([0-9.]+)\s+S ([0-9.]+)\s+FP/case ([0-9.]+)", body)
        rows[name] = dict(raw=raw.groups() if raw else None, csf_fixed=fixed.group(1) if fixed else None, cross=cross.groups() if cross else None)
    print("# Cross tests of the released models\n")
    print("Run on 7 October 2026 with `scripts/final_models.sh` of cmb-valdo-chain. Every number: pooled lesion-level F1 (26-connected components, "
          "touch rule; a reference lesion counts as found when a predicted component touches it), released two-seed ensembles trained on ALL cases of their cohort, "
          "fixed operating point 0.3 / 2 mm3 / <= 2100 voxels. 'with CSF rule' adds the SynthSeg rule (components in free CSF dropped), which the research "
          "repository found harmful on SWI. 'cross-selected' chooses threshold and size per fold on the other folds (the research repository's honest operating point).\n")
    print("| Model | Test data | F1, fixed cell | P / S / FP per case | F1 with CSF rule, fixed cell | F1 with CSF rule, cross-selected | Reading |")
    print("|---|---|---|---|---|---|---|")
    for name, (model, data, note) in LABEL.items():
        r = rows.get(name)
        if not r or not r["raw"]:
            print(f"| `{model}` | {data} | (not scored) | | | | {note} |")
            continue
        f1, p, s, fp = r["raw"]
        cross = f"{r['cross'][0]} (P {r['cross'][1]} S {r['cross'][2]})" if r["cross"] else "-"
        print(f"| `{model}` | {data} | **{f1}** | {p} / {s} / {fp} | {r['csf_fixed'] or '-'} | {cross} | {note} |")
    print("\nFor comparison, the five-fold models of the research repository scored out of fold: VALDO 0.647 (with CSF rule, two seeds), Charite T2* 0.663, "
          "Charite SWI 0.626 (threshold only; with the shared classifier 0.671 / 0.672); the VALDO fold models without adaptation: Charite T2* 0.622, Charite SWI 0.305.")


if __name__ == "__main__":
    main()
