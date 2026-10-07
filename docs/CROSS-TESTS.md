# Cross tests of the released models

Run on 7 October 2026 with `scripts/final_models.sh` and `scripts/final_cross_scores.sh` of cmb-valdo-chain, with the released
two-seed ensembles (trained on ALL cases of their cohort, last state of the 60-epoch schedule) and the released `classifier.pt`.
Every number is a pooled lesion-level F1 (26-connected components, touch rule: a reference lesion counts as found when a predicted
component touches it; every other component is a false alarm). "Stage 2" is what `mb_segment` outputs by default (candidates at
p > 0.15, shared classifier at its threshold 0.30). "Stage 1" is `--stage1-only` at the fixed threshold 0.3 / 2 mm3 / <= 2100 voxels;
"with CSF rule" adds `--tissue-map` (SynthSeg); "cross-selected" chooses threshold and size per fold on the other folds and is the
honest best of stage 1 (not available in the CLI, it needs reference labels).

## The cross tests (nothing of the test cohort in the model's training)

| Model | Test data | Stage 2 (CLI default) | Stage 1, threshold 0.3 | Stage 1 + CSF rule, 0.3 | Stage 1 + CSF rule, cross-selected |
|---|---|---|---|---|---|
| `valdo_t2` | Charite T2* (75 cases, 828 CMB) | **0.604** (P 0.50, S 0.77, 8.6 FP/case) | 0.415 (P 0.28, S 0.77, 21.5 FP/case) | 0.543 | 0.629 (threshold 0.6-0.7) |
| `valdo_t2` | Charite SWI (61 cases, 652 CMB), other sequence | 0.434 (P 0.43, S 0.43) | 0.246 | 0.338 | 0.340 |
| `charite_t2` | VALDO (72 cases, 236 CMB) | 0.506 (P 0.77, S 0.38) | 0.460 (P 0.57, S 0.39, 1.0 FP/case) | 0.488 | 0.467 |
| `charite_swi` | VALDO (72 cases, 236 CMB), other sequence | 0.354 | 0.159 | 0.341 | 0.341 |

## In-sample checks (the model's own training data, optimistic by construction)

| Model | Data | Stage 2 | Stage 1, 0.3 | Stage 1 + CSF rule, 0.3 | cross-selected |
|---|---|---|---|---|---|
| `valdo_t2` | VALDO | 0.663 | 0.584 (P 0.45, S 0.83) | 0.653 | 0.683 (threshold 0.4, 4 voxels) |
| `charite_t2` | Charite T2* | 0.706 | 0.654 | 0.677 | 0.675 |
| `charite_swi` | Charite SWI | 0.742 | 0.682 | 0.647 (the rule deletes true lesions on SWI) | 0.649 |

## What the numbers say

1. **Use the model of your sequence.** A T2* model on SWI or the reverse loses 0.2-0.3 F1.
2. **The released full-data models give higher probabilities than the research fold models.** The research repository's
   five-fold VALDO ensemble scores 0.622 on the Charite T2* cohort at the fixed threshold 0.3 (same evaluation); the released
   full-data ensemble scores 0.415 there, because its F1 optimum sits at threshold 0.6-0.7 (cross-selected 0.629, the same
   as the fold ensemble). In-sample the released models are not better than the fold models out of fold (T2* 0.654 vs 0.663),
   so this is a calibration shift of the last training state, not over-fitting. Consequences: keep the default (stage 2),
   which absorbs most of it (0.604); with `--stage1-only` on another scanner raise `--threshold` to about 0.6.
3. **The in-sample stage-2 numbers (0.66-0.74) are not a performance claim**; the honest numbers for a model on its own cohort
   remain the research repository's out-of-fold values: VALDO 0.647 (two seeds, CSF rule), Charite T2* 0.671 and SWI 0.672
   (with the shared classifier), measured with the fold models.
4. The CSF rule helps on T2* and VALDO and harms on SWI (deletes true lesions), as in the research repository; stage 2 needs no tissue map.
