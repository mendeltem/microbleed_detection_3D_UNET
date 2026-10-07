# Licence of the weight files

The code in this repository is MIT-licensed (`LICENSE`). The weight files distributed with the releases are not code:

| File | Trained on | Licence |
|---|---|---|
| `valdo_t2_seed42.pt`, `valdo_t2_seed1.pt` | VALDO 2021 Task 2 (72 public cases; Sudre et al., Med Image Anal 2024) | CC BY-NC-SA 4.0, as the data (non-commercial, share alike, cite the VALDO paper) |
| `charite_t2_seed42.pt`, `charite_t2_seed1.pt`, `charite_swi_seed42.pt`, `charite_swi_seed1.pt` | in-house clinical cohorts of the Charite (T2*: 75 cases; SWI: 61 cases) | research use only, non-commercial; no patient data are contained in the weights, but the files are derived from clinical data and are released under the responsibility of the author |
| `classifier.pt` | out-of-fold candidates of all three cohorts | same terms as the Charite weights |

The models are research software and not a medical device. They were validated as reported in the README and nowhere else.
