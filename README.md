# microbleed_detection_3D_UNET

Cerebral microbleed (CMB) detection on a single T2*-GRE or SWI volume with pre-trained anisotropic 3D U-Nets.
One command, CPU is enough, three models to choose from. Everything else (how the chain was built, every measurement,
every negative result) lives in the research repository [cmb-valdo-chain](https://github.com/mendeltem/cmb-valdo-chain)
and on the page [how the pipeline was built](https://mendeltem.github.io/valdo-cmb-qc/architecture.html).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU build of PyTorch (or any GPU build you have)
pip install git+https://github.com/mendeltem/microbleed_detection_3D_UNET.git
mb_segment -m charite_swi -i sub-01_swi.nii.gz --mask sub-01_brain_mask.nii.gz -o out/
```

The weights are downloaded on first use (about 90 MB per member, two members per model) and verified by SHA-256.

## Models

| `-m` | sequence | trained on | use it for |
|---|---|---|---|
| `valdo_t2` | T2*-GRE | all 72 cases of the public VALDO 2021 Task 2 (three cohorts, 0.8-4 mm slices, 236 microbleeds) | T2*-GRE from any scanner; the only model trained on public data alone |
| `charite_t2` | T2*-GRE, 5 mm slices | all 75 cases of the Charite in-house T2* cohort (828 microbleeds) | clinical T2* with thick slices |
| `charite_swi` | SWI | all 61 cases of the Charite in-house SWI cohort (652 microbleeds) | SWI |

Each model is the mean of two networks trained with different random seeds (the two-seed ensemble measured in the research
repository). After the network, a second stage looks at every candidate again: a small 3D CNN trained on the candidates of
all three cohorts together decides yes / no (`classifier.pt`, one for all models). `--stage1-only` switches it off.

**How well they work.** The numbers below are lesion-level F1 (a reference lesion counts as found when a predicted component
touches it; every other component is a false alarm), pooled over cases, from the research repository. Cross-validation
numbers come from the five-fold models that preceded the released full-data models; the cross tests were run with the
released models themselves.

| Model | own data, 5-fold cross-validation (research fold models) | cross test with the released model, CLI default (stage 2) |
|---|---|---|
| `valdo_t2` | 0.647 (57 cases); 15 held-out VALDO cases measured once: 0.468 pooled, median per case 0.667; external public SWI cohort (Momeni), no adaptation: 0.613-0.630 | Charite T2* (75 cases): **0.604** (P 0.50, S 0.77); Charite SWI (other sequence): 0.434 |
| `charite_t2` | 0.671 (75 cases, with the classifier) | VALDO (72 cases): 0.506 (P 0.77, S 0.38) |
| `charite_swi` | 0.672 (61 cases, with the classifier) | VALDO (72 cases, other sequence): 0.354 |

The released models were trained on all cases, so they cannot be scored on their own data without optimism. Use the model that
matches your sequence; a T2* model on SWI (and the reverse) was measured and is clearly worse. One caveat measured in the cross tests:
the released full-data models give higher probabilities than the research fold models, so `--stage1-only` at the default threshold 0.3
over-detects on another scanner (0.415 against 0.629 cross-selected); the default stage 2 absorbs most of this, and `--threshold 0.6` is
the measured setting for stage 1 there. All numbers and the reasoning: `docs/CROSS-TESTS.md`.

## What the command does

1. reorients the image to RAS;
2. brain mask: the one you pass with `--mask` (recommended), or `--hd-bet` if [HD-BET](https://github.com/MIC-DKFZ/HD-BET) is
   installed, or `image > 0` with holes filled (right for skull-stripped input such as VALDO);
3. bias-field correction inside the mask: FSL `fast -B` if FSL is installed (what the models were trained with), else N4
   (`pip install mb-segment[n4]`), else none (`--bias` chooses explicitly; without the correction VALDO lost 0.022 F1, within noise);
4. inverted intensity scale `1 - I / max(I)`, resampling to 0.5 x 0.5 x 1 mm with the centre of the field of view preserved, crop to the brain;
5. fast radial symmetry transform as the second input channel;
6. sliding-window inference with the two-member ensemble;
7. stage 2 (default): candidates are the components of p > 0.15; the classifier accepts or rejects each one.
   `--stage1-only`: components of p > 0.3 between 2 mm3 and 2100 voxels; with `--tissue-map seg.nii.gz` (a SynthSeg
   segmentation of the same scan) components in ventricular or sulcal CSF are dropped (the CSF rule; do not use it on SWI,
   where it deletes true lesions);
8. the mask is mapped back to the input grid and orientation.

Outputs in `-o`: `<stem>_cmb_mask.nii.gz`, `<stem>_lesions.csv` / `.json` (volume in mm3, centre in mm, probability, classifier
score, decision, tissue label if a map was given), `<stem>_report.json` (model, members, settings, preprocessing, timings,
versions), optionally `<stem>_cmb_prob_grid.nii.gz` (`--save-prob`).

Runtime, measured on a VALDO case of 512 x 512 x 35 voxels (0.45 x 0.45 x 4 mm) with 8 CPU threads: 345 s in total with the
two-member ensemble, of which 100 s FSL `fast -B` and about 240 s the sliding window (373 windows x 2 members). `--bias none` or `n4`
is faster; `--threads` sets the torch threads; `--device cuda` uses a GPU when one is present (a few seconds per volume).

## Options

```
mb_segment -m MODEL -i IMAGE [-o OUT] [--mask MASK | --hd-bet] [--bias auto|fsl|n4|none]
           [--tissue-map SEG] [--stage1-only] [--no-csf-rule] [--save-prob]
           [--device cpu|cuda] [--threads N] [--batch N] [--weights-dir DIR] [--no-download]
mb_segment --list-models
mb_segment --selftest          # synthetic phantom through the whole chain with random weights, no download
```

## Tests

```bash
pip install -e .[test]
pytest                          # unit tests and the phantom end-to-end run; < 2 min on a CPU
MB_SEGMENT_WEIGHTS=weights pytest                                     # also: the release weights load and run
MB_SEGMENT_LEGACY_ROOT=/path/to/valdo-t2s pytest                      # also: regression against the research pipeline
```

What is tested: the radial symmetry transform peaks at the centre of a disc and stays low on a line; the U-Net has 22.5 M
parameters, pads odd shapes and crops back; research checkpoints load through the key conversion and give identical outputs;
the resampling keeps the centre of the field of view and round-trips a block (Dice > 0.9); the inverted scale reserves 0 for
"outside"; the size limits of the components; the sliding window on volumes smaller and larger than a window; candidate cubes
at the volume border; the phantom through the chain, also in a flipped orientation (same mask after mapping back); the CLI.
With the research data present: our sliding window reproduces the probability maps the research code stored for a fold-0 case,
and our preprocessing reproduces its grid image and FRST channel.

## Reproduce the weights

Research repository `cmb-valdo-chain`: `python -m cmb.training.full_data train --manifest <cases> --out <dir> --seed 42|1`
(all cases, 60 epochs, last state) for the U-Nets, `python -m cmb.stage2.final_classifier` for the classifier,
`tools/export_weights.py` here to convert and register them. The VALDO data are public (CC BY-NC-SA 4.0,
https://zenodo.org/records/4520773); the Charite cohorts never leave the clinic.

## Licence and citation

Code MIT. Weights: see `WEIGHTS-LICENSE.md` (VALDO-derived weights CC BY-NC-SA 4.0, Charite-derived weights research use only).
Not a medical device. If you use this, cite the VALDO challenge paper (Sudre et al., Med Image Anal 2024,
doi:10.1016/j.media.2023.103029) for the public data and the research repository for the method.
