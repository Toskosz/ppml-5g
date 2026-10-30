# PPML-5G experiment status

Updated: 2026-10-30

## Objective

Find a binary CIC-IDS-2018 model that classifies **anomaly vs benign** with:

- F1 score greater than 0.70.
- Real FHE end-to-end single-record latency no greater than 0.70 seconds.

Do not use row-random validation as final evidence. It mixes flows from the
same capture days and attack sessions across train and test.

## Random-forest result

The completed CUDA suite is in:

`cicids2018-kdd-models/run_logs/20261028_110413/`

Key results:

- Best useful forest: 4 trees, depth 4, SVD 15.
  - FHE F1: 0.7206.
  - Real FHE end-to-end latency: 2.159 s in the training harness and
    1.762 s after reloading.
- Fastest notable model: 2 trees, depth 2, SVD 100.
  - FHE inference: 0.584 s after reloading.
  - End-to-end: 0.600 s.
  - FHE F1: only 0.2083.
- No forest configuration satisfies both targets.
- Each CUDA circuit still took approximately 8,133–8,226 seconds
  (about 2.26 hours) to compile.
- CUDA did not consistently improve runtime latency over CPU.

The SVD-100 model is fast because tree depth/node count dominates encrypted
tree inference. Most of its 100 inputs are not used by the shallow trees.

## First alternative-model screen: promising but not valid final evidence

Run:

`cicids2018-kdd-models/plaintext_experiment_runs/20261030_002931_248889/`

This used a row-random 70/15/15 split on about one million sampled flows.
Expanded compact-flow features produced:

- MLP-16 F1: 0.9389.
- MLP-8 F1: 0.9370.
- Logistic regression C=10 F1: 0.9307.
- Logistic regression C=1 F1: 0.9285.
- XGB, four depth-2 trees, F1: 0.9075.

These scores are not deployment evidence because each capture day contributed
flows to every split. They mainly demonstrate that the expanded flow features
are much more informative than the original seven-field/SVD representation.

## Capture-day validation: all supervised candidates failed

Latest committed results:

- Commit: `17025a2` (`more results`).
- Run:
  `cicids2018-kdd-models/plaintext_experiment_runs/20261030_004253_544870/`

Split:

- Train: six capture days through 2018-02-22.
- Validation: 2018-02-23 and 2018-02-28.
- Test: 2018-03-01 and 2018-03-02.
- No capture day appears in more than one split.

Test set:

- 60,777 benign flows.
- 17,516 `bot` flows.
- 5,655 `infilteration` flows (source dataset spelling).
- Attack prevalence: 27.6%.
- Always-predict-attack F1: 0.4326.

Float results:

- MLP-8: F1 0.4328.
- XGB, four depth-2 trees: F1 0.4326.
- LR C=1: F1 0.1021.
- LR C=10: F1 0.0985.

The MLP and XGB scores are effectively the always-attack baseline. The selected
quantized MLP detected nearly every attack but also flagged 98.84% of benign
flows. This is unusable.

Concrete ML quantized-clear tests at 5, 6, and 8 bits also failed. The best was
MLP-8 at F1 0.4347. Quantization is not the root cause because the float models
already fail.

No FHE circuit should be compiled from this run.

## Interpretation

The output remains binary, but the positive class contains different traffic
distributions: Bot, Infiltration, DoS, brute force, and others.

Capture-day validation asks whether the model works on future traffic without
seeing flows from the same capture session. In this dataset it also partly
becomes an unseen-attack-family test because some attacks occur on only one
day.

Two threat models must be evaluated separately:

1. **Known-family detection:** the attack family is represented during
   training, but test traffic comes from separate sessions/time blocks.
2. **Unknown-anomaly detection:** fitting does not use the held-out attack
   families; the model must identify deviations from benign traffic.

## Current scripts

### `plaintext_model_experiments.py`

Location:

`cicids2018-kdd-models/plaintext_model_experiments.py`

Current shortlist:

- LR C=1.
- LR C=10.
- MLP-8.
- XGB with four depth-2 trees.

It supports capture-day holdout, per-label detection, Concrete ML
quantized-clear tests, and gated LR compilation. Every invocation creates a
new microsecond-timestamped directory under `plaintext_experiment_runs/`; old
reports are not overwritten.

Use `--no-compile-fhe` for validation-only runs.

### `anomaly_model_experiments.py`

Location:

`cicids2018-kdd-models/anomaly_model_experiments.py`

This is the next experiment. It is separate from the existing supervised
trainer and writes unique runs under `anomaly_experiment_runs/`.

Unknown-anomaly protocol:

- Fit preprocessing and models using benign training traffic only.
- Select thresholds using only benign traffic from later validation days.
- Test on complete future capture days containing Bot and Infiltration.
- Evaluate thresholds targeting 1% and 5% benign false-positive rates.
- Models:
  - PCA reconstruction error with 8, 16, and 32 components.
  - Standardized benign distance.
  - Robust median/MAD benign distance.
  - One- and four-centroid benign distance.

Known-family protocol:

- Build 30-minute capture blocks.
- Include only original labels with at least three distinct blocks.
- Assign complete blocks chronologically to train/validation/test.
- Test LR C=1 and C=10 in float and Concrete ML 5-bit clear mode.
- Report labels excluded because they do not have enough independent blocks.

This script deliberately does not compile FHE circuits. A circuit is justified
only after a candidate passes quality and false-positive gates.

## Next action

From the model directory:

```bash
cd cicids2018-kdd-models
python anomaly_model_experiments.py
```

For a final full-data rerun after a promising sampled result:

```bash
python anomaly_model_experiments.py --sample-rows 0
```

Review the newest:

`anomaly_experiment_runs/<timestamp>/report.txt`

Decision gates:

1. Unknown-anomaly candidate must reach F1 >= 0.70.
2. Future benign false-positive rate must remain close to its validation
   budget (the report allows at most 1.5 times the requested budget).
3. Known-family LR must reach F1 >= 0.70 on separate time blocks.
4. Inspect per-label recall; aggregate F1 alone is insufficient.
5. Only after passing:
   - Compile a known-family LR through Concrete ML, or
   - Implement a custom arithmetic FHE circuit for the winning PCA/distance
     score.
6. Measure 100 real encrypt/inference/decrypt records and require mean
   end-to-end latency <= 0.70 s.

If both protocols fail, the next options are feature refinement, a small
benign-only quantized autoencoder, or an additional dataset that supports
independent sessions for the same attack families.

## Repository state at handoff

- Existing result directories and reports have been preserved.
- `anomaly_model_experiments.py` is newly added and has passed Python syntax
  and IDE lint checks.
- The external CIC-IDS-2018 CSV files are required to execute experiments and
  are not stored in the repository.
