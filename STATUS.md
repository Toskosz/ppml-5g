# PPML-5G experiment status

Updated: 2026-10-31

## Objective

Find a binary CIC-IDS-2018 model that classifies **anomaly vs benign** with:

- F1 score greater than 0.70.
- Real FHE end-to-end single-record latency no greater than 0.70 seconds.

Current scope decision: CIC-IDS-2018 `infilteration`/Infiltration traffic is
out of scope. It must be excluded from scoped evaluation, not relabeled as
benign. The primary held-out claim under evaluation is therefore Bot detection
against future benign traffic. Results must not be described as general
anomaly detection.

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

Committed supervised holdout results:

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

## Scope decision: exclude Infiltration

Decision made on 2026-10-30:

- Infiltration is not a supported attack type for the current project scope.
- `infilteration` rows (the source dataset spelling) must be removed from
  scoped quality metrics.
- They must not be counted as benign or used to make the supported model look
  better.
- Any resulting claim must say **Bot detection** or list the explicitly
  supported attack families, not “general anomaly detection.”

The pre-correction two-million-row run is commit `012aac3`:

`anomaly_experiment_runs/20261030_234551_431460/`

That report has no scoped Bot-vs-benign columns. Its known-family splitter
assigned blocks independently per label, and Infiltration remained inside the
known-family model. Leave its known-family F1 of 0.9058 out of current claims.

The corrected run is commit `7658050`:

`anomaly_experiment_runs/20261031_004057_494668/`

It uses the same 1,987,045-row sample and seed as `012aac3`. Two other
directories in that commit did not finish. `20261030_233818_174372` loaded all
16,233,002 rows and stopped. `20261031_003425_468997` stopped after cleaning
the two-million-row sample.

Corrected unknown-anomaly results at the 1% validation benign-FPR threshold,
with Infiltration removed from the score:

- PCA-32: Bot-vs-benign F1 0.9845, Bot recall 0.9845, future benign
  false-positive rate 0.45%. Infiltration recall on the same days is 3.85%.
  Unscoped test F1 remains 0.8534 because that score still includes
  Infiltration.
- Standard benign distance: Bot-vs-benign F1 0.9840, Bot recall 0.9845,
  future benign false-positive rate 0.48%. Infiltration recall is 3.49%.

PCA-8, PCA-16, robust distance, and both k-means models miss Bot at the 1%
budget. The 5% budget recovers Bot recall near 98.5% for PCA and standard
distance, with future benign false-positive rates of 4.0–4.8%.

Corrected known-family logistic regression, using one global 30-minute block
assignment. `cross_split_block_overlap_count` is 0. Infiltration's 14 blocks
are excluded:

- Float C=10 F1 0.9926. Float C=1 F1 0.9918.
- Concrete ML 5-bit clear C=1 F1 0.9761. C=10 F1 0.9747.

The printed per-label table is float C=10. Of 23,983 attack test rows, 18,864
are DDoS LOIC-HTTP (recall 99.78%). Bot support in that split is 2,001 rows,
all detected. FTP brute force (2,493) and SSH brute force (605) are fully
detected. Web brute force, XSS, and SQL injection have 14, 5, and 1 test rows.
The 5-bit per-label false-positive rate was absent from that report. The
latency benchmark records it for both compiled logistic regressions.

This sample repeats the earlier two-million-row draw. It confirms the corrected
metrics on that draw. An independent Bot capture is still required before
claiming botnet detection beyond this dataset.

Quality gates from the corrected run:

1. Scoped Bot-vs-benign F1 >= 0.70: pass.
2. Bot recall >= 0.70: pass.
3. Future benign false-positive rate <= 1% at the 1% threshold: pass for
   PCA-32 and standard benign distance.
4. Known-family logistic regression F1 >= 0.70 on disjoint blocks: pass,
   including 5-bit clear mode.
5. Infiltration reported separately and excluded from known-family fitting:
   pass.

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

Use `--no-compile-fhe` for validation-only runs. Its `--compile-fhe` path
uses the capture-day split, whose supervised models failed. It is not the
known-family latency run.

### `anomaly_model_experiments.py`

Location:

`cicids2018-kdd-models/anomaly_model_experiments.py`

This is the quality experiment. It is separate from the existing supervised
trainer and writes unique runs under `anomaly_experiment_runs/`. The corrected
two-million-row run has already passed the quality gates above.

Unknown-anomaly protocol:

- Fit preprocessing and models using benign training traffic only.
- Select thresholds using only benign traffic from later validation days.
- Test on complete future capture days containing Bot and Infiltration.
  Existing reports retain both labels for scientific traceability; scoped
  Bot metrics must exclude Infiltration explicitly.
- Evaluate thresholds targeting 1% and 5% benign false-positive rates.
- Models:
  - PCA reconstruction error with 8, 16, and 32 components.
  - Standardized benign distance.
  - Robust median/MAD benign distance.
  - One- and four-centroid benign distance.

Known-family protocol:

- Build 30-minute capture blocks.
- Exclude unsupported Infiltration traffic.
- Include only remaining attack labels with at least three distinct blocks.
- Give each complete block one global train/validation/test assignment, so
  benign and attack rows from the same period cannot cross splits.
- Test LR C=1 and C=10 in float and Concrete ML 5-bit clear mode.
- Report labels excluded because they do not have enough independent blocks.

This script deliberately does not compile FHE circuits. The latency benchmark
for every qualified model is `qualified_fhe_benchmark.py`.

### `qualified_fhe_benchmark.py`

Location:

`cicids2018-kdd-models/qualified_fhe_benchmark.py`

Benchmarks the four models that passed
`anomaly_experiment_runs/20261031_004057_494668`:

- `standard_benign_distance` at the 1% validation benign-FPR threshold.
- `pca_reconstruction_32` at that same threshold.
- `known_lr_c1` and `known_lr_c10` at 5 bits.

Distance and PCA are custom arithmetic circuits. The logistic regressions use
Concrete ML. The report is rewritten after each model. `--only` repeats a
subset. `--sample-rows 0` is refused. Each run writes a new directory under
`qualified_fhe_runs/`.

End-to-end latency is the mean of encrypt + inference + decrypt. One-time key
generation is reported separately and is outside that mean.

A distance or PCA circuit passes when all of these hold:

- Quantized-clear scoped Bot F1 and Bot recall are >= 0.70.
- Quantized-clear future benign false-positive rate is <= 1%.
- Simulation scoped Bot F1 and Bot recall are >= 0.70.
- Mean end-to-end latency on 100 records is <= 0.70 seconds.

A logistic regression passes when clear test F1 and simulation F1 are >= 0.70
and mean end-to-end latency is <= 0.70 seconds. On the corrected quality run
that test mass was mostly DDoS LOIC-HTTP, plus Bot, FTP brute force, and SSH
brute force.

The default 2,000,000-row sample should reproduce float Bot F1 near 0.9840
for distance and 0.9845 for PCA-32, and clear 5-bit test F1 near 0.9761 for
`known_lr_c1` and 0.9747 for `known_lr_c10`. Those matches are recorded. The
pass/fail decision uses the compiled circuit.

## Next action

Run this on the machine that has the CIC-IDS-2018 CSVs and Concrete ML. Do
not run it in the agent workspace.

```bash
cd cicids2018-kdd-models
python qualified_fhe_benchmark.py --device auto
```

Read `qualified_fhe_runs/<timestamp>/report.txt`. Describe a passing logistic
regression as known-family detection. Describe a passing distance or PCA
circuit as Bot detection against future benign traffic.

Full-data loading remains unsafe under the 20 GiB RAM cap.
`anomaly_model_experiments.py --sample-rows 0` materializes all 16.2 million
rows.

## Repository state at handoff

- Existing result directories and reports have been preserved.
- The corrected two-million-row quality run is complete and passes gates 1–5.
- `qualified_fhe_benchmark.py` times all four qualified models. It has not
  been run.
- Experiments are executed outside this workspace. The CIC-IDS-2018 CSV files
  are not stored in the repository.
