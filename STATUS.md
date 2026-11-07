# PPML-5G experiment status

Updated: 2026-11-07

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

### `score_known_lr_unsampled_test.py`

Location:

`cicids2018-kdd-models/score_known_lr_unsampled_test.py`

Scores `known_lr_c1` and `known_lr_c10` on the flows the source-run coin flip
left out of the original 28 test blocks. The plaintext estimators were not
saved, so the script refits the float model and the 5-bit clear model on the
source training blocks, using the saved preprocessor and the frozen 5-bit
thresholds. FHE scores are simulation of the saved circuits. There is no new
parameter search and no real encrypt / inference / decrypt. Training blocks
and the sampled test rows are not scored.

The completed run is the unsampled test-block rescore below. Do not repeat it.

## Qualified FHE result

Completed CUDA run:

`cicids2018-kdd-models/qualified_fhe_runs/20261031_170906_440932/`

Same 1,987,045-row sample as `20261031_004057_494668`. Float and 5-bit clear
scores match that report to about 1e-5. Wall clock 17:09–17:57 (−03).

Passing models: `known_lr_c1` and `known_lr_c10`. Describe them as
known-family detection on disjoint 30-minute blocks.

- `known_lr_c1`: clear F1 0.9761, simulation F1 0.9746, benign false-positive
  rate 0.80%, mean end-to-end 0.0156 s. Frozen threshold 0.667279.
- `known_lr_c10`: clear F1 0.9747, simulation F1 0.9737, benign false-positive
  rate 0.88%, mean end-to-end 0.0157 s. Frozen threshold 0.821242.

`known_lr_c1` is the stronger model. Of 23,983 attack test rows, 18,864
(78.7%) are DDoS LOIC-HTTP. The sampler kept each of the 16,233,002 source
rows with probability 0.123206, so the share follows the dataset's flow
counts. The
published dataset has 576,191 LOIC-HTTP flows from about 65 minutes on
2018-02-20 (10:12–11:17). The split puts one block of each included family in
test, preferring the latest block. That LOIC-HTTP block is about 153,000 raw
flows. HOIC, 686,012 flows and the largest attack class, stays out because it
has only two blocks.

Clear recall for `known_lr_c1`: Bot 2,001/2,001, FTP brute force 2,493/2,493,
SSH brute force 605/605, LOIC-HTTP 99.63%, web brute force 8/14, XSS 2/5,
SQL injection 0/1. The Bot rows are a later block of the Friday 2018-03-02
capture. An independent Bot recording is still required before calling this
botnet detection beyond CIC-IDS-2018.

The Bot circuits fail. Both still match the float quality run.

- `standard_benign_distance`: quantized threshold 0, so every non-negative
  score is an attack. Quantized and simulation Bot F1 are 0.3659. Benign
  false-positive rate is 100%. Mean end-to-end 0.844 s.
- `pca_reconstruction_32`: quantized-clear Bot F1 0.9840, recall 0.9845,
  benign false-positive rate 0.48%. Simulation Bot F1 0.7603, recall 0.9818,
  benign false-positive rate 17.3%. Mean end-to-end 9.834 s. Latency is the
  only formal gate failure. The anomaly gate does not score simulation benign
  false-positive rate.

There is no FHE Bot-detection result against future benign traffic. Do not
rerun the benchmark, and do not rescore distance or PCA on more rows.

## Unsampled test-block rescore

Completed run:

`cicids2018-kdd-models/qualified_fhe_runs/20261103_232756_062792_unsampled_test/`

Wall clock 2026-11-03 23:27 to 2026-11-04 01:36 (−03). Same coin flip as
`20261031_170906_440932`: probability 0.123206, 1,998,881 sampled source
rows, cleaned sample 1,987,045. The split replay has 130/29/28 blocks and
`cross_split_block_overlap_count` 0. Refitting both logistic regressions on
the 1,438,231 replayed training rows reproduced the published 5-bit clear F1
with delta 0 (`known_lr_c1` 0.976132, `known_lr_c10` 0.974651).

Scored rows are the 1,136,774 cleaned flows inside the original 28 test
blocks that the coin flip skipped. That is 7.11 times the 159,917 sampled
test rows; the expected ratio at this probability is 7.12. Attack prevalence
is 15.0%. Of 170,177 attack rows, 132,909 (78.1%) are DDoS LOIC-HTTP.

Float thresholds were refit on the replayed validation blocks. The 5-bit
thresholds stayed frozen. `known_lr_c1` threshold 0.667279:

- Float: F1 0.9925, benign false-positive rate 0.22%. Float threshold
  0.626061.
- 5-bit clear: F1 0.9767, benign false-positive rate 0.79%.
- FHE simulation: F1 0.9767, benign false-positive rate 0.79%.

`known_lr_c10` threshold 0.821242:

- Float: F1 0.9932, benign false-positive rate 0.21%. Float threshold
  0.735486.
- 5-bit clear: F1 0.9752, benign false-positive rate 0.86%.
- FHE simulation: F1 0.9752, benign false-positive rate 0.86%.

Simulation matches 5-bit clear at four decimals for both models. The only
printed per-label gap is `known_lr_c1` on LOIC-HTTP: clear recall 0.9970,
simulation recall 0.9969.

Per-label support: benign 966,597, Bot 14,073, FTP brute force 18,476, SSH
brute force 4,625, LOIC-HTTP 132,909, web brute force 74, XSS 20. SQL
injection has no unsampled row in these blocks; its only cleaned test row
was in the sampled set. Bot, FTP brute force, and SSH brute force recall are
1.0 in float, 5-bit clear, and simulation for both models. Web brute force
recall is 28/74 in every mode. XSS recall is 8/20, except `known_lr_c10` at
5 bits and in simulation, where it is 11/20.

Both models were scored in one streaming pass, 6,571.7 seconds, about 174
rows/s. The 100-record end-to-end latencies stay 0.0156 s and 0.0157 s.

The published F1 remains the score on the 159,917 sampled test rows. This
run is the rest of those same blocks. The Bot rows are still the held-out
block of the Friday 2018-03-02 capture. Do not repeat this rescore.

## Next action

None on the CIC-IDS-2018 sample. Cite the sampled test in
`20261031_170906_440932`. The unsampled run checks those same blocks and
does not replace that score.

The radio screen finished in
`netslab-5g-oran-idd/screen_runs/20261107_155459_423954`. No model cleared
the scenario-holdout gate. Binary F1 values near 0.9589 match always-attack
on a test set of 6,532 rows (6,016 attack, 516 benign). The lowest benign
false-positive rate is 0.655 (`delta`, 20 trees, depth 4). Do not compile
that screen again.

The Zeek aggregate was read on 2026-11-07: `Network_Dataset.csv`, 227.2 MB,
1,723,817 rows, 26 columns, label `attack_category`. It has no scenario
column and no timestamp, so the run stopped. A row-random split is refused.
Do not rerun `screen_network_benchmark.py` on this summary. The capture
name is the pcap filename and is not in the combined table. Re-extracting
Zeek per pcap is a different job, and those features are not the E2 input.

`Lower_Layer_Data.db` stores `pmi` as the text `(0, 0)` on every row. The
radio loader reads the first number, keeps `ue_id`, drops `rnti`, and uses
`attack_subcategory` as the capture subtype. Quantized trees, forests,
boosting, and the MLP are not written with joblib. Their scores still go to
the result tables. The one-byte `*_q*.joblib` files in the finished radio
run are truncated saves from the earlier pickling error.

The radio screen is plaintext and Concrete ML quantized-clear. Task A is
binary. Task B scores 3-class (`Benign` / high-rate / low-rate) and the six
Version 2 categories. `--compile-fhe` compiles one binary `load` or `delta`
linear model, or one tree of depth at most 4. Forests, XGBoost, and the
quantized MLP are accuracy screens. The Zeek script does not compile.
A row-random split is refused. If the summary table cannot name capture
scenarios, rerun the radio screen with `--lower-layer-dir` pointed at the
per-scenario `lower_layer` text files.

CIC-IDS-2018 is unchanged. An independent Bot recording is still required
before calling this botnet detection beyond CIC-IDS-2018. There is still no
passing FHE circuit for Bot detection against future benign traffic.

Full-data loading remains unsafe under the 20 GiB RAM cap.
`anomaly_model_experiments.py --sample-rows 0` and
`qualified_fhe_benchmark.py --sample-rows 0` must not be used. The earlier
full load, `anomaly_experiment_runs/20261030_233818_174372`, read all
16,233,002 rows and stopped.

## Repository state at handoff

- Existing result directories and reports have been preserved.
- The corrected two-million-row quality run is complete and passes gates 1–5.
- The qualified FHE benchmark has been run on CUDA. `known_lr_c1` and
  `known_lr_c10` pass. Distance and PCA fail.
- The unsampled rescore has been run. On the other 1,136,774 flows in the
  original 28 test blocks, both logistic regressions stay above F1 0.97 in
  float, 5-bit clear, and FHE simulation. The published F1 is still the
  sampled test.
- Experiments are executed outside this workspace. The CIC-IDS-2018 CSV files
  are not stored in the repository.
