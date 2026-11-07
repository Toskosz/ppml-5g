# NetsLab xApp experiment

Binary benign-vs-attack detection on E2 radio telemetry, for a model that
later compiles to FHE and has to finish inside a one-second xApp loop.
A 3-class grouping and the six Version 2 categories are scored on the same
radio table. The Zeek flow table is a separate accuracy benchmark. It is
not the compiled xApp input.

The radio input is `Lower_Layer_Data.db` (one UE record per second, about
45k rows, 22 columns). Version 2 only. Legacy floods are dropped when the
file can identify them.

Run this on the dataset machine, from this directory. This workspace does
not have the database or the Concrete ML runtime.

```bash
# Summary tables only. Do not pull the ~146 GB archive for this screen.
kaggle datasets download -d netslabdemo/netslab-5g-oran-idd \
  -p data --unzip

python screen_radio_xapp.py --self-check
python screen_radio_xapp.py --db data/Lower_Layer_Data.db
python screen_network_benchmark.py --csv data/Network_Dataset.csv
```

Each run writes a new directory under `screen_runs/`. FHE compilation stays
off unless `--compile-fhe` is passed, and a row-random split is refused.

If the summary table has no capture-scenario id and time-gap segmentation
cannot build a train/validation/test split, stop. Point the same script at
the per-scenario radio logs (one file, one scenario) instead of splitting
rows:

```bash
python screen_radio_xapp.py --lower-layer-dir data/lower_layer
```

## Features

Identifiers stay out of the model: `timestamp`, `ue_id`, `cellid`. Two UEs
and one cell would let a model memorize the lab. `ue_id` and `timestamp` are
used only to build scenarios and one-second deltas. A table that only names
the UE as `rnti` uses that column as `ue_id`. When both are present, `ue_id`
is kept and `rnti` is dropped. `rnti` is not a model feature in either case.

No SVD, no PCA, no one-hot, no SMOTE. Byte counts use `log1p` (values below
0 clipped to 0). `MinMaxScaler` is fit on the training scenarios only, and
transformed values are clipped to `[0, 1]` so a later quantizer stays inside
the training range. `ri`, `pmi`, `cqi`, MCS, and `in_sync` stay numeric. `pmi` may be stored as a
pair such as `(0, 0)`; the feature is the first number.

1. **load** — `log1p(dlBytes)`, `log1p(ulBytes)`, `dlBler`, `ulBler`, `dlMcs`,
   `ulMcs`, `phr`, `pcmax`, `ri`, `pmi`, `in_sync`.
2. **full** — load plus `rsrp`, `rsrq`, `rssi`, `sinr`, `cqi`, `pucchSnr`,
   `puschSnr`.
3. **full_reduced** — full without `pucchSnr`, `puschSnr`, and `cqi`, and
   only when those three are the lowest mutual-information features on the
   training scenarios. The mutual-information ranking is always written to
   the run directory. This set is a comparison. It is not compiled.
4. **delta** — load plus the one-second change in bytes and BLER. Built only
   when every load model and every full model misses the float quality gate.
   Within one `ue_id` and one scenario, byte deltas are
   `log1p(current) - log1p(previous)` and BLER deltas are raw differences.
   The delta is 0 on the first sample and whenever the previous sample is not
   0.5–1.5 seconds earlier. Deltas do not cross a scenario boundary.

If full passes the float gate and load does not, the report calls that a
channel confound (`rsrp` / `sinr` moving between lab sessions). The xApp
candidate is load, or delta when load and full both miss. A full-only win is
not compiled.

## Split

The split unit is a capture scenario. Adjacent one-second rows from the same
flood stay in the same split.

Scenario id, in order:

1. A source-file column, when it is present and is not a row id.
2. Otherwise the label subtype plus a new run wherever the timestamp gap
   inside that subtype exceeds 5 seconds (`--scenario-gap-seconds`). The
   subtype is `attack_subcategory` when that column is present (`DoS_SYN`,
   `Web_XSS`), not the coarse category. A series sampled about once a second
   still splits at 5 seconds. A slower series starts a new run only after a
   gap of at least 15 times its median spacing, so a few missing samples are
   not a new capture.

Assignment is deterministic. Scenario ids are sorted within each Version 2
category (`Benign`, `DoS`, `DDoS`, `Probe`, `BruteForce`, `Web`):

- Three or more scenarios: the last id is test, the second-to-last is
  validation, and the rest train.
- Two scenarios: the first trains, the second tests.
- One scenario: it stays in train, and the report lists the category as
  not held out.

Validation exists only when some category has at least three scenarios. A
category with two scenarios cannot spare a third split without giving up
its only training scenario. If no validation scenario exists, or if no test
scenario exists, the run stops and names `--lower-layer-dir`. It does not
cut a scenario into rows to manufacture a validation set.

The decision threshold maximizes binary F1 on validation scenarios and is
frozen before test. Logistic regression uses `class_weight="balanced"`.
Class counts are recorded. Rows are not resampled.

The family-holdout report refits the compile-candidate linear models and the
depth-3 and depth-4 trees: train and validate without Web and score every
Web row, then the same for BruteForce. Thresholds use the remaining
validation scenarios. Those fits are not compiled. If Web or BruteForce
recall on the main binary test split is below 0.10, the claim is flood/probe
detection on radio KPIs.

## Models

Float sklearn first, then Concrete ML quantized-clear at the bit width listed
for that shape. `SGDClassifier` is not a Concrete ML 1.9 estimator.
`LinearSVC` is the second linear model.

Task A, binary, every feature set:

| Model | Quantized bits | Role |
|---|---|---|
| Logistic regression `C=1` | 6 and 8 | Compile candidate |
| Logistic regression `C=10` | 6 | Compile candidate |
| `LinearSVC` `C=1` | 6 and 8 | Compile candidate |
| One tree, depth 3 | 6 | Compile candidate |
| One tree, depth 4 | 5 and 6 | Compile candidate |
| One tree, depth 6 | 6 | Accuracy screen. `--compile-wide` can compile it |
| Random forest, 10 trees, depth 3 | 5 | Accuracy screen |
| Random forest, 20 trees, depth 4 | 5 | Accuracy screen |
| XGBoost, 10 trees, depth 3 | 5 | Binary accuracy screen |
| XGBoost, 20 trees, depth 4 | 5 | Binary accuracy screen |
| Quantized MLP, one hidden layer at 1.5× the input width, accumulator width 8 | 4 | Accuracy screen |

Task B uses the same scenario split on `load` and `full` only.

- **3-class:** `Benign`, `HighRate` (`DoS`, `DDoS`, `Probe`), `LowRate`
  (`BruteForce`, `Web`).
- **6-class:** the six Version 2 categories. Rows labeled only `Attack`
  stay out of both tasks.

Task B trains logistic regression, the trees, both forests, and the MLP.
It does not train `XGBClassifier`. In Concrete ML a multi-class boosted
model is one-vs-rest, so a 20-tree booster becomes 120 trees for six
classes. Forests and the MLP emit the class vector from one model.
Task B is not compiled. Macro F1 uses the same 0.70 / 5% benign
false-positive gates and is reported per class.

`--compile-fhe` compiles one binary `load` or `delta` model: logistic
regression, then `LinearSVC`, then the shallowest single tree of depth 3
or 4, lower bit width first. `--compile-wide` adds the depth-6 tree.
Forests, XGBoost, and the MLP stay uncompiled. A 4-tree depth-4 forest on
this Concrete ML stack measured about 1.8–2.2 seconds end to end, so the
150–850 ms band quoted for 20 trees is not used as a latency budget.

The family-holdout report refits only those compile candidates.

## Network benchmark

`screen_network_benchmark.py` reads `Network_Dataset.csv`. It prints a
line before that read. Label detection scores distinct values, so a wide
table does not run the category parser on every cell. The script drops
identifiers, log-compresses heavy numeric tails, one-hot encodes
categoricals with at most 16 levels, and keeps the top 16 and top 32
mutual-information columns on a training-scenario sample of at most
200,000 rows. The model fit still uses every training row. The same
scenario rule applies. The script refuses `--compile-fhe`: these features
are not on the E2 path an xApp reads.

## Gates

On the scenario-held-out test set, a binary model is eligible to compile
when all of these hold:

- Binary F1 at least 0.70.
- Benign false-positive rate at most 5%, with benign rows present in test.
- Quantized-clear F1 within 0.05 of that model's float F1, and the float
  model itself meets the F1 and false-positive gates.
- Feature set is `load` or `delta`.
- The shape is a compile candidate above.

The compiled circuit is scored with FHE simulation at the frozen threshold.
Mean encrypt + inference + decrypt uses 100 shuffled test rows after a
fixed seed. Key generation is reported and is not part of that mean.

Latency: preferred mean end-to-end at most 0.25 seconds so two UE records
fit in the tick. Hard stop at 0.70 seconds, and only as a single record per
tick. A circuit that misses 0.70 seconds fails. A slower shape is not
compiled in its place.
