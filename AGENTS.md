# AGENTS.md — PPML-5G

## What this is

FHE-based Random Forest classifiers for 5G/O-RAN network intrusion detection, built on [Zama Concrete ML](https://github.com/zama-ai/concrete-ml). Three model families train on public IDS datasets and compile to FHE circuits for encrypted inference.

## Setup

```bash
pip install -r requirements.txt        # Linux (includes CUDA deps)
pip install -r macos_requirements.txt   # macOS (no CUDA)
```

Pinned versions matter: `concrete-ml==1.9.0`, `numpy==1.26.4`. Do not upgrade these independently.

## Running

**Training / FHE circuit compilation** (run from inside each model directory — scripts use relative paths like `CIC-IDS-2018/`):

```bash
python cicids2018-kdd-models/model.py
python cicids2018-models/cicids2018-modelv3.py
python cicunswnb15-models/model.py
```

**Single-record inference timing:**

```bash
python cicids2018-kdd-models/best_model_single_record.py
python cicids2018-models/single_record_inference.py
python cicunswnb15-models/single_record_inference.py
```

FHE circuit compilation for large models (many estimators / deep trees) can take hours.

## Datasets

Datasets are **not** in the repo (`*.csv` is gitignored). They must be downloaded separately:

| Model dir | Dataset | Download |
|---|---|---|
| `cicids2018-kdd-models/` | CSE-CIC-IDS-2018 | `aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/` |
| `cicids2018-models/` | CSE-CIC-IDS-2018 | same as above |
| `cicunswnb15-models/` | CIC-UNSW-NB15 (2024) | `wget -r -np -nd -A "*.csv" http://cicresearch.ca/CICDataset/CIC-UNSW/` |

Training scripts expect data at the working directory (e.g. `CIC-IDS-2018/` folder or `CICFlowMeter_out.csv`).

## Architecture

- No package structure, no tests, no CI, no linter/formatter — each directory is a standalone experiment.
- `cicids2018-kdd-models/` and `cicids2018-models/` are **binary** classifiers (benign/attack).
- `cicunswnb15-models/` is **multi-class** (9 categories); its metrics use `average='weighted'`.
- All three share the same 7-feature CICFlowMeter schema: `syn_cnt`, `ack_cnt`, `fin_cnt`, `rst_cnt`, `tot_l_fw_pkt` (numerical) + `protocol`, `dst_port` (categorical).
- Preprocessing pipeline: `MinMaxScaler` + `OneHotEncoder` → `TruncatedSVD(n_components)` → saved as `.pkl`.

## Gotchas

- **CWD matters.** All scripts use relative paths. Run `python model.py` from inside the model directory.
- **`*.pkl` is gitignored**, but some legacy `.pkl` files were committed before the gitignore rule. Do not assume all preprocessors are in the repo.
- **`cicids2018-kdd-models/`** uses `TruncatedSVD(100)`. **`cicids2018-models/`** uses `TruncatedSVD(200)`. The SVD dimension differs.
- Inference scripts (`single_record_inference.py`) re-run the full preprocessing pipeline on every invocation (they call `load_and_preprocess()` which re-reads the CSV). This is a known performance issue documented in README TODOs.
- The `cicunswnb15-models/` label column is `Label` (capital L), while the CIC-IDS-2018 models use `label` (lowercase) after `clean_col_names()`.
- `Train.txt` and `Test.txt` at the repo root are legacy KDD Cup 1999 data files — they are **not** used by any current model.
