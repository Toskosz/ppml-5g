# AGENTS.md — PPML-5G

## What this is

FHE-based Random Forest classifiers for 5G/O-RAN network intrusion detection, built on [Zama Concrete ML](https://github.com/zama-ai/concrete-ml). Two model families train on public IDS datasets and compile to FHE circuits for encrypted inference.

## Setup

```bash
pip install -r requirements.txt        # Linux (includes CUDA deps)
pip install -r macos_requirements.txt   # macOS (no CUDA)

# Optional: GPU-accelerated Concrete FHE runtime (Linux + NVIDIA CUDA >= 11.8)
pip uninstall -y concrete-python
pip install -r requirements-gpu.txt
```

Pinned versions matter: `concrete-ml==1.9.0`, `numpy==1.26.4`. Do not upgrade these independently.

FHE device (both model dirs): `PPML_FHE_DEVICE=auto|cpu|cuda` or `run_all_and_report.py --device ...`. Default `auto` uses CUDA when the GPU Concrete wheel reports available+enabled; otherwise CPU.

## Running

**Training / FHE circuit compilation** (run from inside each model directory — scripts use relative paths):

```bash
cd cicids2018-kdd-models && PPML_FHE_DEVICE=auto python svd100_model_fast.py
cd cicunswnb15-models && PPML_FHE_DEVICE=auto python svd100_model_fast.py
# Full suite + report.txt from either directory:
python run_all_and_report.py --device auto --only-fast
```

**Single-record inference timing:**

```bash
python cicids2018-kdd-models/best_model_single_record.py
python cicunswnb15-models/single_record_inference.py
```

FHE circuit compilation for large models (many estimators / deep trees) can take hours.

## Datasets

Datasets are **not** in the repo (`*.csv` is gitignored). They must be downloaded separately:

| Model dir | Dataset | Download |
|---|---|---|
| `cicids2018-kdd-models/` | CSE-CIC-IDS-2018 | `aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/` |
| `cicunswnb15-models/` | CIC-UNSW-NB15 (2024) | `wget -r -np -nd -A "*.csv" http://cicresearch.ca/CICDataset/CIC-UNSW/` |

Training scripts expect data at the working directory (e.g. `CIC-IDS-2018/` folder or `CICFlowMeter_out.csv`).

## Architecture

- No package structure, no tests, no CI, no linter/formatter — each directory is a standalone experiment.
- `cicids2018-kdd-models/` is a **binary** classifier (benign/attack), training with both `TruncatedSVD(100)` and `TruncatedSVD(200)`.
- `cicunswnb15-models/` is a **binary** classifier (benign/attack), with `svd100_model.py` / `svd200_model.py` plus `*_fast.py` variants, latency helpers, GPU support, and `run_all_and_report.py` (same patterns as `cicids2018-kdd-models/`).
- Both share the same 7-feature CICFlowMeter schema: `syn_flag_cnt`, `ack_flag_cnt`, `fin_flag_cnt`, `rst_flag_cnt`, `totlen_fwd_pkts` (numerical) + `protocol`, `dst_port` (categorical).
- Preprocessing pipeline: `MinMaxScaler` + `OneHotEncoder` → `TruncatedSVD(n_components)` → saved as `.pkl`.

## Gotchas

- **CWD matters.** All scripts use relative paths. Run the selected trainer from inside the model directory.
- **`*.pkl` is gitignored**, but some legacy `.pkl` files were committed before the gitignore rule. Do not assume all preprocessors are in the repo.
- **`cicids2018-kdd-models/`** trains with both `TruncatedSVD(100)` and `TruncatedSVD(200)`.
- Inference scripts (`best_model_single_record.py`, `single_record_inference.py`) re-read the CSV and apply saved preprocessor/SVD on every invocation. This is a known performance issue documented in README TODOs.
- The `cicunswnb15-models/` label column is `Label` (capital L), while the CIC-IDS-2018 models use `label` (lowercase) after `clean_col_names()`.
- `Train.txt` and `Test.txt` at the repo root are legacy KDD Cup 1999 data files — they are **not** used by any current model.
