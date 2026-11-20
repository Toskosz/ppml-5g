# AGENTS.md — PPML-5G

## What this is

FHE-based Random Forest classifiers for 5G/O-RAN network intrusion detection, built on [Zama Concrete ML](https://github.com/zama-ai/concrete-ml). Two model families train on public IDS datasets and compile to FHE circuits for encrypted inference.

## Agent environment

Training, FHE compilation, and inference run on another machine. This workspace does not have the dataset or the Concrete ML runtime. Do not install dependencies and do not run the experiment scripts. Change the script, document the command in `STATUS.md`, and leave execution to that machine.

## Setup

```bash
pip install -r requirements.txt        # Linux (includes CUDA deps)
pip install -r macos_requirements.txt   # macOS (no CUDA)

# Optional: GPU-accelerated Concrete FHE runtime (Linux + NVIDIA CUDA >= 11.8)
pip uninstall -y concrete-python
pip install -r requirements-gpu.txt
```

Pinned versions matter: `concrete-ml==1.9.0`, `numpy==1.26.4`. Do not upgrade these independently.

FHE device: `PPML_FHE_DEVICE=auto|cpu|cuda`, or `--device` on `run_all_and_report.py` and `compile_screen_fhe.py`. Default `auto` uses CUDA when the GPU Concrete wheel reports available+enabled; otherwise CPU.

## Running

**Training / FHE circuit compilation** (run from inside each model directory — scripts use relative paths):

```bash
cd cicids2018-kdd-models && PPML_FHE_DEVICE=auto python svd_model.py
cd cicunswnb15-models && python screen_feature_sets.py
cd cicunswnb15-models && python compile_screen_fhe.py
# CIC-IDS full suite + run_logs/<timestamp>/report.txt from inside that model directory.
python run_all_and_report.py --device auto
```

**Single-record inference timing:**

```bash
python cicids2018-kdd-models/best_model_single_record.py
```

FHE circuit compilation for large models (many estimators / deep trees) can take hours.

The current CIC-IDS-2018 latency step is `cicids2018-kdd-models/qualified_fhe_benchmark.py`. Run it on the dataset machine. The command and pass rule are in `STATUS.md`.

The current CIC-UNSW-NB15 step is `cicunswnb15-models/compile_screen_fhe.py`. Run it on the dataset machine after the feature-set screen. The command and pass rule are in `STATUS.md`.

## Datasets

Datasets are **not** in the repo (`*.csv` is gitignored). They must be downloaded separately:

| Model dir | Dataset | Download |
|---|---|---|
| `cicids2018-kdd-models/` | CSE-CIC-IDS-2018 | `aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/` |
| `cicunswnb15-models/` | CIC-UNSW-NB15 (2024) | `wget -r -np -nd -A "*.csv" http://cicresearch.ca/CICDataset/CIC-UNSW/` |

Training scripts expect data at the working directory (e.g. `CIC-IDS-2018/` folder or `CICFlowMeter_out.csv`).

## Architecture

- No package structure, no tests, no CI, no linter/formatter — each directory is a standalone experiment.
- `cicids2018-kdd-models/` is a **binary** classifier (benign/attack), training with `TruncatedSVD` at 15, 25, 40, and 100 components (`svd_model.py`, 5-bit quantization).
- `cicunswnb15-models/` is a **binary** classifier (benign/attack) on CIC-UNSW-NB15. `screen_feature_sets.py` screens the columns in `FEATURES.md`. `compile_screen_fhe.py` compiles those winning forests and times 100 real FHE records.
- CIC-IDS-2018 uses the 7-feature CICFlowMeter schema: `syn_flag_cnt`, `ack_flag_cnt`, `fin_flag_cnt`, `rst_flag_cnt`, `totlen_fwd_pkts` (numerical) + `protocol`, `dst_port` (categorical). Its preprocessing is `MinMaxScaler` + `OneHotEncoder` → `TruncatedSVD(n_components)` → saved as `.pkl`.

## Gotchas

- **CWD matters.** All scripts use relative paths. Run the selected trainer from inside the model directory.
- **`*.pkl` is gitignored**, but some legacy `.pkl` files were committed before the gitignore rule. Do not assume all preprocessors are in the repo.
- **`cicids2018-kdd-models/`** trains with `TruncatedSVD` at 15, 25, 40, and 100 components.
- `cicids2018-kdd-models/best_model_single_record.py` re-reads the CSV and applies the saved preprocessor/SVD on every invocation. This is a known performance issue documented in README TODOs.
- The `cicunswnb15-models/` label column is `Label` (capital L), while the CIC-IDS-2018 models use `label` (lowercase) after `clean_col_names()`.
- `Train.txt` and `Test.txt` at the repo root are legacy KDD Cup 1999 data files — they are **not** used by any current model.
