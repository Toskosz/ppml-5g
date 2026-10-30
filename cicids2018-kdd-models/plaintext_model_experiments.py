"""Plaintext screening for FHE-friendly CIC-IDS-2018 classifiers.

This script intentionally does not compile FHE circuits or overwrite artifacts
from ``svd_model.py``. Every invocation creates a new timestamped directory
under ``plaintext_experiment_runs/`` containing the rationale, arguments,
incremental results, and final report.

Run from this directory, as with the existing trainers:

    python plaintext_model_experiments.py
    python plaintext_model_experiments.py --families lr xgb
    python plaintext_model_experiments.py --feature-sets baseline_svd15
    python plaintext_model_experiments.py --sample-rows 0  # full dataset

The default one-million-row sample is an unbiased streaming sample. It is
large enough for screening while keeping the expanded feature experiment
practical. Re-run promising configurations on the full dataset before FHE
compilation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

from data_load import clean_col_names, resolve_usecols


SEED = 42
BASELINE_NUMERIC = [
    "syn_flag_cnt",
    "ack_flag_cnt",
    "fin_flag_cnt",
    "rst_flag_cnt",
    "totlen_fwd_pkts",
]
BASELINE_CATEGORICAL = ["protocol", "dst_port"]

# Compact, identifier-free flow representation. These fields cover volume,
# packet sizes, timing, TCP state, directionality, windows, and idle/active
# behavior. IP addresses, Flow ID, and Timestamp are deliberately excluded:
# they make random-split scores look better through environment/date leakage
# but are poor deployment features.
FLOW_NUMERIC = [
    "flow_duration",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "totlen_fwd_pkts",
    "totlen_bwd_pkts",
    "fwd_pkt_len_max",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_max",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "flow_byts_s",
    "flow_pkts_s",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "fwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_mean",
    "bwd_iat_std",
    "pkt_len_mean",
    "pkt_len_std",
    "down_up_ratio",
    "syn_flag_cnt",
    "ack_flag_cnt",
    "fin_flag_cnt",
    "rst_flag_cnt",
    "psh_flag_cnt",
    "urg_flag_cnt",
    "init_fwd_win_byts",
    "init_bwd_win_byts",
    "active_mean",
    "idle_mean",
]
FLOW_SOURCE_COLUMNS = FLOW_NUMERIC + ["protocol", "dst_port"]

COMMON_PORTS = {
    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    110: "pop3",
    123: "ntp",
    135: "rpc",
    137: "netbios",
    139: "netbios",
    143: "imap",
    161: "snmp",
    389: "ldap",
    443: "https",
    445: "smb",
    993: "imaps",
    995: "pop3s",
    1433: "mssql",
    3306: "mysql",
    3389: "rdp",
    5432: "postgres",
    8080: "http_alt",
}

RATIONALE = [
    (
        "Model families",
        "Logistic regression is the strongest latency candidate because its FHE "
        "circuit is dominated by one dot product. Shallow XGBoost tests whether "
        "boosting can recover attack recall with the same small tree depths that "
        "were fast in the random-forest runs. The MLPs are intentionally limited "
        "to one 8- or 16-neuron hidden layer to bound later FHE cost.",
    ),
    (
        "Class imbalance",
        "CIC-IDS-2018 is about 83% benign in the current binary dataset. LR uses "
        "balanced class weights; XGBoost compares natural and benign/attack-ratio "
        "weights; MLP training uses a reproducible balanced subset because "
        "sklearn 1.5 MLPClassifier does not accept class weights.",
    ),
    (
        "Decision threshold",
        "The classification threshold is selected only on a validation split by "
        "maximizing binary F1, then frozen for the test split. The test split is "
        "never used to choose a threshold. F1 at the default 0.5 threshold is "
        "also recorded so threshold gains remain visible.",
    ),
    (
        "Feature controls",
        "baseline_svd15 reproduces the current seven source fields and 15-component "
        "SVD as a control. compact_flow adds identifier-free CICFlowMeter volume, "
        "packet-size, timing, flag, window, and active/idle statistics without SVD.",
    ),
    (
        "Destination port",
        "The current categorical destination-port encoding creates roughly 64,000 "
        "one-hot columns. compact_flow instead uses log-scaled port number, IANA "
        "range, and a small common-service category. This retains useful service "
        "information while keeping a future FHE input vector compact.",
    ),
    (
        "Heavy-tailed values",
        "compact_flow applies signed log1p before MinMax scaling. Durations, byte "
        "counts, rates, and inter-arrival statistics span many orders of magnitude; "
        "compressing tails reduces quantization pressure in a later FHE model.",
    ),
    (
        "Leakage and comparability",
        "The default stratified random split matches the existing experiment for a "
        "fair comparison, but can overstate deployment performance because flows "
        "from the same capture day appear in train and test. IP addresses, Flow ID, "
        "and Timestamp are excluded. A later winner should also be validated with "
        "capture-day holdout.",
    ),
    (
        "FHE limitation",
        "This is a plaintext screen. Plaintext latency does not predict FHE latency. "
        "LR and XGBoost map directly to Concrete ML model families; sklearn MLP is "
        "only an architecture screen and must be retrained as a quantization-aware "
        "Concrete ML NeuralNetClassifier before an FHE claim can be made.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen FHE-friendly XGB, LR, and small-MLP models in plaintext.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=("baseline_svd15", "compact_flow"),
        default=("baseline_svd15", "compact_flow"),
        help="Feature representations to evaluate.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("xgb", "lr", "mlp"),
        default=("xgb", "lr", "mlp"),
        help="Model families to evaluate.",
    )
    parser.add_argument(
        "--data-folder",
        default="CIC-IDS-2018",
        help="Folder containing original CIC-IDS-2018 day CSV files.",
    )
    parser.add_argument(
        "--combined-csv",
        default="CIC-IDS-2018-Combined.csv",
        help="Existing slim CSV usable when only baseline_svd15 is requested.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1_000_000,
        help="Target streaming sample size; 0 reads all rows.",
    )
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--mlp-max-epochs", type=int, default=50)
    parser.add_argument("--target-f1", type=float, default=0.70)
    parser.add_argument(
        "--output-root",
        default="plaintext_experiment_runs",
        help="New timestamped run directories are created here.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save fitted plaintext estimators and feature transformers in the run directory.",
    )
    return parser.parse_args()


def create_run_dir(output_root: str) -> Path:
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_logger(run_dir: Path):
    log_path = run_dir / "run.log"

    def log(message: str) -> None:
        timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def write_rationale(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("CIC-IDS-2018 plaintext experiment rationale\n")
        handle.write("=" * 48 + "\n\n")
        for title, body in RATIONALE:
            handle.write(f"{title}\n{'-' * len(title)}\n{body}\n\n")


def count_csv_data_rows(path: str) -> int:
    newline_count = 0
    last_byte = b""
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
    line_count = newline_count + (1 if last_byte and last_byte != b"\n" else 0)
    return max(0, line_count - 1)


def discover_sources(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    needs_raw = "compact_flow" in args.feature_sets
    raw_files = sorted(glob.glob(os.path.join(args.data_folder, "**", "*.csv"), recursive=True))
    if needs_raw:
        if not raw_files:
            raise FileNotFoundError(
                "compact_flow requires the original CIC-IDS-2018 day CSV files. "
                f"No CSVs were found under '{args.data_folder}'."
            )
        needed = sorted(set(FLOW_SOURCE_COLUMNS + BASELINE_NUMERIC + BASELINE_CATEGORICAL + ["label"]))
        return raw_files, needed

    if raw_files:
        return raw_files, sorted(set(BASELINE_NUMERIC + BASELINE_CATEGORICAL + ["label"]))
    if os.path.isfile(args.combined_csv):
        return [args.combined_csv], sorted(
            set(BASELINE_NUMERIC + BASELINE_CATEGORICAL + ["label"])
        )
    raise FileNotFoundError(
        f"Neither original CSVs under '{args.data_folder}' nor '{args.combined_csv}' exist."
    )


def load_streaming_sample(
    sources: list[str],
    needed_columns: list[str],
    target_rows: int,
    chunk_rows: int,
    log,
) -> pd.DataFrame:
    log("Counting source rows to set an unbiased streaming sampling probability...")
    row_counts = [count_csv_data_rows(path) for path in sources]
    total_rows = sum(row_counts)
    if total_rows == 0:
        raise ValueError("The selected CSV sources contain no data rows.")
    probability = 1.0 if target_rows <= 0 else min(1.0, target_rows / total_rows)
    log(
        f"Source rows: {total_rows:,}; sampling probability: {probability:.6f}; "
        f"requested target: {'all' if target_rows <= 0 else f'{target_rows:,}'}."
    )

    rng = np.random.default_rng(SEED)
    sampled_chunks: list[pd.DataFrame] = []
    sampled_rows = 0
    for file_index, path in enumerate(sources, start=1):
        usecols = resolve_usecols(path, needed_columns)
        log(f"Reading source {file_index}/{len(sources)}: {path}")
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            chunksize=chunk_rows,
            low_memory=False,
        ):
            chunk = clean_col_names(chunk)
            if probability < 1.0:
                chunk = chunk.loc[rng.random(len(chunk)) < probability]
            if not chunk.empty:
                sampled_chunks.append(chunk)
                sampled_rows += len(chunk)
        log(f"Sample accumulated: {sampled_rows:,} rows.")

    if not sampled_chunks:
        raise ValueError("Sampling produced zero rows. Increase --sample-rows.")
    sample = pd.concat(sampled_chunks, ignore_index=True)
    del sampled_chunks
    if target_rows > 0 and len(sample) > target_rows:
        sample = sample.sample(n=target_rows, random_state=SEED).reset_index(drop=True)
    log(f"Loaded unbiased sample with {len(sample):,} rows.")
    return sample


def clean_for_feature_set(sample: pd.DataFrame, feature_set: str, log) -> pd.DataFrame:
    if feature_set == "baseline_svd15":
        numeric = BASELINE_NUMERIC
        required = BASELINE_NUMERIC + BASELINE_CATEGORICAL + ["label"]
    else:
        numeric = FLOW_NUMERIC + ["dst_port"]
        required = FLOW_SOURCE_COLUMNS + ["label"]

    frame = sample.loc[:, required].copy()
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(frame)
    frame.dropna(subset=required, inplace=True)
    frame["protocol"] = frame["protocol"].astype(str)
    if feature_set == "baseline_svd15":
        frame["dst_port"] = frame["dst_port"].astype(str)
    else:
        frame = frame.loc[frame["dst_port"].between(0, 65535)].copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["binary_label"] = (frame["label"] != "benign").astype(np.int8)
    dropped = before - len(frame)
    log(
        f"{feature_set}: retained {len(frame):,} rows; dropped {dropped:,} rows "
        "with missing, infinite, or invalid required values."
    )
    if frame["binary_label"].nunique() != 2:
        raise ValueError(f"{feature_set}: both benign and attack rows are required.")
    return frame


def split_frame(
    frame: pd.DataFrame,
    test_size: float,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("--test-size and --validation-size must be positive and sum to < 1.")
    train_validation, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=SEED,
        stratify=frame["binary_label"],
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=SEED,
        stratify=train_validation["binary_label"],
    )
    return train, validation, test


def engineer_compact_flow(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in FLOW_NUMERIC:
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        result[column] = np.sign(values) * np.log1p(np.abs(values))

    ports = frame["dst_port"].round().astype(np.int64)
    result["dst_port_log"] = np.log1p(ports) / math.log1p(65535)
    result["dst_port_range"] = np.select(
        [ports <= 1023, ports <= 49151],
        ["well_known", "registered"],
        default="dynamic",
    )
    result["dst_port_service"] = ports.map(COMMON_PORTS).fillna("other")
    result["protocol"] = frame["protocol"].astype(str)
    return result


def fit_feature_transform(
    feature_set: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    log,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    if feature_set == "baseline_svd15":
        preprocessor = ColumnTransformer(
            [
                ("numeric", MinMaxScaler(), BASELINE_NUMERIC),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    BASELINE_CATEGORICAL,
                ),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        train_sparse = preprocessor.fit_transform(train)
        validation_sparse = preprocessor.transform(validation)
        test_sparse = preprocessor.transform(test)
        svd = TruncatedSVD(n_components=15, random_state=SEED)
        x_train = svd.fit_transform(train_sparse)
        x_validation = svd.transform(validation_sparse)
        x_test = svd.transform(test_sparse)
        transformer = {
            "feature_set": feature_set,
            "preprocessor": preprocessor,
            "svd": svd,
            "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
        }
        del train_sparse, validation_sparse, test_sparse
    else:
        train_engineered = engineer_compact_flow(train)
        validation_engineered = engineer_compact_flow(validation)
        test_engineered = engineer_compact_flow(test)
        categorical = ["protocol", "dst_port_range", "dst_port_service"]
        numeric = FLOW_NUMERIC + ["dst_port_log"]
        preprocessor = ColumnTransformer(
            [
                ("numeric", MinMaxScaler(), numeric),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                ),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        x_train = preprocessor.fit_transform(train_engineered)
        x_validation = preprocessor.transform(validation_engineered)
        x_test = preprocessor.transform(test_engineered)
        transformer = {
            "feature_set": feature_set,
            "preprocessor": preprocessor,
            "numeric_log1p": FLOW_NUMERIC,
            "port_engineering": "log-scaled number + IANA range + common service",
        }

    x_train = np.asarray(x_train, dtype=np.float32)
    x_validation = np.asarray(x_validation, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    log(
        f"{feature_set}: transformed dimensions={x_train.shape[1]}, "
        f"fit/transform time={time.perf_counter() - started:.1f}s."
    )
    return x_train, x_validation, x_test, transformer


def make_model_configs(families: list[str], jobs: int, mlp_max_epochs: int) -> list[dict]:
    configs: list[dict] = []
    if "lr" in families:
        for c_value in (0.1, 1.0, 10.0):
            configs.append(
                {
                    "id": f"lr_balanced_c{c_value:g}",
                    "family": "lr",
                    "params": {
                        "C": c_value,
                        "class_weight": "balanced",
                        "max_iter": 500,
                        "solver": "lbfgs",
                    },
                    "fhe_priority": "high",
                    "fhe_note": "One linear dot product; strongest <=0.7 s candidate.",
                }
            )
    if "xgb" in families:
        for depth, estimators in ((2, 2), (2, 4), (3, 2)):
            for weighting in ("natural", "balanced"):
                configs.append(
                    {
                        "id": f"xgb_d{depth}_n{estimators}_{weighting}",
                        "family": "xgb",
                        "params": {
                            "max_depth": depth,
                            "n_estimators": estimators,
                            "learning_rate": 0.3,
                            "weighting": weighting,
                        },
                        "fhe_priority": "high" if depth == 2 else "medium",
                        "fhe_note": (
                            "Depth 2 and <=4 trees bracket the fastest existing forest shapes."
                            if depth == 2
                            else "Depth 3 may improve F1 but has up to 7 nodes per tree."
                        ),
                        "jobs": jobs,
                    }
                )
    if "mlp" in families:
        for width in (8, 16):
            configs.append(
                {
                    "id": f"mlp_{width}",
                    "family": "mlp",
                    "params": {
                        "hidden_layer_sizes": (width,),
                        "max_iter": mlp_max_epochs,
                        "balanced_training_subset": True,
                    },
                    "fhe_priority": "medium" if width == 8 else "low",
                    "fhe_note": (
                        "Architecture screen only; requires QAT retraining before FHE."
                    ),
                }
            )
    return configs


def instantiate_model(config: dict, class_ratio: float, jobs: int):
    family = config["family"]
    params = config["params"]
    if family == "lr":
        return LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=params["max_iter"],
            solver=params["solver"],
            n_jobs=jobs,
            random_state=SEED,
        )
    if family == "xgb":
        scale_pos_weight = class_ratio if params["weighting"] == "balanced" else 1.0
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=jobs,
            random_state=SEED,
            use_label_encoder=False,
            verbosity=0,
        )
    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=params["max_iter"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=SEED,
        verbose=False,
    )


def balanced_training_indices(y_train: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    benign = np.flatnonzero(y_train == 0)
    attack = np.flatnonzero(y_train == 1)
    count = min(len(benign), len(attack))
    indices = np.concatenate(
        [
            rng.choice(benign, size=count, replace=False),
            rng.choice(attack, size=count, replace=False),
        ]
    )
    rng.shuffle(indices)
    return indices


def positive_scores(model, values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected binary predict_proba output with two columns.")
    return probabilities[:, 1]


def choose_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = (scores >= threshold).astype(np.int8)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def measure_plaintext_latency(model, x_test: np.ndarray) -> dict[str, float]:
    batch_count = min(10_000, len(x_test))
    single_count = min(100, len(x_test))
    model.predict_proba(x_test[: min(100, batch_count)])

    started = time.perf_counter()
    model.predict_proba(x_test[:batch_count])
    batch_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for index in range(single_count):
        model.predict_proba(x_test[index : index + 1])
    single_seconds = time.perf_counter() - started
    return {
        "plaintext_batch_us_per_record": 1e6 * batch_seconds / batch_count,
        "plaintext_single_ms_mean": 1e3 * single_seconds / single_count,
    }


def run_one_model(
    config: dict,
    feature_set: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_f1: float,
    jobs: int,
    run_dir: Path,
    save_models: bool,
    log,
) -> dict[str, Any]:
    benign_count = int((y_train == 0).sum())
    attack_count = int((y_train == 1).sum())
    class_ratio = benign_count / attack_count
    model = instantiate_model(config, class_ratio, jobs)
    fit_x, fit_y = x_train, y_train
    balanced_rows = None
    if config["family"] == "mlp":
        indices = balanced_training_indices(y_train)
        fit_x, fit_y = x_train[indices], y_train[indices]
        balanced_rows = len(indices)

    log(f"{feature_set}/{config['id']}: fitting on {len(fit_y):,} rows...")
    started = time.perf_counter()
    model.fit(fit_x, fit_y)
    fit_seconds = time.perf_counter() - started

    validation_scores = positive_scores(model, x_validation)
    threshold, validation_f1 = choose_f1_threshold(y_validation, validation_scores)
    test_scores = positive_scores(model, x_test)
    default_metrics = classification_metrics(y_test, test_scores, 0.5)
    tuned_metrics = classification_metrics(y_test, test_scores, threshold)
    latency = measure_plaintext_latency(model, x_test)

    result = {
        "status": "ok",
        "feature_set": feature_set,
        "model_id": config["id"],
        "family": config["family"],
        "params": json.dumps(config["params"], sort_keys=True),
        "input_dimensions": int(x_train.shape[1]),
        "train_rows": int(len(fit_y)),
        "balanced_training_rows": balanced_rows,
        "fit_seconds": fit_seconds,
        "selected_threshold": threshold,
        "validation_f1": validation_f1,
        "test_f1_default_0_5": default_metrics["f1"],
        "test_accuracy": tuned_metrics["accuracy"],
        "test_precision": tuned_metrics["precision"],
        "test_recall": tuned_metrics["recall"],
        "test_f1": tuned_metrics["f1"],
        "test_average_precision": tuned_metrics["average_precision"],
        "test_roc_auc": tuned_metrics["roc_auc"],
        "quality_target_met": tuned_metrics["f1"] >= target_f1,
        "fhe_priority": config["fhe_priority"],
        "fhe_note": config["fhe_note"],
        **latency,
    }
    log(
        f"{feature_set}/{config['id']}: test F1={result['test_f1']:.4f}, "
        f"precision={result['test_precision']:.4f}, recall={result['test_recall']:.4f}, "
        f"threshold={threshold:.4f}, fit={fit_seconds:.1f}s."
    )

    if save_models:
        model_dir = run_dir / "models" / feature_set
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "config": config,
                "feature_set": feature_set,
            },
            model_dir / f"{config['id']}.joblib",
        )
    return result


def write_results(results: list[dict], run_dir: Path) -> None:
    pd.DataFrame(results).to_csv(run_dir / "results.csv", index=False)
    write_json(run_dir / "results.json", results)


def write_report(
    results: list[dict],
    run_dir: Path,
    args: argparse.Namespace,
    dataset_summary: dict[str, Any],
) -> None:
    successful = pd.DataFrame([row for row in results if row.get("status") == "ok"])
    failed = [row for row in results if row.get("status") != "ok"]
    lines = [
        "CIC-IDS-2018 plaintext FHE-candidate experiment",
        "=" * 51,
        f"Run directory: {run_dir}",
        f"Generated: {dt.datetime.now().astimezone().isoformat()}",
        f"Arguments: {json.dumps(vars(args), sort_keys=True)}",
        "",
        "Dataset",
        "-------",
    ]
    lines.extend(f"{key}: {value}" for key, value in dataset_summary.items())
    lines.extend(["", "Selection rules", "---------------"])
    lines.append(
        f"Quality target: test binary F1 >= {args.target_f1:.3f}. Threshold selected on validation."
    )
    lines.append(
        "FHE latency target is not asserted here; fhe_priority is a structural screening label."
    )
    lines.extend(["", "Results (sorted by test F1)", "---------------------------"])
    if not successful.empty:
        columns = [
            "feature_set",
            "model_id",
            "input_dimensions",
            "test_f1",
            "test_precision",
            "test_recall",
            "selected_threshold",
            "test_f1_default_0_5",
            "fit_seconds",
            "quality_target_met",
            "fhe_priority",
        ]
        ranked = successful.sort_values("test_f1", ascending=False)[columns].copy()
        for column in (
            "test_f1",
            "test_precision",
            "test_recall",
            "selected_threshold",
            "test_f1_default_0_5",
            "fit_seconds",
        ):
            ranked[column] = ranked[column].map(lambda value: f"{value:.4f}")
        lines.append(ranked.to_string(index=False))
    else:
        lines.append("No model completed successfully.")

    lines.extend(["", "Recommended FHE follow-up", "-------------------------"])
    if not successful.empty:
        candidates = successful[
            (successful["quality_target_met"] == True)  # noqa: E712
            & (successful["fhe_priority"].isin(["high", "medium"]))
        ].sort_values(["fhe_priority", "test_f1"], ascending=[True, False])
        if candidates.empty:
            lines.append(
                "No configuration met both the plaintext quality gate and a high/medium "
                "structural FHE priority. Do not spend hours compiling these results yet."
            )
        else:
            for _, row in candidates.iterrows():
                lines.append(
                    f"- {row['feature_set']}/{row['model_id']}: F1={row['test_f1']:.4f}, "
                    f"priority={row['fhe_priority']}. {row['fhe_note']}"
                )
            lines.append(
                "Before compiling: retrain the selected configuration on the full dataset, "
                "then verify its quantized clear-mode F1 in Concrete ML."
            )
    if failed:
        lines.extend(["", "Failures", "--------"])
        for row in failed:
            lines.append(f"- {row.get('feature_set')}/{row.get('model_id')}: {row.get('error')}")

    lines.extend(["", "Recorded rationale", "------------------"])
    for title, body in RATIONALE:
        lines.extend([f"{title}:", body, ""])
    (run_dir / "report.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = create_run_dir(args.output_root)
    log = make_logger(run_dir)
    write_json(run_dir / "arguments.json", vars(args))
    write_rationale(run_dir / "rationale.txt")
    write_json(
        run_dir / "status.json",
        {"status": "running", "started": dt.datetime.now().astimezone().isoformat()},
    )
    log(f"New isolated run directory: {run_dir}")
    log("No existing model, report, preprocessor, or FHE artifact will be modified.")

    results: list[dict] = []
    try:
        sources, needed_columns = discover_sources(args)
        sample = load_streaming_sample(
            sources,
            needed_columns,
            args.sample_rows,
            args.chunk_rows,
            log,
        )
        model_configs = make_model_configs(
            list(args.families),
            args.jobs,
            args.mlp_max_epochs,
        )
        dataset_summary: dict[str, Any] = {
            "sample_rows_before_feature_cleaning": len(sample),
            "source_files": len(sources),
            "source_rows": sum(count_csv_data_rows(path) for path in sources),
            "split": (
                f"{1 - args.test_size - args.validation_size:.0%} train / "
                f"{args.validation_size:.0%} validation / {args.test_size:.0%} test"
            ),
            "split_strategy": "stratified random binary-label split, seed 42",
        }

        for feature_set in args.feature_sets:
            frame = clean_for_feature_set(sample, feature_set, log)
            benign = int((frame["binary_label"] == 0).sum())
            attack = int((frame["binary_label"] == 1).sum())
            dataset_summary[f"{feature_set}_rows"] = len(frame)
            dataset_summary[f"{feature_set}_benign"] = benign
            dataset_summary[f"{feature_set}_attack"] = attack

            train, validation, test = split_frame(
                frame,
                args.test_size,
                args.validation_size,
            )
            y_train = train["binary_label"].to_numpy(dtype=np.int8, copy=True)
            y_validation = validation["binary_label"].to_numpy(dtype=np.int8, copy=True)
            y_test = test["binary_label"].to_numpy(dtype=np.int8, copy=True)
            x_train, x_validation, x_test, transformer = fit_feature_transform(
                feature_set,
                train,
                validation,
                test,
                log,
            )
            dataset_summary[f"{feature_set}_input_dimensions"] = x_train.shape[1]
            if feature_set == "baseline_svd15":
                dataset_summary[f"{feature_set}_svd_explained_variance"] = transformer[
                    "explained_variance_ratio"
                ]
            if args.save_models:
                transformer_dir = run_dir / "models" / feature_set
                transformer_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(transformer, transformer_dir / "feature_transformer.joblib")

            del frame, train, validation, test
            for config in model_configs:
                try:
                    result = run_one_model(
                        config,
                        feature_set,
                        x_train,
                        y_train,
                        x_validation,
                        y_validation,
                        x_test,
                        y_test,
                        args.target_f1,
                        args.jobs,
                        run_dir,
                        args.save_models,
                        log,
                    )
                except Exception as exc:  # Preserve partial results and continue the grid.
                    log(f"{feature_set}/{config['id']} failed: {exc}")
                    result = {
                        "status": "failed",
                        "feature_set": feature_set,
                        "model_id": config["id"],
                        "family": config["family"],
                        "params": json.dumps(config["params"], sort_keys=True),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                results.append(result)
                write_results(results, run_dir)
            del x_train, x_validation, x_test, y_train, y_validation, y_test

        write_report(results, run_dir, args, dataset_summary)
        write_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "finished": dt.datetime.now().astimezone().isoformat(),
                "successful_models": sum(row.get("status") == "ok" for row in results),
                "failed_models": sum(row.get("status") != "ok" for row in results),
            },
        )
        log(f"Experiment complete. Read {run_dir / 'report.txt'}")
        return 0
    except Exception as exc:
        write_results(results, run_dir)
        write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "finished": dt.datetime.now().astimezone().isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        log(f"Experiment failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
