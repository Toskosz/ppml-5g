"""Evaluate FHE-oriented anomaly detectors on CIC-IDS-2018.

This experiment answers two different questions without modifying any existing
model or report:

1. Unknown-anomaly protocol: fit PCA/distance models using benign training
   traffic only, set thresholds from future benign validation traffic, and test
   on complete later capture days containing unseen attacks.
2. Known-family protocol: train compact logistic regression on attack families
   that have at least three separate 30-minute capture blocks, assigning whole
   blocks to train/validation/test.

Every invocation creates a new timestamped directory under
``anomaly_experiment_runs/``.

Run from this directory:

    python anomaly_model_experiments.py
    python anomaly_model_experiments.py --sample-rows 0
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from plaintext_model_experiments import (
    FLOW_NUMERIC,
    FLOW_SOURCE_COLUMNS,
    SEED,
    capture_date,
    classification_metrics,
    create_run_dir,
    engineer_compact_flow,
    load_streaming_sample,
    per_label_detection,
    positive_scores,
    write_json,
)


RATIONALE = [
    (
        "Unknown-anomaly protocol",
        "Models are fitted using benign traffic only. Thresholds are selected from "
        "benign traffic on later validation days at explicit false-positive-rate "
        "budgets. Attack rows do not influence fitting or threshold selection.",
    ),
    (
        "Capture-day test",
        "The latest two capture days remain completely held out. This tests future "
        "benign traffic and attack families absent from model fitting.",
    ),
    (
        "PCA reconstruction",
        "Distance from a low-dimensional benign subspace can detect flows that do "
        "not follow normal correlations. PCA uses linear transforms plus squared "
        "reconstruction error, making a future custom FHE circuit plausible.",
    ),
    (
        "Benign distance",
        "Standardized and robust squared distances are very small arithmetic "
        "circuits. K-means distance allows several benign modes, but selecting the "
        "nearest centroid adds encrypted comparisons and may cost more in FHE.",
    ),
    (
        "Known-family baseline",
        "A separate LR baseline includes only labels with at least three distinct "
        "time blocks. Whole blocks, rather than rows, are assigned to splits. This "
        "tests recognition of previously seen attack families without claiming "
        "unknown-attack detection.",
    ),
    (
        "FHE gate",
        "This script does not compile circuits. Compilation is justified only if a "
        "candidate clears F1 >= 0.70 while respecting its benign false-positive "
        "budget on the stricter test protocol.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test benign-only anomaly scores and a known-family LR baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-folder", default="CIC-IDS-2018")
    parser.add_argument("--sample-rows", type=int, default=1_000_000)
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument(
        "--benign-fpr",
        nargs="+",
        type=float,
        default=(0.01, 0.05),
        help="Validation benign false-positive budgets used to select thresholds.",
    )
    parser.add_argument(
        "--pca-components",
        nargs="+",
        type=int,
        default=(8, 16, 32),
    )
    parser.add_argument("--time-block-minutes", type=int, default=30)
    parser.add_argument("--target-f1", type=float, default=0.70)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--output-root", default="anomaly_experiment_runs")
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def make_logger(run_dir: Path):
    log_path = run_dir / "run.log"

    def log(message: str) -> None:
        timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def write_rationale(path: Path) -> None:
    lines = ["CIC-IDS-2018 anomaly experiment rationale", "=" * 42, ""]
    for title, body in RATIONALE:
        lines.extend([title, "-" * len(title), body, ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def discover_sources(data_folder: str) -> list[str]:
    sources = sorted(glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True))
    if not sources:
        raise FileNotFoundError(
            f"No original CIC-IDS-2018 CSV files found under '{data_folder}'."
        )
    return sources


def parse_timestamps(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce", dayfirst=True)


def clean_sample(sample: pd.DataFrame, log) -> pd.DataFrame:
    required = FLOW_SOURCE_COLUMNS + ["label", "timestamp", "source_day"]
    frame = sample.loc[:, required].copy()
    for column in FLOW_NUMERIC + ["dst_port"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(frame)
    frame.dropna(subset=FLOW_SOURCE_COLUMNS + ["label", "source_day"], inplace=True)
    frame = frame.loc[frame["dst_port"].between(0, 65535)].copy()
    frame["protocol"] = frame["protocol"].astype(str)
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["binary_label"] = (frame["label"] != "benign").astype(np.int8)
    frame["capture_timestamp"] = parse_timestamps(frame["timestamp"])
    log(
        f"Clean compact-flow sample: {len(frame):,} rows "
        f"({before - len(frame):,} removed for invalid features)."
    )
    return frame


def ordered_capture_days(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["source_day"].unique(), key=capture_date)


def capture_day_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    days = ordered_capture_days(frame)
    if len(days) < 10:
        raise ValueError(f"Expected 10 capture days, found {len(days)}.")
    train_days = days[:6]
    validation_days = days[6:8]
    test_days = days[8:]
    train = frame.loc[frame["source_day"].isin(train_days)].copy()
    validation = frame.loc[frame["source_day"].isin(validation_days)].copy()
    test = frame.loc[frame["source_day"].isin(test_days)].copy()
    return train, validation, test, {
        "train_days": train_days,
        "validation_days": validation_days,
        "test_days": test_days,
    }


def make_compact_preprocessor() -> ColumnTransformer:
    numeric = FLOW_NUMERIC + ["dst_port_log"]
    categorical = ["protocol", "dst_port_range", "dst_port_service"]
    return ColumnTransformer(
        [
            ("numeric", MinMaxScaler(clip=True), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_transform_compact(
    fit_frame: pd.DataFrame,
    frames: list[pd.DataFrame],
) -> tuple[list[np.ndarray], ColumnTransformer]:
    preprocessor = make_compact_preprocessor()
    fit_engineered = engineer_compact_flow(fit_frame)
    preprocessor.fit(fit_engineered)
    transformed = [
        np.asarray(preprocessor.transform(engineer_compact_flow(frame)), dtype=np.float32)
        for frame in frames
    ]
    return transformed, preprocessor


def threshold_from_benign(
    validation_scores: np.ndarray,
    validation_y: np.ndarray,
    target_fpr: float,
) -> float:
    if not 0 < target_fpr < 1:
        raise ValueError("--benign-fpr values must be between 0 and 1.")
    benign_scores = validation_scores[validation_y == 0]
    if not len(benign_scores):
        raise ValueError("Validation split has no benign rows.")
    return float(np.quantile(benign_scores, 1.0 - target_fpr))


def score_pca(model: PCA, values: np.ndarray) -> np.ndarray:
    projected = model.transform(values)
    reconstructed = model.inverse_transform(projected)
    return np.mean(np.square(values - reconstructed), axis=1)


def make_distance_scorer(
    benign_train: np.ndarray,
    robust: bool,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, np.ndarray]]:
    if robust:
        center = np.median(benign_train, axis=0)
        scale = np.median(np.abs(benign_train - center), axis=0) * 1.4826
    else:
        center = np.mean(benign_train, axis=0)
        scale = np.std(benign_train, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)

    def scorer(values: np.ndarray) -> np.ndarray:
        standardized = (values - center) / scale
        return np.mean(np.square(standardized), axis=1)

    return scorer, {"center": center, "scale": scale}


def score_kmeans(model: MiniBatchKMeans, values: np.ndarray) -> np.ndarray:
    distances = model.transform(values)
    return np.min(np.square(distances), axis=1)


def evaluate_anomaly_scores(
    model_id: str,
    score_validation: np.ndarray,
    score_test: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
    test_labels: np.ndarray,
    target_fprs: list[float],
    fit_seconds: float,
    fhe_note: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    for target_fpr in target_fprs:
        threshold = threshold_from_benign(score_validation, validation_y, target_fpr)
        metrics = classification_metrics(test_y, score_test, threshold)
        actual_fpr = float((score_test[test_y == 0] >= threshold).mean())
        result = {
            "protocol": "unknown_anomaly_capture_day",
            "model_id": model_id,
            "target_validation_benign_fpr": target_fpr,
            "threshold": threshold,
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_average_precision": metrics["average_precision"],
            "test_roc_auc": metrics["roc_auc"],
            "test_benign_fpr": actual_fpr,
            "fit_seconds": fit_seconds,
            "fhe_note": fhe_note,
        }
        results.append(result)
        breakdown.extend(
            per_label_detection(
                test_labels,
                score_test,
                threshold,
                "compact_flow",
                model_id,
                mode=f"unknown_anomaly_fpr_{target_fpr:g}",
            )
        )
    return results, breakdown


def run_unknown_anomaly_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
    log,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    benign_train = train.loc[train["binary_label"] == 0]
    (x_train_benign, x_validation, x_test), preprocessor = fit_transform_compact(
        benign_train,
        [benign_train, validation, test],
    )
    validation_y = validation["binary_label"].to_numpy(dtype=np.int8)
    test_y = test["binary_label"].to_numpy(dtype=np.int8)
    test_labels = test["label"].to_numpy(dtype=str)
    results: list[dict] = []
    breakdown: list[dict] = []

    if args.save_models:
        model_dir = run_dir / "models" / "unknown_anomaly"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, model_dir / "feature_transformer.joblib")

    for n_components in sorted(set(args.pca_components)):
        if n_components >= x_train_benign.shape[1]:
            log(
                f"Skipping PCA-{n_components}: input has only "
                f"{x_train_benign.shape[1]} dimensions."
            )
            continue
        started = time.perf_counter()
        model = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=SEED,
        )
        model.fit(x_train_benign)
        fit_seconds = time.perf_counter() - started
        score_validation = score_pca(model, x_validation)
        score_test = score_pca(model, x_test)
        model_results, model_breakdown = evaluate_anomaly_scores(
            f"pca_reconstruction_{n_components}",
            score_validation,
            score_test,
            validation_y,
            test_y,
            test_labels,
            list(args.benign_fpr),
            fit_seconds,
            "Linear projection/reconstruction plus squared error.",
        )
        results.extend(model_results)
        breakdown.extend(model_breakdown)
        log(
            f"PCA-{n_components}: best test F1="
            f"{max(row['test_f1'] for row in model_results):.4f}."
        )
        if args.save_models:
            joblib.dump(model, model_dir / f"pca_{n_components}.joblib")

    for robust in (False, True):
        model_id = "robust_benign_distance" if robust else "standard_benign_distance"
        started = time.perf_counter()
        scorer, parameters = make_distance_scorer(x_train_benign, robust=robust)
        fit_seconds = time.perf_counter() - started
        model_results, model_breakdown = evaluate_anomaly_scores(
            model_id,
            scorer(x_validation),
            scorer(x_test),
            validation_y,
            test_y,
            test_labels,
            list(args.benign_fpr),
            fit_seconds,
            "Subtract, scale, square, and sum; smallest candidate arithmetic circuit.",
        )
        results.extend(model_results)
        breakdown.extend(model_breakdown)
        log(
            f"{model_id}: best test F1="
            f"{max(row['test_f1'] for row in model_results):.4f}."
        )
        if args.save_models:
            joblib.dump(parameters, model_dir / f"{model_id}.joblib")

    for cluster_count in (1, 4):
        started = time.perf_counter()
        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            batch_size=8192,
            n_init=3,
            random_state=SEED,
        )
        model.fit(x_train_benign)
        fit_seconds = time.perf_counter() - started
        model_results, model_breakdown = evaluate_anomaly_scores(
            f"benign_kmeans_{cluster_count}",
            score_kmeans(model, x_validation),
            score_kmeans(model, x_test),
            validation_y,
            test_y,
            test_labels,
            list(args.benign_fpr),
            fit_seconds,
            (
                "One squared centroid distance."
                if cluster_count == 1
                else "Four squared distances plus encrypted minimum comparisons."
            ),
        )
        results.extend(model_results)
        breakdown.extend(model_breakdown)
        log(
            f"KMeans-{cluster_count}: best test F1="
            f"{max(row['test_f1'] for row in model_results):.4f}."
        )
        if args.save_models:
            joblib.dump(model, model_dir / f"kmeans_{cluster_count}.joblib")

    summary = {
        "input_dimensions": int(x_train_benign.shape[1]),
        "benign_train_rows": int(len(x_train_benign)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }
    return results, breakdown, summary


def build_known_family_split(
    frame: pd.DataFrame,
    block_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid = frame.dropna(subset=["capture_timestamp"]).copy()
    valid["time_block"] = valid["capture_timestamp"].dt.floor(f"{block_minutes}min")
    valid["block_id"] = (
        valid["source_day"].astype(str) + "|" + valid["time_block"].astype(str)
    )
    assignment = pd.Series(index=valid.index, dtype="object")
    excluded: dict[str, int] = {}
    included: dict[str, int] = {}

    for label, label_frame in valid.groupby("label", observed=True):
        blocks = (
            label_frame.groupby("block_id", observed=True)["capture_timestamp"]
            .min()
            .sort_values()
            .index.tolist()
        )
        if len(blocks) < 3:
            excluded[str(label)] = len(blocks)
            continue
        test_count = max(1, round(len(blocks) * 0.15))
        validation_count = max(1, round(len(blocks) * 0.15))
        if test_count + validation_count >= len(blocks):
            excluded[str(label)] = len(blocks)
            continue
        train_blocks = blocks[: -(test_count + validation_count)]
        validation_blocks = blocks[-(test_count + validation_count) : -test_count]
        test_blocks = blocks[-test_count:]
        assignment.loc[
            label_frame.index[label_frame["block_id"].isin(train_blocks)]
        ] = "train"
        assignment.loc[
            label_frame.index[label_frame["block_id"].isin(validation_blocks)]
        ] = "validation"
        assignment.loc[
            label_frame.index[label_frame["block_id"].isin(test_blocks)]
        ] = "test"
        included[str(label)] = len(blocks)

    train = valid.loc[assignment == "train"].copy()
    validation = valid.loc[assignment == "validation"].copy()
    test = valid.loc[assignment == "test"].copy()
    for split_name, split in (("train", train), ("validation", validation), ("test", test)):
        if split["binary_label"].nunique() != 2:
            raise ValueError(f"Known-family {split_name} split lacks one binary class.")
    return train, validation, test, {
        "block_minutes": block_minutes,
        "included_label_block_counts": included,
        "excluded_label_block_counts": excluded,
        "rows_without_parseable_timestamp": int(frame["capture_timestamp"].isna().sum()),
    }


def choose_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    index = int(np.nanargmax(f1_values))
    return float(thresholds[index]), float(f1_values[index])


def run_known_family_lr(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
    log,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    train, validation, test, split_summary = build_known_family_split(
        frame,
        args.time_block_minutes,
    )
    (x_train, x_validation, x_test), preprocessor = fit_transform_compact(
        train,
        [train, validation, test],
    )
    y_train = train["binary_label"].to_numpy(dtype=np.int8)
    y_validation = validation["binary_label"].to_numpy(dtype=np.int8)
    y_test = test["binary_label"].to_numpy(dtype=np.int8)
    test_labels = test["label"].to_numpy(dtype=str)
    results: list[dict] = []
    breakdown: list[dict] = []

    if args.save_models:
        model_dir = run_dir / "models" / "known_family"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, model_dir / "feature_transformer.joblib")

    for c_value in (1.0, 10.0):
        for quantized in (False, True):
            model_id = f"known_lr_c{c_value:g}"
            mode = "concrete_quantized_5bit" if quantized else "float_plaintext"
            started = time.perf_counter()
            if quantized:
                from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression

                model = ConcreteLogisticRegression(
                    n_bits=5,
                    C=c_value,
                    class_weight="balanced",
                    max_iter=500,
                    solver="lbfgs",
                    n_jobs=args.jobs,
                    random_state=SEED,
                )
            else:
                model = LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=500,
                    solver="lbfgs",
                    n_jobs=args.jobs,
                    random_state=SEED,
                )
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - started
            if quantized:
                validation_scores = np.asarray(
                    model.predict_proba(x_validation, fhe="disable")
                )[:, 1]
                test_scores = np.asarray(model.predict_proba(x_test, fhe="disable"))[:, 1]
            else:
                validation_scores = positive_scores(model, x_validation)
                test_scores = positive_scores(model, x_test)
            threshold, validation_f1 = choose_f1_threshold(y_validation, validation_scores)
            metrics = classification_metrics(y_test, test_scores, threshold)
            result = {
                "protocol": "known_family_time_block",
                "model_id": model_id,
                "mode": mode,
                "n_bits": 5 if quantized else None,
                "threshold": threshold,
                "validation_f1": validation_f1,
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_average_precision": metrics["average_precision"],
                "test_roc_auc": metrics["roc_auc"],
                "fit_seconds": fit_seconds,
            }
            results.append(result)
            breakdown.extend(
                per_label_detection(
                    test_labels,
                    test_scores,
                    threshold,
                    "compact_flow",
                    model_id,
                    mode=mode,
                    n_bits=5 if quantized else None,
                )
            )
            log(
                f"{model_id}/{mode}: test F1={metrics['f1']:.4f}, "
                f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}."
            )
            if args.save_models:
                joblib.dump(model, model_dir / f"{model_id}_{mode}.joblib")

    split_summary.update(
        {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "input_dimensions": x_train.shape[1],
        }
    )
    return results, breakdown, split_summary


def write_outputs(
    run_dir: Path,
    unknown_results: list[dict],
    known_results: list[dict],
    breakdown: list[dict],
) -> None:
    all_results = unknown_results + known_results
    pd.DataFrame(all_results).to_csv(run_dir / "results.csv", index=False)
    write_json(run_dir / "results.json", all_results)
    pd.DataFrame(breakdown).to_csv(run_dir / "label_breakdown.csv", index=False)
    write_json(run_dir / "label_breakdown.json", breakdown)


def format_result_table(frame: pd.DataFrame, columns: list[str]) -> str:
    table = frame.loc[:, columns].copy()
    for column in (
        "target_validation_benign_fpr",
        "test_benign_fpr",
        "test_f1",
        "test_precision",
        "test_recall",
        "threshold",
        "fit_seconds",
    ):
        if column in table:
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    return table.to_string(index=False)


def write_report(
    run_dir: Path,
    args: argparse.Namespace,
    dataset_summary: dict[str, Any],
    unknown_results: list[dict],
    known_results: list[dict],
    breakdown: list[dict],
) -> None:
    unknown = pd.DataFrame(unknown_results)
    known = pd.DataFrame(known_results)
    lines = [
        "CIC-IDS-2018 FHE-oriented anomaly experiment",
        "=" * 47,
        f"Run directory: {run_dir}",
        f"Generated: {dt.datetime.now().astimezone().isoformat()}",
        f"Arguments: {json.dumps(vars(args), sort_keys=True)}",
        "",
        "Dataset and splits",
        "------------------",
    ]
    lines.extend(f"{key}: {value}" for key, value in dataset_summary.items())

    lines.extend(["", "1. Unknown-anomaly capture-day results", "--------------------------------------"])
    if unknown.empty:
        lines.append("No unknown-anomaly candidate completed.")
    else:
        ranked = unknown.sort_values(
            ["test_f1", "test_benign_fpr"],
            ascending=[False, True],
        )
        lines.append(
            format_result_table(
                ranked,
                [
                    "model_id",
                    "target_validation_benign_fpr",
                    "test_benign_fpr",
                    "test_f1",
                    "test_precision",
                    "test_recall",
                    "fit_seconds",
                    "fhe_note",
                ],
            )
        )
        best = ranked.iloc[0]
        lines.append(
            f"Best unknown-anomaly candidate: {best['model_id']} at validation "
            f"FPR target {best['target_validation_benign_fpr']:.1%}; "
            f"test F1={best['test_f1']:.4f}, test benign FPR={best['test_benign_fpr']:.4f}."
        )

    lines.extend(["", "2. Known-family time-block LR results", "-------------------------------------"])
    if known.empty:
        lines.append("No known-family LR candidate completed.")
    else:
        ranked_known = known.sort_values("test_f1", ascending=False)
        lines.append(
            format_result_table(
                ranked_known,
                [
                    "model_id",
                    "mode",
                    "n_bits",
                    "test_f1",
                    "test_precision",
                    "test_recall",
                    "threshold",
                    "fit_seconds",
                ],
            )
        )

    lines.extend(["", "3. Per-label detection for best candidates", "------------------------------------------"])
    breakdown_frame = pd.DataFrame(breakdown)
    selections: list[tuple[str, str]] = []
    if not unknown.empty:
        best = unknown.sort_values("test_f1", ascending=False).iloc[0]
        selections.append(
            (
                best["model_id"],
                f"unknown_anomaly_fpr_{best['target_validation_benign_fpr']:g}",
            )
        )
    if not known.empty:
        best_known = known.sort_values("test_f1", ascending=False).iloc[0]
        selections.append((best_known["model_id"], best_known["mode"]))
    for model_id, mode in selections:
        selected = breakdown_frame[
            (breakdown_frame["model_id"] == model_id)
            & (breakdown_frame["mode"] == mode)
        ][
            [
                "model_id",
                "mode",
                "original_label",
                "support",
                "predicted_attack_rate",
                "metric_name",
            ]
        ].copy()
        selected["predicted_attack_rate"] = selected["predicted_attack_rate"].map(
            lambda value: f"{value:.4f}"
        )
        lines.append(selected.to_string(index=False))

    lines.extend(["", "4. FHE decision", "---------------"])
    passing_unknown = (
        unknown.loc[
            (unknown["test_f1"] >= args.target_f1)
            & (
                unknown["test_benign_fpr"]
                <= unknown["target_validation_benign_fpr"] * 1.5
            )
        ]
        if not unknown.empty
        else pd.DataFrame()
    )
    passing_known = (
        known.loc[known["test_f1"] >= args.target_f1]
        if not known.empty
        else pd.DataFrame()
    )
    if passing_unknown.empty and passing_known.empty:
        lines.append(
            "No candidate passes the F1/FPR gate. Do not compile an FHE circuit from this run."
        )
    else:
        if not passing_unknown.empty:
            lines.append(
                "Unknown-anomaly candidates eligible for a custom FHE prototype: "
                + ", ".join(passing_unknown["model_id"].unique())
            )
        if not passing_known.empty:
            lines.append(
                "Known-family LR candidates eligible for Concrete ML FHE benchmarking: "
                + ", ".join(passing_known["model_id"].unique())
            )

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
    log(f"New isolated anomaly run: {run_dir}")
    log("Existing supervised reports and model artifacts will not be modified.")

    unknown_results: list[dict] = []
    known_results: list[dict] = []
    breakdown: list[dict] = []
    try:
        sources = discover_sources(args.data_folder)
        needed_columns = sorted(set(FLOW_SOURCE_COLUMNS + ["label", "timestamp"]))
        sample = load_streaming_sample(
            sources,
            needed_columns,
            args.sample_rows,
            args.chunk_rows,
            log,
        )
        frame = clean_sample(sample, log)
        train, validation, test, day_summary = capture_day_split(frame)
        unknown_results, unknown_breakdown, unknown_summary = run_unknown_anomaly_models(
            train,
            validation,
            test,
            args,
            run_dir,
            log,
        )
        breakdown.extend(unknown_breakdown)
        write_outputs(run_dir, unknown_results, known_results, breakdown)

        known_results, known_breakdown, known_summary = run_known_family_lr(
            frame,
            args,
            run_dir,
            log,
        )
        breakdown.extend(known_breakdown)
        write_outputs(run_dir, unknown_results, known_results, breakdown)

        dataset_summary = {
            "sample_rows_before_cleaning": len(sample),
            "clean_rows": len(frame),
            "benign_rows": int((frame["binary_label"] == 0).sum()),
            "attack_rows": int((frame["binary_label"] == 1).sum()),
            "unknown_anomaly_day_split": day_summary,
            "unknown_anomaly": unknown_summary,
            "known_family_time_blocks": known_summary,
        }
        write_report(
            run_dir,
            args,
            dataset_summary,
            unknown_results,
            known_results,
            breakdown,
        )
        write_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "finished": dt.datetime.now().astimezone().isoformat(),
                "unknown_anomaly_results": len(unknown_results),
                "known_family_results": len(known_results),
            },
        )
        log(f"Experiment complete. Read {run_dir / 'report.txt'}")
        return 0
    except Exception as exc:
        write_outputs(run_dir, unknown_results, known_results, breakdown)
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
