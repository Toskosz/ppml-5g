"""Score the passing known-family logistic regressions on the unsampled test flows.

The source run kept each source row with probability 2,000,000 / 16,233,002
and then held out 28 thirty-minute blocks. This script scores the flows that
coin flip left out of those same 28 blocks. Training blocks are not scored.

Two executions are reported for each model:

- Plaintext: a float logistic regression, and the 5-bit Concrete ML model in
  the clear. The original estimators were not saved, so both are refit on the
  source run's training blocks with the same settings and the saved
  preprocessor. The 5-bit decision threshold stays the one frozen in that run.
- FHE: simulation of the saved circuit. There is no new parameter search and
  no real encrypt / inference / decrypt.

Run from this directory on the machine that has the CIC-IDS-2018 CSVs,
Concrete ML, and the saved circuit zips. Do not run it in the agent workspace.

    python score_known_lr_unsampled_test.py
    python score_known_lr_unsampled_test.py --only known_lr_c1

FHE simulation of the unsampled test-block flows is on the order of one hour
per model. Each invocation writes
``qualified_fhe_runs/<timestamp>_unsampled_test/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from anomaly_model_experiments import (
    build_known_family_split,
    choose_f1_threshold,
    clean_sample,
    discover_sources,
    make_logger,
)
from data_load import clean_col_names, resolve_usecols
from gpu_utils import release_memory
from plaintext_model_experiments import (
    FLOW_SOURCE_COLUMNS,
    SEED,
    count_csv_data_rows,
    engineer_compact_flow,
    load_streaming_sample,
    write_json,
)

SOURCE_RUN = Path("qualified_fhe_runs/20261031_170906_440932")
SAMPLE_ROWS = 2_000_000
SOURCE_ROWS = 16_233_002
REPLAY_LOADED_ROWS = 1_998_881
REPLAY_CLEAN_ROWS = 1_987_045
REPLAY_TEST_ROWS = 159_917
REPLAY_TEST_BLOCKS = 28
SCOPE = "unsampled_test_blocks"
MODES = ("float_plaintext", "concrete_clear_5bit", "fhe_simulation")
MODELS = {
    "known_lr_c1": {
        "c_value": 1.0,
        "directory": "fhe/known_lr_c1_q5/cuda",
        "threshold": 0.667279,
        "sample_clear_f1": 0.976132,
    },
    "known_lr_c10": {
        "c_value": 10.0,
        "directory": "fhe/known_lr_c10_q5/cuda",
        "threshold": 0.821242,
        "sample_clear_f1": 0.974651,
    },
}


class Counts:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.support: dict[str, int] = {}
        self.detected: dict[str, int] = {}

    def update(self, labels: np.ndarray, attack: np.ndarray, predicted: np.ndarray) -> None:
        attack = attack.astype(bool, copy=False)
        predicted = predicted.astype(bool, copy=False)
        self.tp += int(np.count_nonzero(predicted & attack))
        self.fp += int(np.count_nonzero(predicted & ~attack))
        self.fn += int(np.count_nonzero(~predicted & attack))
        self.tn += int(np.count_nonzero(~predicted & ~attack))
        for label in np.unique(labels):
            mask = labels == label
            name = str(label)
            self.support[name] = self.support.get(name, 0) + int(mask.sum())
            self.detected[name] = self.detected.get(name, 0) + int(
                np.count_nonzero(predicted[mask])
            )

    def metrics(self) -> dict[str, float]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        benign = self.fp + self.tn
        return {
            "rows": self.tp + self.fp + self.tn + self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "benign_fpr": self.fp / benign if benign else 0.0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score the saved known-family logistic regressions on the "
            "unsampled flows in the original 28 test blocks, in plaintext "
            "and in FHE simulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-folder", default="CIC-IDS-2018")
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument("--time-block-minutes", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--only",
        nargs="+",
        choices=tuple(MODELS),
        default=None,
    )
    parser.add_argument("--output-root", default="qualified_fhe_runs")
    return parser.parse_args()


def selected_models(args: argparse.Namespace) -> tuple[str, ...]:
    if args.only:
        return tuple(args.only)
    return tuple(MODELS)


def create_run_dir(output_root: str) -> Path:
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / f"{stamp}_unsampled_test"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def require_circuit(model_dir: Path) -> None:
    missing = [
        name
        for name in ("client.zip", "server.zip", "fhe_device.txt")
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Saved circuit {model_dir} is missing {', '.join(missing)}."
        )


def threshold_for(model_dir: Path, fallback: float, log) -> float:
    meta_path = model_dir / "run_meta.json"
    if not meta_path.is_file():
        log(f"No run_meta.json in {model_dir}. Using recorded threshold {fallback:.6f}.")
        return fallback
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "selected_threshold" not in meta:
        log(f"{meta_path} has no selected_threshold. Using {fallback:.6f}.")
        return fallback
    threshold = float(meta["selected_threshold"])
    log(f"Loaded frozen 5-bit threshold {threshold:.6f} from {meta_path}.")
    return threshold


def load_simulated_server(model_dir: Path, log):
    """Load the saved MLIR with Concrete's simulation backend.

    Crypto parameters stay the ones already stored in the circuit.
    """
    import jsonpickle
    from concrete.fhe import Server
    from concrete.fhe.compilation.composition import CompositionClause, CompositionRule
    from concrete.fhe.compilation.configuration import Configuration
    from concrete.fhe.compilation.specs import ClientSpecs

    server_zip = model_dir / "server.zip"
    log(f"Loading saved circuit from {server_zip} in simulation mode.")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary:
        unpacked = Path(temporary)
        shutil.unpack_archive(server_zip, unpacked, "zip")
        mlir = (unpacked / "circuit.mlir").read_text(encoding="utf-8")
        configuration = Configuration().fork(
            **jsonpickle.loads((unpacked / "configuration.json").read_text(encoding="utf-8"))
        )
        configuration = configuration.fork(use_gpu=False)
        composition_rules = None
        rules_path = unpacked / "composition_rules.json"
        if rules_path.is_file():
            raw_rules = json.loads(rules_path.read_text(encoding="utf-8"))
            if raw_rules:
                composition_rules = [
                    CompositionRule(
                        CompositionClause(rule[0][0], rule[0][1]),
                        CompositionClause(rule[1][0], rule[1][1]),
                    )
                    for rule in raw_rules
                ]
        tfhers_specs = None
        specs_path = unpacked / "client.specs.json"
        if specs_path.is_file():
            tfhers_specs = ClientSpecs.deserialize(specs_path.read_bytes()).tfhers_specs
        server = Server.create(
            mlir,
            configuration,
            True,
            composition_rules=composition_rules,
            tfhers_specs=tfhers_specs,
        )
    log(f"Simulation backend ready in {time.perf_counter() - started:.1f}s.")
    return server


def simulated_positive_scores(processing_model, sim_client, sim_server, features: np.ndarray) -> np.ndarray:
    quantized = np.asarray(processing_model.quantize_input(features))
    quantized_rows: list[np.ndarray] = []
    for row_index in range(len(quantized)):
        encrypted = sim_client.simulate_encrypt(quantized[row_index : row_index + 1])
        executed = sim_server.run(encrypted)
        clear_quantized = sim_client.simulate_decrypt(executed)
        row = np.asarray(clear_quantized)
        if row.ndim == 2:
            if row.shape[0] != 1:
                raise RuntimeError(f"Simulated output shape {row.shape} is not one record.")
            row = row[0]
        quantized_rows.append(np.reshape(row, (-1,)))
    dequantized = np.asarray(processing_model.dequantize_output(np.asarray(quantized_rows)))
    probabilities = np.asarray(processing_model.post_processing(dequantized))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise RuntimeError(f"Expected probabilities of shape (rows, 2), got {probabilities.shape}.")
    return probabilities[:, 1].astype(np.float64, copy=False)


def transform_frame(frame: pd.DataFrame, preprocessor) -> np.ndarray:
    return np.asarray(
        preprocessor.transform(engineer_compact_flow(frame)),
        dtype=np.float32,
    )


def replay_training_split(args: argparse.Namespace, preprocessor, log) -> dict[str, Any]:
    sources = discover_sources(args.data_folder)
    log(f"Replay sources: {len(sources)} files.")
    needed = sorted(set(FLOW_SOURCE_COLUMNS + ["label", "timestamp"]))
    sample = load_streaming_sample(sources, needed, SAMPLE_ROWS, args.chunk_rows, log)
    if len(sample) != REPLAY_LOADED_ROWS:
        raise RuntimeError(
            f"Seeded sample has {len(sample):,} rows. The source run loaded "
            f"{REPLAY_LOADED_ROWS:,}."
        )
    frame = clean_sample(sample, log)
    del sample
    release_memory()
    if len(frame) != REPLAY_CLEAN_ROWS:
        raise RuntimeError(
            f"Cleaned sample has {len(frame):,} rows. The source run cleaned "
            f"{REPLAY_CLEAN_ROWS:,}."
        )
    train, validation, test, summary = build_known_family_split(frame, args.time_block_minutes)
    del frame
    release_memory()
    test_blocks = set(test["block_id"].astype(str))
    included_labels = set(test["label"].astype(str))
    if len(test) != REPLAY_TEST_ROWS or len(test_blocks) != REPLAY_TEST_BLOCKS:
        raise RuntimeError(
            f"Replay produced {len(test):,} test rows in {len(test_blocks)} blocks. "
            f"The source run had {REPLAY_TEST_ROWS:,} rows in {REPLAY_TEST_BLOCKS} blocks."
        )
    if (test["label"] == "infilteration").any():
        raise RuntimeError("Infiltration rows entered the replayed test split.")
    log(f"Known-family replay: {summary}")
    features = {
        "train": transform_frame(train, preprocessor),
        "validation": transform_frame(validation, preprocessor),
        "test": transform_frame(test, preprocessor),
    }
    labels = {
        "train": train["binary_label"].to_numpy(dtype=np.int8),
        "validation": validation["binary_label"].to_numpy(dtype=np.int8),
        "test": test["binary_label"].to_numpy(dtype=np.int8),
    }
    del train, validation, test
    release_memory()
    return {
        "features": features,
        "labels": labels,
        "test_blocks": test_blocks,
        "included_labels": included_labels,
        "summary": summary,
    }


def fit_plaintext_models(model_id: str, replay: dict[str, Any], args, threshold: float, log) -> dict[str, Any]:
    from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression

    spec = MODELS[model_id]
    x_train = replay["features"]["train"]
    y_train = replay["labels"]["train"]
    log(f"Fitting float and 5-bit {model_id} on {len(x_train):,} replayed training rows.")
    float_model = LogisticRegression(
        C=spec["c_value"],
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        n_jobs=args.jobs,
        random_state=SEED,
    )
    concrete_model = ConcreteLogisticRegression(
        n_bits=5,
        C=spec["c_value"],
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        n_jobs=args.jobs,
        random_state=SEED,
    )
    started = time.perf_counter()
    float_model.fit(x_train, y_train)
    concrete_model.fit(x_train, y_train)
    log(f"Fit finished in {time.perf_counter() - started:.1f}s.")
    float_validation = float_model.predict_proba(replay["features"]["validation"])[:, 1]
    float_threshold, float_validation_f1 = choose_f1_threshold(
        replay["labels"]["validation"],
        float_validation,
    )
    clear_test = np.asarray(
        concrete_model.predict_proba(replay["features"]["test"], fhe="disable")
    )[:, 1]
    predicted = clear_test >= threshold
    attack = replay["labels"]["test"] == 1
    tp = int(np.count_nonzero(predicted & attack))
    fp = int(np.count_nonzero(predicted & ~attack))
    fn = int(np.count_nonzero(~predicted & attack))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    replay_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delta = replay_f1 - spec["sample_clear_f1"]
    log(
        f"{model_id} replayed 5-bit clear F1 on the original test sample is "
        f"{replay_f1:.4f} (source run {spec['sample_clear_f1']:.4f}, delta {delta:+.4f})."
    )
    if abs(delta) > 0.02:
        raise RuntimeError(
            f"{model_id} replay F1 delta {delta:+.4f} is too far from the saved model."
        )
    return {
        "float_model": float_model,
        "concrete_model": concrete_model,
        "float_threshold": float(float_threshold),
        "float_validation_f1": float(float_validation_f1),
        "replay_clear_f1": replay_f1,
        "replay_clear_f1_delta": delta,
    }


def attach_block_id(frame: pd.DataFrame, block_minutes: int) -> pd.DataFrame:
    time_block = frame["capture_timestamp"].dt.floor(f"{block_minutes}min")
    frame["block_id"] = frame["source_day"].astype(str) + "|" + time_block.astype(str)
    return frame


class UnsampledTestReader:
    """Replay the seeded coin flip and yield unsampled rows in the test blocks.

    ``load_streaming_sample`` draws one uniform number per source row, in file
    order, and keeps the row when the draw is below ``SAMPLE_ROWS / SOURCE_ROWS``.
    Chunk boundaries do not change that sequence. This reader uses a fresh
    generator with the same seed, so a row kept here is a row the source run
    skipped.
    """

    def __init__(self, args: argparse.Namespace, replay: dict[str, Any], log) -> None:
        self.args = args
        self.test_blocks = replay["test_blocks"]
        self.included_labels = replay["included_labels"]
        self.log = log
        self.sources = discover_sources(args.data_folder)
        self.needed = sorted(set(FLOW_SOURCE_COLUMNS + ["label", "timestamp"]))
        total_rows = sum(count_csv_data_rows(path) for path in self.sources)
        if total_rows != SOURCE_ROWS:
            raise RuntimeError(
                f"Source CSVs contain {total_rows:,} rows. The source run counted "
                f"{SOURCE_ROWS:,}."
            )
        self.probability = min(1.0, SAMPLE_ROWS / total_rows)
        self.sampled_rows = 0
        self.candidate_rows = 0

    def __iter__(self):
        self.sampled_rows = 0
        self.candidate_rows = 0
        rng = np.random.default_rng(SEED)
        for file_index, path in enumerate(self.sources, start=1):
            usecols = resolve_usecols(path, self.needed)
            self.log(f"Reading {file_index}/{len(self.sources)} {path}")
            source_day = Path(path).name
            for chunk in pd.read_csv(
                path,
                usecols=usecols,
                chunksize=self.args.chunk_rows,
                low_memory=False,
            ):
                chunk = clean_col_names(chunk)
                draws = rng.random(len(chunk))
                self.sampled_rows += int(np.count_nonzero(draws < self.probability))
                positions = np.flatnonzero(draws >= self.probability)
                if len(positions) == 0:
                    continue
                selected = chunk.iloc[positions].copy()
                selected["source_day"] = source_day
                cleaned = clean_sample(selected, lambda _message: None)
                if cleaned.empty:
                    continue
                labeled = attach_block_id(cleaned, self.args.time_block_minutes)
                mask = labeled["block_id"].isin(self.test_blocks) & labeled["label"].isin(
                    self.included_labels
                )
                held = labeled.loc[mask]
                if held.empty:
                    continue
                self.candidate_rows += len(held)
                yield held

    def assert_replay(self) -> None:
        if self.sampled_rows != REPLAY_LOADED_ROWS:
            raise RuntimeError(
                f"The replayed coin flip kept {self.sampled_rows:,} rows. "
                f"The source run kept {REPLAY_LOADED_ROWS:,}."
            )
        if self.candidate_rows < 1:
            raise RuntimeError("No unsampled rows fell in the original test blocks.")


def score_unsampled_test(
    args: argparse.Namespace,
    preprocessor,
    replay: dict[str, Any],
    fitted_models: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
    log,
) -> list[dict[str, Any]]:
    from concrete.fhe import Client
    from concrete.ml.deployment import FHEModelClient

    states: dict[str, dict[str, Any]] = {}
    books: dict[str, dict[str, Counts]] = {}
    for model_id, fitted in fitted_models.items():
        model_dir = args.source_run / MODELS[model_id]["directory"]
        processing = FHEModelClient(str(model_dir))
        server = load_simulated_server(model_dir, log)
        states[model_id] = {
            "model_dir": model_dir,
            "fitted": fitted,
            "threshold": thresholds[model_id],
            "processing": processing,
            "server": server,
            "sim_client": Client(server.client_specs, is_simulated=True),
        }
        books[model_id] = {mode: Counts() for mode in MODES}

    started = time.perf_counter()
    seen = 0
    next_log = 50_000
    reader = UnsampledTestReader(args, replay, log)
    for labeled in reader:
        features = transform_frame(labeled, preprocessor)
        attack = labeled["binary_label"].to_numpy(dtype=np.int8) == 1
        labels = labeled["label"].to_numpy(dtype=str)
        prepared = {
            model_id: {
                "float_plaintext": state["fitted"]["float_model"].predict_proba(features)[:, 1],
                "concrete_clear_5bit": np.asarray(
                    state["fitted"]["concrete_model"].predict_proba(features, fhe="disable")
                )[:, 1],
            }
            for model_id, state in states.items()
        }
        for start in range(0, len(features), 4_096):
            stop = min(start + 4_096, len(features))
            for model_id, state in states.items():
                mode_scores = {
                    mode: scores[start:stop] for mode, scores in prepared[model_id].items()
                }
                mode_scores["fhe_simulation"] = simulated_positive_scores(
                    state["processing"].model,
                    state["sim_client"],
                    state["server"],
                    features[start:stop],
                )
                mode_thresholds = {
                    "float_plaintext": state["fitted"]["float_threshold"],
                    "concrete_clear_5bit": state["threshold"],
                    "fhe_simulation": state["threshold"],
                }
                for mode, scores in mode_scores.items():
                    predicted = np.asarray(scores) >= mode_thresholds[mode]
                    books[model_id][mode].update(labels[start:stop], attack[start:stop], predicted)
            seen += stop - start
            if seen >= next_log:
                elapsed = time.perf_counter() - started
                log(
                    f"Scored {seen:,} unsampled test rows in {elapsed / 60:.1f} min "
                    f"({seen / elapsed:.1f} rows/s)."
                )
                next_log = seen + 50_000
    reader.assert_replay()
    elapsed = time.perf_counter() - started
    log(
        f"Finished {seen:,} unsampled test rows in {elapsed / 3600:.2f} h. "
        f"The coin flip kept {reader.sampled_rows:,} source rows."
    )
    results = []
    for model_id, state in states.items():
        fitted = state["fitted"]
        results.append(
            {
                "model_id": model_id,
                "model_dir": str(state["model_dir"]),
                "quantized_threshold": state["threshold"],
                "float_threshold": fitted["float_threshold"],
                "float_validation_f1": fitted["float_validation_f1"],
                "replay_clear_f1": fitted["replay_clear_f1"],
                "replay_clear_f1_delta": fitted["replay_clear_f1_delta"],
                "pass_seconds": elapsed,
                "scopes": {mode: {SCOPE: books[model_id][mode].metrics()} for mode in MODES},
                "labels": {mode: {SCOPE: books[model_id][mode]} for mode in MODES},
            }
        )
    return results


def format_counts(counts: Counts) -> str:
    rows = []
    for label in sorted(counts.support):
        support = counts.support[label]
        rate = counts.detected[label] / support if support else 0.0
        metric = "false_positive_rate" if label == "benign" else "detection_recall"
        rows.append(f"{label:>24} {support:10d} {rate:8.4f}  {metric}")
    return "\n".join(rows) if rows else "No rows."


def write_report(run_dir: Path, args: argparse.Namespace, replay: dict[str, Any], results: list[dict]) -> None:
    lines = [
        "CIC-IDS-2018 unsampled test-block rescore",
        "=" * 44,
        f"Run directory: {run_dir}",
        f"Generated: {dt.datetime.now().astimezone().isoformat()}",
        f"Source run: {args.source_run}",
        f"Sampling probability: {replay['sampling_probability']:.6f}",
        f"Replayed sampled source rows: {replay['sampled_source_rows']}",
        f"Unsampled cleaned test-block rows: {replay['unsampled_test_rows']}",
        "",
        "Only flows skipped by the source-run coin flip, and only when they",
        "fall in the original 28 held-out 30-minute blocks, are scored.",
        "Training blocks and the 159,917 sampled test rows are left out.",
        "Float and 5-bit clear models are refit on the source training blocks.",
        "FHE scores come from simulation of the saved circuit.",
        "",
        "Included labels: " + ", ".join(sorted(replay["included_labels"])),
        "",
    ]
    for result in results:
        lines.extend(["", result["model_id"], "-" * len(result["model_id"])])
        lines.append(f"quantized_threshold: {result['quantized_threshold']:.6f}")
        lines.append(f"float_threshold: {result['float_threshold']:.6f}")
        lines.append(f"float_validation_f1: {result['float_validation_f1']:.6f}")
        lines.append(f"replay_clear_f1: {result['replay_clear_f1']:.6f}")
        lines.append(f"replay_clear_f1_delta: {result['replay_clear_f1_delta']:+.6f}")
        lines.append(f"pass_seconds: {result['pass_seconds']:.1f}")
        lines.extend(["", SCOPE])
        for mode in MODES:
            metrics = result["scopes"][mode][SCOPE]
            lines.append(
                f"  {mode}: rows={metrics['rows']} f1={metrics['f1']:.4f} "
                f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
                f"benign_fpr={metrics['benign_fpr']:.4f}"
            )
        lines.extend(["", f"{SCOPE} per-label"])
        for mode in MODES:
            lines.extend([f"  {mode}", format_counts(result["labels"][mode][SCOPE])])
    lines.extend(
        [
            "",
            "Claim",
            "-----",
            "unsampled_test_blocks is the rest of the original held-out blocks.",
            "The published F1 remains the score on the 159,917 sampled test rows.",
            "The 100-record latency measurement from the source run is unchanged.",
            "",
        ]
    )
    (run_dir / "report.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    exported = {key: value for key, value in result.items() if key != "labels"}
    exported["labels"] = {
        mode: {
            scope: {
                "support": counts.support,
                "detected": counts.detected,
            }
            for scope, counts in scopes.items()
        }
        for mode, scopes in result["labels"].items()
    }
    return exported


def count_unsampled_test(args: argparse.Namespace, replay: dict[str, Any], log) -> UnsampledTestReader:
    log("Checking the coin flip before FHE simulation.")
    reader = UnsampledTestReader(args, replay, log)
    for _frame in reader:
        pass
    reader.assert_replay()
    log(
        f"Coin flip kept {reader.sampled_rows:,} source rows. "
        f"Unsampled cleaned test-block rows: {reader.candidate_rows:,}."
    )
    return reader


def main() -> int:
    args = parse_args()
    if args.chunk_rows < 1 or args.time_block_minutes < 1 or args.jobs < 1:
        raise SystemExit("Chunk size, block minutes, and jobs must be at least 1.")
    preprocessor_path = args.source_run / "models" / "known_family_feature_transformer.joblib"
    if not preprocessor_path.is_file():
        raise SystemExit(f"Missing preprocessor: {preprocessor_path}")
    for model_id in selected_models(args):
        require_circuit(args.source_run / MODELS[model_id]["directory"])

    run_dir = create_run_dir(args.output_root)
    log = make_logger(run_dir)
    write_json(
        run_dir / "arguments.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    log(f"Unsampled test-block rescore: {run_dir}")
    log("Plaintext models are refit on the source training blocks. FHE uses the saved circuits.")
    results: list[dict[str, Any]] = []
    try:
        import joblib

        preprocessor = joblib.load(preprocessor_path)
        replay = replay_training_split(args, preprocessor, log)
        write_json(run_dir / "test_blocks.json", sorted(replay["test_blocks"]))
        write_json(run_dir / "included_labels.json", sorted(replay["included_labels"]))
        fitted_models: dict[str, dict[str, Any]] = {}
        thresholds: dict[str, float] = {}
        for model_id in selected_models(args):
            model_dir = args.source_run / MODELS[model_id]["directory"]
            thresholds[model_id] = threshold_for(model_dir, MODELS[model_id]["threshold"], log)
            fitted_models[model_id] = fit_plaintext_models(
                model_id,
                replay,
                args,
                thresholds[model_id],
                log,
            )
        replay["features"] = {}
        replay["labels"] = {}
        release_memory()
        census = count_unsampled_test(args, replay, log)
        replay["sampling_probability"] = census.probability
        replay["sampled_source_rows"] = census.sampled_rows
        replay["unsampled_test_rows"] = census.candidate_rows
        write_json(
            run_dir / "unsampled_test_census.json",
            {
                "sampling_probability": census.probability,
                "sampled_source_rows": census.sampled_rows,
                "unsampled_test_rows": census.candidate_rows,
            },
        )
        results = score_unsampled_test(
            args,
            preprocessor,
            replay,
            fitted_models,
            thresholds,
            log,
        )
        write_json(run_dir / "results.json", [public_result(item) for item in results])
        write_report(run_dir, args, replay, results)
        release_memory()
        log(f"Rescore complete. Read {run_dir / 'report.txt'}")
        return 0
    except Exception as exc:
        if results:
            write_json(run_dir / "results.json", [public_result(item) for item in results])
        log(f"Rescore failed: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
