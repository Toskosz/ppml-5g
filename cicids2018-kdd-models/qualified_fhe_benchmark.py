"""Compile and time every CIC-IDS-2018 model that passed the quality gates.

The corrected run ``anomaly_experiment_runs/20261031_004057_494668`` qualified
four models:

- ``standard_benign_distance`` at a 1% validation benign false-positive rate.
- ``pca_reconstruction_32`` at that same operating point.
- ``known_lr_c1`` and ``known_lr_c10`` in Concrete ML 5-bit clear mode.

Distance and PCA are compiled as custom arithmetic circuits. The logistic
regressions use Concrete ML's estimator. Each successful compile is timed on
100 real encrypt / inference / decrypt records. One model failing does not
skip the remaining models, and the report is rewritten after each model.

Run from this directory on the machine that has the CIC-IDS-2018 CSVs and
Concrete ML. Do not run it in the agent workspace.

    python qualified_fhe_benchmark.py
    python qualified_fhe_benchmark.py --device cpu
    python qualified_fhe_benchmark.py --only standard_benign_distance

Every invocation writes a new directory under ``qualified_fhe_runs/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from anomaly_model_experiments import (
    SCOPED_UNKNOWN_ATTACK_LABELS,
    build_known_family_split,
    capture_day_split,
    choose_f1_threshold,
    clean_sample,
    discover_sources,
    fit_transform_compact,
    make_distance_scorer,
    make_logger,
    score_pca,
    threshold_from_benign,
)
from plaintext_model_experiments import (
    FLOW_SOURCE_COLUMNS,
    SEED,
    classification_metrics,
    create_run_dir,
    load_streaming_sample,
    per_label_detection,
    write_json,
)


QUALIFIED_MODELS = (
    "standard_benign_distance",
    "pca_reconstruction_32",
    "known_lr_c1",
    "known_lr_c10",
)
ANOMALY_MODELS = ("standard_benign_distance", "pca_reconstruction_32")
LR_C_VALUES = {"known_lr_c1": 1.0, "known_lr_c10": 10.0}
REFERENCES = {
    "standard_benign_distance": {"bot_f1": 0.9840, "bot_recall": 0.9845, "benign_fpr": 0.0048},
    "pca_reconstruction_32": {"bot_f1": 0.9845, "bot_recall": 0.9845, "benign_fpr": 0.0045},
    "known_lr_c1": {"test_f1": 0.9761},
    "known_lr_c10": {"test_f1": 0.9747},
}
PCA_COMPONENTS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and time every qualified CIC-IDS-2018 FHE candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-folder", default="CIC-IDS-2018")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2_000_000,
        help=(
            "Streaming sample size. The default is the sample that passed the "
            "quality gates. Values <= 0 are refused."
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    parser.add_argument("--time-block-minutes", type=int, default=30)
    parser.add_argument("--n-bits", type=int, default=5)
    parser.add_argument("--benign-fpr", type=float, default=0.01)
    parser.add_argument("--target-f1", type=float, default=0.70)
    parser.add_argument("--max-e2e-s", type=float, default=0.70)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--p-error", type=float, default=0.01)
    parser.add_argument("--fhe-records", type=int, default=100)
    parser.add_argument("--fhe-sim-rows", type=int, default=10_000)
    parser.add_argument(
        "--calibration-rows",
        type=int,
        default=10_000,
        help="Representative rows used to calibrate and compile a custom circuit.",
    )
    parser.add_argument(
        "--rounding-threshold-bits",
        type=int,
        default=8,
        help="Accumulator rounding for the custom distance and PCA circuits.",
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--only",
        nargs="+",
        choices=QUALIFIED_MODELS,
        default=None,
        help="Benchmark this subset. The default is all four qualified models.",
    )
    parser.add_argument("--output-root", default="qualified_fhe_runs")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rows <= 0:
        raise SystemExit(
            "Refusing --sample-rows <= 0. That loads all 16,233,002 rows and "
            "does not fit in the 20 GiB RAM cap. Use the default 2000000."
        )
    if args.n_bits < 2:
        raise SystemExit("--n-bits must be at least 2.")
    if not 0 < args.benign_fpr < 1:
        raise SystemExit("--benign-fpr must be between 0 and 1.")
    if not 0 < args.target_f1 < 1:
        raise SystemExit("--target-f1 must be between 0 and 1.")
    if args.max_e2e_s <= 0:
        raise SystemExit("--max-e2e-s must be positive.")
    if args.fhe_records < 1 or args.fhe_sim_rows < 1 or args.calibration_rows < 1:
        raise SystemExit("Record, simulation, and calibration counts must be at least 1.")
    if not 0 < args.p_error < 1:
        raise SystemExit("--p-error must be between 0 and 1.")
    if args.rounding_threshold_bits < 1:
        raise SystemExit("--rounding-threshold-bits must be at least 1.")


def selected_models(args: argparse.Namespace) -> tuple[str, ...]:
    if args.only:
        return tuple(args.only)
    return QUALIFIED_MODELS


def write_rationale(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Qualified-model FHE benchmark",
                "============================",
                "",
                "Models come from anomaly_experiment_runs/20261031_004057_494668.",
                "standard_benign_distance and pca_reconstruction_32 qualified at the",
                "1% validation benign false-positive operating point, with Infiltration",
                "removed from the Bot-vs-benign score. known_lr_c1 and known_lr_c10",
                "qualified in Concrete ML 5-bit clear mode on globally disjoint",
                "30-minute blocks.",
                "",
                "The custom circuits output a score. The decision threshold is applied",
                "after decryption. For distance and PCA, that threshold is chosen from",
                "quantized-clear validation benign scores at the 1% false-positive",
                "budget. Logistic regression keeps the validation F1 threshold used by",
                "the quality experiment.",
                "",
                "End-to-end latency is the mean of encrypt + inference + decrypt.",
                "One-time key generation is reported separately and is outside that mean.",
                "",
                "A logistic-regression pass is known-family detection. A distance or",
                "PCA pass is Bot detection against future benign traffic.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def choose_indices(groups: np.ndarray, count: int, log) -> np.ndarray:
    count = min(int(count), len(groups))
    indices = np.arange(len(groups))
    if count == len(groups):
        return indices
    try:
        _, selected = train_test_split(
            indices,
            test_size=count,
            random_state=SEED,
            stratify=groups,
        )
    except ValueError as exc:
        log(f"Stratified subsample was not possible ({exc}). Using a seeded sample.")
        selected = np.random.default_rng(SEED).choice(indices, size=count, replace=False)
    return np.asarray(selected)


def anomaly_groups(labels: np.ndarray, y: np.ndarray) -> np.ndarray:
    groups = np.full(len(labels), "other", dtype=object)
    groups[y == 0] = "benign"
    groups[np.asarray(labels) == "bot"] = "bot"
    return groups


def calibration_inputset(matrices: list[np.ndarray], count: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    total = sum(len(matrix) for matrix in matrices)
    parts: list[np.ndarray] = []
    for matrix in matrices:
        take = min(len(matrix), max(1, int(round(count * len(matrix) / total))))
        chosen = rng.choice(len(matrix), size=take, replace=False)
        parts.append(np.asarray(matrix[chosen], dtype=np.float32))
    lows = np.min(np.vstack([matrix.min(axis=0) for matrix in matrices]), axis=0)
    highs = np.max(np.vstack([matrix.max(axis=0) for matrix in matrices]), axis=0)
    parts.append(lows.reshape(1, -1).astype(np.float32))
    parts.append(highs.reshape(1, -1).astype(np.float32))
    return np.ascontiguousarray(np.vstack(parts))


def with_feature_bounds(primary: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    rows = [primary]
    for matrix in others:
        rows.append(np.asarray(matrix.min(axis=0, keepdims=True), dtype=primary.dtype))
        rows.append(np.asarray(matrix.max(axis=0, keepdims=True), dtype=primary.dtype))
    return np.ascontiguousarray(np.vstack(rows))


def load_frame(args: argparse.Namespace, log) -> pd.DataFrame:
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
    del sample
    return frame


def benign_false_positive_rate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    benign = y_true == 0
    if not np.any(benign):
        raise ValueError("The evaluated rows contain no benign examples.")
    return float((scores[benign] >= threshold).mean())


def scoped_bot_metrics(
    test_y: np.ndarray,
    test_labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    metrics = classification_metrics(test_y, scores, threshold)
    scoped_mask = np.isin(test_labels, ("benign", *SCOPED_UNKNOWN_ATTACK_LABELS))
    scoped_y = np.isin(test_labels[scoped_mask], SCOPED_UNKNOWN_ATTACK_LABELS).astype(np.int8)
    scoped = classification_metrics(scoped_y, scores[scoped_mask], threshold)
    return {
        "test_f1": metrics["f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_benign_fpr": benign_false_positive_rate(test_y, scores, threshold),
        "bot_f1": scoped["f1"],
        "bot_precision": scoped["precision"],
        "bot_recall": scoped["recall"],
    }


def quantized_lr_scores(model, values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values, fhe="disable"))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected binary quantized predict_proba output with two columns.")
    return probabilities[:, 1]


def quantized_module_scores(module, values: np.ndarray, fhe_mode: str) -> np.ndarray:
    scores = np.asarray(
        module.forward(np.ascontiguousarray(values, dtype=np.float32), fhe=fhe_mode)
    )
    return np.reshape(scores, (len(values),))


def torch_float_scores(module, values: np.ndarray) -> np.ndarray:
    import torch

    module.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        scores = module(tensor).detach().cpu().numpy()
    return np.reshape(scores, (len(values),))


def assert_torch_matches_numpy(module, numpy_scores: np.ndarray, values: np.ndarray, log) -> None:
    comparable = min(512, len(values))
    torch_scores = torch_float_scores(module, values[:comparable])
    delta = float(np.max(np.abs(torch_scores - numpy_scores[:comparable])))
    log(f"Float torch graph vs numpy max abs score delta={delta:.6f} on {comparable} rows.")
    if delta > 1e-2:
        raise RuntimeError(
            f"Custom circuit does not match the numpy score (max abs delta {delta:.6f})."
        )


def make_distance_module(center: np.ndarray, scale: np.ndarray):
    import torch

    class _Score(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            center_array = np.asarray(center, dtype=np.float32)
            scale_array = np.asarray(scale, dtype=np.float32)
            self.register_buffer("center", torch.from_numpy(center_array.copy()))
            self.register_buffer("inv_scale", torch.from_numpy((1.0 / scale_array).copy()))
            self.register_buffer(
                "inv_count",
                torch.tensor([1.0 / center_array.shape[0]], dtype=torch.float32),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            standardized = (inputs - self.center) * self.inv_scale
            squared = standardized * standardized
            return squared.sum(dim=1, keepdim=True) * self.inv_count

    return _Score()


def make_pca_module(mean: np.ndarray, components: np.ndarray):
    import torch

    class _Score(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mean_array = np.asarray(mean, dtype=np.float32)
            component_array = np.asarray(components, dtype=np.float32)
            self.register_buffer("mean", torch.from_numpy(mean_array.copy()))
            self.register_buffer("components", torch.from_numpy(component_array.copy()))
            self.register_buffer(
                "inv_count",
                torch.tensor([1.0 / mean_array.shape[0]], dtype=torch.float32),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            centered = inputs - self.mean
            projected = centered @ self.components.T
            reconstructed = projected @ self.components + self.mean
            error = inputs - reconstructed
            return (error * error).sum(dim=1, keepdim=True) * self.inv_count

    return _Score()


def gpu_parameter_failure(device: str, exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}"
    return device == "cuda" and (
        "NoParametersFound" in message
        or "no parameters" in message.lower()
        or "Unfeasible noise constraint" in message
        or "noise constraint" in message.lower()
    )


def compile_torch_for_device(build_module: Callable, inputset: np.ndarray, args, log):
    from concrete.ml.torch.compile import compile_torch_model
    from gpu_utils import resolve_fhe_device

    requested = (args.device or "auto").strip().lower()
    resolved = resolve_fhe_device(requested, log_fn=log)
    allow_fallback = requested == "auto" or os.environ.get("PPML_FHE_GPU_FALLBACK", "0") == "1"
    n_bits = {
        "op_inputs": args.n_bits,
        "op_weights": args.n_bits,
        "model_inputs": args.n_bits,
        "model_outputs": args.n_bits,
    }

    def attempt(device: str):
        log(
            f"Compiling custom circuit on device={device} with n_bits={args.n_bits} "
            f"and rounding_threshold_bits={args.rounding_threshold_bits}..."
        )
        started = time.perf_counter()
        compiled = compile_torch_model(
            build_module(),
            inputset,
            n_bits=n_bits,
            p_error=args.p_error,
            rounding_threshold_bits=args.rounding_threshold_bits,
            device=device,
            verbose=False,
        )
        return compiled, time.perf_counter() - started

    try:
        compiled, seconds = attempt(resolved)
    except Exception as exc:
        if gpu_parameter_failure(resolved, exc) and allow_fallback:
            log(f"WARNING: CUDA compilation failed ({type(exc).__name__}: {exc}). Falling back to cpu.")
            compiled, seconds = attempt("cpu")
            resolved = "cpu"
        else:
            raise
    log(f"Custom circuit compiled in {seconds:.1f}s on device={resolved}.")
    return compiled, resolved, seconds


def measure_latency(model_dir: Path, rows: np.ndarray, run_dir: Path, model_id: str, log) -> dict:
    from fhe_latency import measure_fhe_roundtrip

    latency_rows = np.ascontiguousarray(rows, dtype=np.float32)
    log(
        f"Measuring {len(latency_rows):,} real round trips for {model_id}. "
        "Key generation is excluded from the end-to-end mean."
    )
    latency = measure_fhe_roundtrip(
        str(model_dir),
        latency_rows,
        n_records=len(latency_rows),
        log_fn=log,
    )
    write_json(
        run_dir / f"latency_times_{model_id}.json",
        {
            "encrypt_times": latency["encrypt_times"],
            "inference_times": latency["inference_times"],
            "decrypt_times": latency["decrypt_times"],
        },
    )
    return {
        key: value
        for key, value in latency.items()
        if key not in {"encrypt_times", "inference_times", "decrypt_times", "predictions"}
    }


def save_circuit(run_dir: Path, model_id: str, compiled, device: str, meta: dict) -> Path:
    from gpu_utils import save_fhe_model

    model_dir = run_dir / "fhe" / model_id / device
    save_fhe_model(str(model_dir), compiled, device, meta=meta)
    return model_dir


def failed_result(model_id: str, protocol: str, exc: Exception) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "protocol": protocol,
        "status": "failed",
        "gate": "FAIL",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def anomaly_gate(result: dict[str, Any], args: argparse.Namespace) -> str:
    required = (
        "quantized_bot_f1",
        "quantized_bot_recall",
        "quantized_benign_fpr",
        "simulation_bot_f1",
        "simulation_bot_recall",
        "mean_e2e_s",
    )
    if result.get("status") != "ok" or any(key not in result for key in required):
        return "FAIL"
    passed = (
        result["quantized_bot_f1"] >= args.target_f1
        and result["quantized_bot_recall"] >= args.target_f1
        and result["quantized_benign_fpr"] <= args.benign_fpr
        and result["simulation_bot_f1"] >= args.target_f1
        and result["simulation_bot_recall"] >= args.target_f1
        and result["mean_e2e_s"] <= args.max_e2e_s
    )
    return "PASS" if passed else "FAIL"


def lr_gate(result: dict[str, Any], args: argparse.Namespace) -> str:
    if result.get("status") != "ok":
        return "FAIL"
    if "clear_f1" not in result or "simulation_f1" not in result or "mean_e2e_s" not in result:
        return "FAIL"
    passed = (
        result["clear_f1"] >= args.target_f1
        and result["simulation_f1"] >= args.target_f1
        and result["mean_e2e_s"] <= args.max_e2e_s
    )
    return "PASS" if passed else "FAIL"


def remember_reference(result: dict[str, Any], observed: dict[str, float], log) -> None:
    reference = REFERENCES.get(result["model_id"])
    if reference is None:
        return
    deltas = {
        f"{key}_delta": float(observed[key] - reference[key])
        for key in reference
        if key in observed
    }
    result["reference_deltas"] = deltas
    log(
        f"{result['model_id']} reference deltas vs 20261031_004057_494668: "
        + ", ".join(f"{key}={value:+.4f}" for key, value in deltas.items())
    )


def finish_custom_model(
    result: dict[str, Any],
    compiled,
    compiled_device: str,
    compile_seconds: float,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    log,
) -> dict[str, Any]:
    model_id = result["model_id"]
    result.update(
        {
            "compiled_device": compiled_device,
            "compile_seconds": compile_seconds,
            "n_bits": args.n_bits,
            "p_error": args.p_error,
            "rounding_threshold_bits": args.rounding_threshold_bits,
        }
    )
    model_dir = save_circuit(
        run_dir,
        f"{model_id}_q{args.n_bits}",
        compiled,
        compiled_device,
        {
            "model_id": model_id,
            "n_bits": args.n_bits,
            "p_error": args.p_error,
            "compiled_device": compiled_device,
            "rounding_threshold_bits": args.rounding_threshold_bits,
        },
    )
    result["model_dir"] = str(model_dir)
    errors: list[str] = []
    try:
        validation_scores = quantized_module_scores(compiled, x_validation, "disable")
        test_scores = quantized_module_scores(compiled, x_test, "disable")
        threshold = threshold_from_benign(validation_scores, y_validation, args.benign_fpr)
        quality = scoped_bot_metrics(y_test, test_labels, test_scores, threshold)
        result.update(
            {
                "threshold": threshold,
                "quantized_bot_f1": quality["bot_f1"],
                "quantized_bot_precision": quality["bot_precision"],
                "quantized_bot_recall": quality["bot_recall"],
                "quantized_benign_fpr": quality["test_benign_fpr"],
                "quantized_test_f1": quality["test_f1"],
                "quality_f1": quality["bot_f1"],
                "quality_recall": quality["bot_recall"],
                "quality_benign_fpr": quality["test_benign_fpr"],
                "clear_breakdown": per_label_detection(
                    test_labels,
                    test_scores,
                    threshold,
                    "compact_flow",
                    model_id,
                    mode="quantized_clear",
                    n_bits=args.n_bits,
                ),
            }
        )
        log(
            f"{model_id} quantized-clear Bot F1={quality['bot_f1']:.4f}, "
            f"recall={quality['bot_recall']:.4f}, benign FPR={quality['test_benign_fpr']:.4f}."
        )
    except Exception as exc:
        errors.append(f"quantized clear {type(exc).__name__}: {exc}")
        log(errors[-1])

    try:
        groups = anomaly_groups(test_labels, y_test)
        simulation_indices = choose_indices(groups, args.fhe_sim_rows, log)
        started = time.perf_counter()
        simulated = quantized_module_scores(compiled, x_test[simulation_indices], "simulate")
        result["simulation_seconds"] = time.perf_counter() - started
        if "threshold" not in result:
            raise RuntimeError("Simulation has no quantized threshold.")
        simulated_metrics = scoped_bot_metrics(
            y_test[simulation_indices],
            test_labels[simulation_indices],
            simulated,
            result["threshold"],
        )
        result.update(
            {
                "simulation_rows": int(len(simulation_indices)),
                "simulation_bot_f1": simulated_metrics["bot_f1"],
                "simulation_bot_recall": simulated_metrics["bot_recall"],
                "simulation_benign_fpr": simulated_metrics["test_benign_fpr"],
                "simulation_f1": simulated_metrics["bot_f1"],
                "simulation_breakdown": per_label_detection(
                    test_labels[simulation_indices],
                    simulated,
                    result["threshold"],
                    "compact_flow",
                    model_id,
                    mode="fhe_simulation",
                    n_bits=args.n_bits,
                ),
            }
        )
        log(
            f"{model_id} simulation Bot F1={simulated_metrics['bot_f1']:.4f}, "
            f"recall={simulated_metrics['bot_recall']:.4f}, "
            f"benign FPR={simulated_metrics['test_benign_fpr']:.4f}."
        )
    except Exception as exc:
        errors.append(f"simulation {type(exc).__name__}: {exc}")
        log(errors[-1])

    try:
        groups = anomaly_groups(test_labels, y_test)
        latency_indices = choose_indices(groups, args.fhe_records, log)
        result.update(measure_latency(model_dir, x_test[latency_indices], run_dir, model_id, log))
        log(
            f"{model_id} mean end-to-end={result['mean_e2e_s']:.6f}s "
            f"on device={compiled_device}."
        )
    except Exception as exc:
        errors.append(f"latency {type(exc).__name__}: {exc}")
        log(errors[-1])

    if errors:
        result["status"] = "failed"
        result["error"] = " | ".join(errors)
    else:
        result["status"] = "ok"
    result["gate"] = anomaly_gate(result, args)
    return result


def run_distance_model(arrays: dict[str, Any], args, run_dir: Path, log) -> dict[str, Any]:
    model_id = "standard_benign_distance"
    log(f"Fitting {model_id}.")
    started = time.perf_counter()
    scorer, parameters = make_distance_scorer(arrays["x_train_benign"], robust=False)
    fit_seconds = time.perf_counter() - started
    float_validation = scorer(arrays["x_validation"])
    float_test = scorer(arrays["x_test"])
    float_threshold = threshold_from_benign(float_validation, arrays["y_validation"], args.benign_fpr)
    float_metrics = scoped_bot_metrics(
        arrays["y_test"],
        arrays["test_labels"],
        float_test,
        float_threshold,
    )
    result = {
        "model_id": model_id,
        "protocol": "unknown_anomaly_capture_day",
        "fit_seconds": fit_seconds,
        "float_bot_f1": float_metrics["bot_f1"],
        "float_bot_recall": float_metrics["bot_recall"],
        "float_benign_fpr": float_metrics["test_benign_fpr"],
        "float_threshold": float_threshold,
    }
    remember_reference(
        result,
        {
            "bot_f1": float_metrics["bot_f1"],
            "bot_recall": float_metrics["bot_recall"],
            "benign_fpr": float_metrics["test_benign_fpr"],
        },
        log,
    )
    module = make_distance_module(parameters["center"], parameters["scale"])
    assert_torch_matches_numpy(module, float_test, arrays["x_test"], log)
    inputset = calibration_inputset(
        [arrays["x_train_benign"], arrays["x_validation"], arrays["x_test"]],
        args.calibration_rows,
    )
    compiled, device, compile_seconds = compile_torch_for_device(
        lambda: make_distance_module(parameters["center"], parameters["scale"]),
        inputset,
        args,
        log,
    )
    write_json(
        run_dir / "models" / f"{model_id}_parameters.json",
        {
            "center": parameters["center"].tolist(),
            "scale": parameters["scale"].tolist(),
            "float_threshold": float_threshold,
        },
    )
    return finish_custom_model(
        result,
        compiled,
        device,
        compile_seconds,
        arrays["x_validation"],
        arrays["y_validation"],
        arrays["x_test"],
        arrays["y_test"],
        arrays["test_labels"],
        args,
        run_dir,
        log,
    )


def run_pca_model(arrays: dict[str, Any], args, run_dir: Path, log) -> dict[str, Any]:
    model_id = "pca_reconstruction_32"
    if PCA_COMPONENTS >= arrays["x_train_benign"].shape[1]:
        raise RuntimeError(
            f"PCA-{PCA_COMPONENTS} needs fewer components than "
            f"{arrays['x_train_benign'].shape[1]} inputs."
        )
    log(f"Fitting {model_id}.")
    started = time.perf_counter()
    model = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=SEED)
    model.fit(arrays["x_train_benign"])
    fit_seconds = time.perf_counter() - started
    float_validation = score_pca(model, arrays["x_validation"])
    float_test = score_pca(model, arrays["x_test"])
    float_threshold = threshold_from_benign(float_validation, arrays["y_validation"], args.benign_fpr)
    float_metrics = scoped_bot_metrics(
        arrays["y_test"],
        arrays["test_labels"],
        float_test,
        float_threshold,
    )
    result = {
        "model_id": model_id,
        "protocol": "unknown_anomaly_capture_day",
        "fit_seconds": fit_seconds,
        "float_bot_f1": float_metrics["bot_f1"],
        "float_bot_recall": float_metrics["bot_recall"],
        "float_benign_fpr": float_metrics["test_benign_fpr"],
        "float_threshold": float_threshold,
    }
    remember_reference(
        result,
        {
            "bot_f1": float_metrics["bot_f1"],
            "bot_recall": float_metrics["bot_recall"],
            "benign_fpr": float_metrics["test_benign_fpr"],
        },
        log,
    )
    assert_torch_matches_numpy(
        make_pca_module(model.mean_, model.components_),
        float_test,
        arrays["x_test"],
        log,
    )
    inputset = calibration_inputset(
        [arrays["x_train_benign"], arrays["x_validation"], arrays["x_test"]],
        args.calibration_rows,
    )
    compiled, device, compile_seconds = compile_torch_for_device(
        lambda: make_pca_module(model.mean_, model.components_),
        inputset,
        args,
        log,
    )
    write_json(
        run_dir / "models" / f"{model_id}_parameters.json",
        {
            "mean": np.asarray(model.mean_).tolist(),
            "components": np.asarray(model.components_).tolist(),
            "float_threshold": float_threshold,
        },
    )
    return finish_custom_model(
        result,
        compiled,
        device,
        compile_seconds,
        arrays["x_validation"],
        arrays["y_validation"],
        arrays["x_test"],
        arrays["y_test"],
        arrays["test_labels"],
        args,
        run_dir,
        log,
    )


def run_lr_model(
    model_id: str,
    c_value: float,
    arrays: dict[str, Any],
    args,
    run_dir: Path,
    log,
) -> dict[str, Any]:
    from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
    from gpu_utils import compile_for_device

    log(f"Fitting {model_id} at {args.n_bits} bits.")
    model = ConcreteLogisticRegression(
        n_bits=args.n_bits,
        C=c_value,
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        n_jobs=args.jobs,
        random_state=SEED,
    )
    started = time.perf_counter()
    model.fit(arrays["x_train"], arrays["y_train"])
    fit_seconds = time.perf_counter() - started
    validation_scores = quantized_lr_scores(model, arrays["x_validation"])
    test_scores = quantized_lr_scores(model, arrays["x_test"])
    threshold, validation_f1 = choose_f1_threshold(arrays["y_validation"], validation_scores)
    metrics = classification_metrics(arrays["y_test"], test_scores, threshold)
    benign_fpr = benign_false_positive_rate(arrays["y_test"], test_scores, threshold)
    result: dict[str, Any] = {
        "model_id": model_id,
        "protocol": "known_family_time_block",
        "c_value": c_value,
        "n_bits": args.n_bits,
        "fit_seconds": fit_seconds,
        "threshold": threshold,
        "validation_f1": validation_f1,
        "clear_f1": metrics["f1"],
        "clear_precision": metrics["precision"],
        "clear_recall": metrics["recall"],
        "clear_benign_fpr": benign_fpr,
        "quality_f1": metrics["f1"],
        "quality_recall": metrics["recall"],
        "quality_benign_fpr": benign_fpr,
        "clear_breakdown": per_label_detection(
            arrays["test_labels"],
            test_scores,
            threshold,
            "compact_flow",
            model_id,
            mode=f"concrete_quantized_{args.n_bits}bit",
            n_bits=args.n_bits,
        ),
    }
    remember_reference(result, {"test_f1": metrics["f1"]}, log)
    log(
        f"{model_id} clear F1={metrics['f1']:.4f}, precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, benign FPR={benign_fpr:.4f}."
    )
    if metrics["f1"] < args.target_f1:
        result["status"] = "skipped"
        result["gate"] = "FAIL"
        result["error"] = (
            f"Clear F1 {metrics['f1']:.4f} is below {args.target_f1:.2f}. "
            "Compilation was not started."
        )
        log(result["error"])
        return result

    compile_rows = with_feature_bounds(
        arrays["x_train"],
        [arrays["x_validation"], arrays["x_test"]],
    )
    started = time.perf_counter()
    _, compiled_device = compile_for_device(
        model,
        compile_rows,
        device=args.device,
        log_fn=log,
        p_error=args.p_error,
    )
    compile_seconds = time.perf_counter() - started
    model_dir = save_circuit(
        run_dir,
        f"{model_id}_q{args.n_bits}",
        model,
        compiled_device,
        {
            "model_id": model_id,
            "c_value": c_value,
            "n_bits": args.n_bits,
            "selected_threshold": threshold,
            "p_error": args.p_error,
            "compiled_device": compiled_device,
        },
    )
    result.update(
        {
            "compiled_device": compiled_device,
            "compile_seconds": compile_seconds,
            "p_error": args.p_error,
            "model_dir": str(model_dir),
        }
    )
    errors: list[str] = []
    try:
        simulation_indices = choose_indices(arrays["y_test"], args.fhe_sim_rows, log)
        started = time.perf_counter()
        simulated = np.asarray(
            model.predict_proba(arrays["x_test"][simulation_indices], fhe="simulate")
        )[:, 1]
        result["simulation_seconds"] = time.perf_counter() - started
        simulated_metrics = classification_metrics(
            arrays["y_test"][simulation_indices],
            simulated,
            threshold,
        )
        result.update(
            {
                "simulation_rows": int(len(simulation_indices)),
                "simulation_f1": simulated_metrics["f1"],
                "simulation_precision": simulated_metrics["precision"],
                "simulation_recall": simulated_metrics["recall"],
                "simulation_benign_fpr": benign_false_positive_rate(
                    arrays["y_test"][simulation_indices],
                    simulated,
                    threshold,
                ),
                "simulation_breakdown": per_label_detection(
                    arrays["test_labels"][simulation_indices],
                    simulated,
                    threshold,
                    "compact_flow",
                    model_id,
                    mode="fhe_simulation",
                    n_bits=args.n_bits,
                ),
            }
        )
        log(f"{model_id} simulation F1={simulated_metrics['f1']:.4f}.")
    except Exception as exc:
        errors.append(f"simulation {type(exc).__name__}: {exc}")
        log(errors[-1])
    try:
        latency_indices = choose_indices(arrays["y_test"], args.fhe_records, log)
        result.update(
            measure_latency(
                model_dir,
                arrays["x_test"][latency_indices],
                run_dir,
                model_id,
                log,
            )
        )
        log(f"{model_id} mean end-to-end={result['mean_e2e_s']:.6f}s on device={compiled_device}.")
    except Exception as exc:
        errors.append(f"latency {type(exc).__name__}: {exc}")
        log(errors[-1])
    if errors:
        result["status"] = "failed"
        result["error"] = " | ".join(errors)
    else:
        result["status"] = "ok"
    result["gate"] = lr_gate(result, args)
    return result


def prepare_anomaly_arrays(
    frame: pd.DataFrame,
    run_dir: Path,
    log,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import joblib

    train, validation, test, summary = capture_day_split(frame)
    benign_train = train.loc[train["binary_label"] == 0]
    (x_train_benign, x_validation, x_test), preprocessor = fit_transform_compact(
        benign_train,
        [benign_train, validation, test],
    )
    arrays = {
        "x_train_benign": x_train_benign,
        "x_validation": x_validation,
        "x_test": x_test,
        "y_validation": validation["binary_label"].to_numpy(dtype=np.int8),
        "y_test": test["binary_label"].to_numpy(dtype=np.int8),
        "test_labels": test["label"].to_numpy(dtype=str),
    }
    summary = dict(summary)
    summary.update(
        {
            "benign_train_rows": int(len(benign_train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "input_dimensions": int(x_train_benign.shape[1]),
        }
    )
    joblib.dump(preprocessor, run_dir / "models" / "unknown_anomaly_feature_transformer.joblib")
    log(f"Unknown-anomaly split: {summary}")
    return arrays, summary


def prepare_known_arrays(
    frame: pd.DataFrame,
    block_minutes: int,
    run_dir: Path,
    log,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import joblib

    train, validation, test, summary = build_known_family_split(frame, block_minutes)
    if (test["label"] == "infilteration").any() or (train["label"] == "infilteration").any():
        raise AssertionError("Infiltration rows entered the known-family split.")
    (x_train, x_validation, x_test), preprocessor = fit_transform_compact(
        train,
        [train, validation, test],
    )
    arrays = {
        "x_train": x_train,
        "x_validation": x_validation,
        "x_test": x_test,
        "y_train": train["binary_label"].to_numpy(dtype=np.int8),
        "y_validation": validation["binary_label"].to_numpy(dtype=np.int8),
        "y_test": test["binary_label"].to_numpy(dtype=np.int8),
        "test_labels": test["label"].to_numpy(dtype=str),
        "preprocessor": preprocessor,
    }
    summary = dict(summary)
    summary.update(
        {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "input_dimensions": int(x_train.shape[1]),
        }
    )
    if summary["cross_split_block_overlap_count"] != 0:
        raise AssertionError("Known-family split contains cross-split block overlap.")
    joblib.dump(preprocessor, run_dir / "models" / "known_family_feature_transformer.joblib")
    log(f"Known-family split: {summary}")
    return arrays, summary


def format_breakdown(rows: list[dict] | None) -> str:
    if not rows:
        return "No per-label rows."
    frame = pd.DataFrame(rows)
    show = frame.loc[:, ["original_label", "support", "predicted_attack_rate", "metric_name"]].copy()
    show["predicted_attack_rate"] = show["predicted_attack_rate"].map(lambda value: f"{value:.4f}")
    return show.to_string(index=False)


def summary_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "model_id": result.get("model_id", ""),
                "protocol": result.get("protocol", ""),
                "quality_f1": result.get("quality_f1"),
                "quality_recall": result.get("quality_recall"),
                "quality_benign_fpr": result.get("quality_benign_fpr"),
                "simulation_f1": result.get("simulation_f1"),
                "mean_e2e_s": result.get("mean_e2e_s"),
                "gate": result.get("gate", "FAIL"),
            }
        )
    frame = pd.DataFrame(rows)
    for column, digits in (
        ("quality_f1", 4),
        ("quality_recall", 4),
        ("quality_benign_fpr", 4),
        ("simulation_f1", 4),
        ("mean_e2e_s", 6),
    ):
        frame[column] = frame[column].map(
            lambda value, digits=digits: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    return frame


def append_model_details(lines: list[str], result: dict[str, Any]) -> None:
    lines.extend(["", result["model_id"], "-" * len(result["model_id"])])
    lines.append(f"Protocol: {result.get('protocol', '')}")
    lines.append(f"Status: {result.get('status', '')}")
    lines.append(f"Gate: {result.get('gate', 'FAIL')}")
    scalar_keys = (
        "fit_seconds",
        "float_bot_f1",
        "float_bot_recall",
        "float_benign_fpr",
        "float_threshold",
        "quantized_bot_f1",
        "quantized_bot_precision",
        "quantized_bot_recall",
        "quantized_benign_fpr",
        "quantized_test_f1",
        "clear_f1",
        "clear_precision",
        "clear_recall",
        "clear_benign_fpr",
        "validation_f1",
        "threshold",
        "compiled_device",
        "compile_seconds",
        "n_bits",
        "p_error",
        "rounding_threshold_bits",
        "simulation_rows",
        "simulation_seconds",
        "simulation_f1",
        "simulation_precision",
        "simulation_recall",
        "simulation_bot_f1",
        "simulation_bot_recall",
        "simulation_benign_fpr",
        "n_records",
        "keygen_s",
        "mean_encrypt_s",
        "mean_inference_s",
        "mean_decrypt_s",
        "mean_e2e_s",
        "std_encrypt_s",
        "std_inference_s",
        "std_decrypt_s",
    )
    for key in scalar_keys:
        if key in result and result[key] is not None:
            value = result[key]
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")
    if result.get("reference_deltas"):
        lines.append(f"reference_deltas: {result['reference_deltas']}")
    if result.get("model_dir"):
        lines.append(f"model_dir: {result['model_dir']}")
    if result.get("clear_breakdown"):
        lines.extend(["", "Quantized-clear per-label detection:", format_breakdown(result["clear_breakdown"])])
    if result.get("simulation_breakdown"):
        lines.extend(["", "Simulation per-label detection:", format_breakdown(result["simulation_breakdown"])])
    if result.get("error"):
        lines.extend(["", f"Error: {result['error']}"])


def write_report(
    run_dir: Path,
    args: argparse.Namespace,
    dataset_summary: dict[str, Any],
    results: list[dict],
) -> None:
    lines = [
        "CIC-IDS-2018 qualified-model FHE benchmark",
        "=" * 46,
        f"Run directory: {run_dir}",
        f"Generated: {dt.datetime.now().astimezone().isoformat()}",
        f"Arguments: {vars(args)}",
        "",
        "Qualified models from anomaly_experiment_runs/20261031_004057_494668:",
        ", ".join(QUALIFIED_MODELS),
        "",
        "Dataset and splits",
        "------------------",
    ]
    lines.extend(f"{key}: {value}" for key, value in dataset_summary.items())
    lines.extend(
        [
            "",
            "Summary",
            "-------",
            "quality_f1 is scoped Bot-vs-benign F1 for distance and PCA, and",
            "known-family test F1 for logistic regression. For distance and PCA,",
            "quality_benign_fpr is the full-test quantized-clear false-positive rate",
            "and is a pass requirement. Simulation benign FPR is in the model section.",
            "Logistic regression passes on clear F1, simulation F1, and latency.",
            "Mean end-to-end latency excludes one-time key generation.",
            "",
            summary_frame(results).to_string(index=False) if results else "No model finished.",
        ]
    )
    for result in results:
        append_model_details(lines, result)
    passing = [result["model_id"] for result in results if result.get("gate") == "PASS"]
    lines.extend(
        [
            "",
            "Passing models",
            "---------------",
            ", ".join(passing) if passing else "None.",
            "",
            "Claim",
            "-----",
            "Describe a passing logistic regression as known-family detection.",
            "Describe a passing distance or PCA circuit as Bot detection against",
            "future benign traffic. Infiltration stays out of that scoped score.",
            "",
        ]
    )
    (run_dir / "report.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def public_results(results: list[dict]) -> list[dict]:
    hidden = {"clear_breakdown", "simulation_breakdown", "traceback"}
    return [{key: value for key, value in result.items() if key not in hidden} for result in results]


def store_progress(
    run_dir: Path,
    args: argparse.Namespace,
    dataset_summary: dict[str, Any],
    results: list[dict],
    status: str,
    error: str | None = None,
) -> None:
    write_json(run_dir / "results.json", public_results(results))
    if results:
        write_report(run_dir, args, dataset_summary, results)
    payload: dict[str, Any] = {
        "status": status,
        "updated": dt.datetime.now().astimezone().isoformat(),
        "models": [result.get("model_id") for result in results],
        "passing_models": [result["model_id"] for result in results if result.get("gate") == "PASS"],
    }
    if error:
        payload["error"] = error
    write_json(run_dir / "status.json", payload)


def run_named_model(model_id: str, anomaly_arrays, known_arrays, args, run_dir: Path, log) -> dict:
    if model_id == "standard_benign_distance":
        return run_distance_model(anomaly_arrays, args, run_dir, log)
    if model_id == "pca_reconstruction_32":
        return run_pca_model(anomaly_arrays, args, run_dir, log)
    if model_id in LR_C_VALUES:
        return run_lr_model(model_id, LR_C_VALUES[model_id], known_arrays, args, run_dir, log)
    raise KeyError(model_id)


def main() -> int:
    args = parse_args()
    validate_args(args)
    run_dir = create_run_dir(args.output_root)
    (run_dir / "models").mkdir()
    log = make_logger(run_dir)
    write_json(run_dir / "arguments.json", vars(args))
    write_rationale(run_dir / "rationale.txt")
    models = selected_models(args)
    log(f"New qualified FHE benchmark: {run_dir}")
    log("Models: " + ", ".join(models))
    log("Existing anomaly reports and model artifacts will not be modified.")
    dataset_summary: dict[str, Any] = {"sample_rows_requested": args.sample_rows}
    results: list[dict] = []
    try:
        frame = load_frame(args, log)
        dataset_summary["clean_rows"] = int(len(frame))

        def attempt(model_id: str, anomaly_arrays, known_arrays) -> None:
            protocol = (
                "unknown_anomaly_capture_day"
                if model_id in ANOMALY_MODELS
                else "known_family_time_block"
            )
            log(f"Starting {model_id}.")
            try:
                result = run_named_model(
                    model_id,
                    anomaly_arrays,
                    known_arrays,
                    args,
                    run_dir,
                    log,
                )
            except Exception as exc:
                log(f"{model_id} failed: {type(exc).__name__}: {exc}")
                result = failed_result(model_id, protocol, exc)
            results.append(result)
            store_progress(run_dir, args, dataset_summary, results, "running")
            log(f"{model_id} gate={result.get('gate', 'FAIL')}.")

        if any(model_id in ANOMALY_MODELS for model_id in models):
            anomaly_arrays, anomaly_summary = prepare_anomaly_arrays(frame, run_dir, log)
            dataset_summary["unknown_anomaly"] = anomaly_summary
            for model_id in models:
                if model_id in ANOMALY_MODELS:
                    attempt(model_id, anomaly_arrays, None)
            del anomaly_arrays
        if any(model_id in LR_C_VALUES for model_id in models):
            known_arrays, known_summary = prepare_known_arrays(
                frame,
                args.time_block_minutes,
                run_dir,
                log,
            )
            dataset_summary["known_family_time_blocks"] = known_summary
            del frame
            for model_id in models:
                if model_id in LR_C_VALUES:
                    attempt(model_id, None, known_arrays)
        else:
            del frame
        store_progress(run_dir, args, dataset_summary, results, "completed")
        passing = [result["model_id"] for result in results if result.get("gate") == "PASS"]
        log(
            "Benchmark finished. Passing models: "
            + (", ".join(passing) if passing else "none")
            + f". Read {run_dir / 'report.txt'}"
        )
        return 0
    except Exception as exc:
        store_progress(
            run_dir,
            args,
            dataset_summary,
            results,
            "failed",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        log(f"Benchmark failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
