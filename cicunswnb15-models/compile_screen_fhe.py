#!/usr/bin/env python3
"""Compile the CIC-UNSW-NB15 screen winners and time real FHE inference.

Run from this directory on the dataset machine. This workspace does not
execute the script. The source report is the finished screen:

    screen_runs/20261120_125444_465319/report.txt

    python compile_screen_fhe.py
    python compile_screen_fhe.py --device cpu
    python compile_screen_fhe.py --only core core+segment
    python compile_screen_fhe.py --bits 5
    python compile_screen_fhe.py --sim-rows 10000

Each plaintext winner is refit at 8, 7, 6, and 5 bits. Clear validation
and test scores are compared with that report. The bit width is the
highest validation attack F1, then the lower benign false-positive rate,
then fewer bits. --bits compiles that width instead of the chosen one.

The chosen forest is compiled with the same device helper and p_error as
the CIC-IDS-2018 forests. FHE simulation is scored on the test rows.
Each of 100 records is timed as encrypt, inference, and decrypt. The
latency that matters is the mean of the inference phase alone. Encrypt,
decrypt, their sum, and one-time key generation are recorded beside it.
One model failing does not stop the next. Output is fhe_runs/<local timestamp>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

SOURCE_RUN = "20261120_125444_465319"
SOURCE_REPORT = Path("screen_runs") / SOURCE_RUN / "report.txt"
FEATURE_SETS = (
    "core",
    "core+length",
    "core+segment",
    "core+window",
    "core+length+segment",
    "core+length+window",
    "core+segment+window",
    "core+length+segment+window",
)
RECALL_NAMES = ("Shellcode", "Reconnaissance", "Analysis", "Backdoor", "Worms")
BIT_WIDTHS = (8, 7, 6, 5)
TARGET_F1 = 0.70
MAX_INFERENCE_S = 0.70
REPRODUCTION_TOLERANCE = 0.001
SIMULATION_TOLERANCE = 0.005
P_ERROR = 0.01
SIM_BATCH_ROWS = 4096

ROW_RE = re.compile(
    r"^(plaintext|quantized)\s+"
    r"(validation|test)\s+"
    r"(\S+)\s+"
    r"trees=(\d+)\s+depth=(\d+)\s+bits=(\S+)\s+"
    r"f1=([0-9.]+)\s+fpr=([0-9.]+)"
    r"(?:\s+(.*))?$"
)
RECALL_RE = re.compile(
    r"(Shellcode|Reconnaissance|Analysis|Backdoor|Worms)=([0-9.]+)"
)

CSV_FIELDS = [
    "feature_set",
    "n_estimators",
    "max_depth",
    "n_bits",
    "clear_validation_f1",
    "clear_validation_fpr",
    "clear_test_f1",
    "clear_test_fpr",
    "screen_test_f1",
    "screen_test_fpr",
    "simulation_rows",
    "simulation_f1",
    "simulation_fpr",
    "simulation_disagree",
    "real_records",
    "real_disagree",
    "compile_seconds",
    "compiled_device",
    "keygen_s",
    "mean_encrypt_s",
    "mean_inference_s",
    "mean_decrypt_s",
    "mean_e2e_s",
    "gate",
    "error",
]


def parse_report(text: str) -> tuple[list[dict], dict[tuple, dict]]:
    """Read winner shapes and the printed clear scores from a screen report."""
    winners = []
    rows: dict[tuple, dict] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = ROW_RE.match(line.strip())
        if not match:
            continue
        phase, split, feature_set, trees, depth, bits, f1, fpr, tail = match.groups()
        bit_width = None if bits == "-" else int(bits)
        row = {
            "phase": phase,
            "split": split,
            "feature_set": feature_set,
            "trees": int(trees),
            "depth": int(depth),
            "bits": bit_width,
            "f1": float(f1),
            "fpr": float(fpr),
        }
        for name, value in RECALL_RE.findall(tail or ""):
            row[name] = float(value)
        key = (phase, feature_set, split, bit_width)
        if key in rows:
            raise SystemExit(f"duplicate score on line {line_number}: {key}")
        rows[key] = row
        if phase == "plaintext" and split == "validation":
            winners.append(row)
    if not winners:
        raise SystemExit("source report has no plaintext validation winners")
    return winners, rows


def choose_bit(validation_rows: list[dict]) -> dict:
    """Validation attack F1, then lower benign FPR, then fewer bits."""
    if not validation_rows:
        raise ValueError("no validation rows")
    return max(validation_rows, key=lambda row: (row["f1"], -row["fpr"], -int(row["bits"])))


def label_from_values(values: list[float]) -> int:
    """Class id from one decrypted record.

    A single output is the class label. Two outputs in the unit interval
    are class probabilities. Anything wider is an argmax.
    """
    if len(values) == 1:
        return int(round(values[0]))
    if len(values) == 2 and all(-0.05 <= value <= 1.05 for value in values):
        return int(values[1] >= values[0])
    return max(range(len(values)), key=lambda index: values[index])


def flatten_decrypted(decrypted) -> list[float]:
    if hasattr(decrypted, "tolist"):
        decrypted = decrypted.tolist()
    if isinstance(decrypted, (int, float)):
        return [float(decrypted)]
    values: list[float] = []

    def walk(node) -> None:
        if hasattr(node, "tolist"):
            node = node.tolist()
        if isinstance(node, (int, float)):
            values.append(float(node))
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
            return
        values.append(float(node))

    walk(decrypted)
    return values


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def decide(result: dict) -> tuple[str, list[str]]:
    """Pass when the refit matches the screen, simulation matches clear, and inference latency is inside the budget."""
    reasons = []
    reasons.extend(result.get("errors") or [])
    reasons.extend(result.get("reproduction_failures") or [])
    clear_f1 = result.get("clear_test_f1")
    sim_f1 = result.get("simulation_f1")
    sim_fpr = result.get("simulation_fpr")
    sim_clear_f1 = result.get("simulation_clear_f1")
    sim_clear_fpr = result.get("simulation_clear_fpr")
    inference = result.get("mean_inference_s")
    if clear_f1 is None:
        reasons.append("clear test score is missing")
    elif clear_f1 < TARGET_F1:
        reasons.append(f"clear test F1 {clear_f1:.4f} is below {TARGET_F1:.2f}")
    if None in (sim_f1, sim_fpr, sim_clear_f1, sim_clear_fpr):
        reasons.append("simulation score is missing")
    else:
        if sim_f1 < TARGET_F1:
            reasons.append(f"simulation F1 {sim_f1:.4f} is below {TARGET_F1:.2f}")
        f1_gap = abs(sim_f1 - sim_clear_f1)
        fpr_gap = abs(sim_fpr - sim_clear_fpr)
        if f1_gap > SIMULATION_TOLERANCE:
            reasons.append(f"simulation F1 differs from clear by {f1_gap:.4f}")
        if fpr_gap > SIMULATION_TOLERANCE:
            reasons.append(f"simulation benign FPR differs from clear by {fpr_gap:.4f}")
    if inference is None:
        reasons.append("inference latency measurement is missing")
    elif inference > MAX_INFERENCE_S:
        reasons.append(
            f"mean inference {inference:.4f}s is above {MAX_INFERENCE_S:.2f}s"
        )
    return ("FAIL", reasons) if reasons else ("PASS", [])


def score_gap(observed: float, expected: float) -> str | None:
    if abs(observed - expected) > REPRODUCTION_TOLERANCE:
        return f"{observed - expected:+.4f}"
    return None


def resolve_source(path: Path) -> Path:
    if path.is_file():
        return path
    beside_script = Path(__file__).resolve().parent / path
    if beside_script.is_file():
        return beside_script
    raise SystemExit(f"source report not found: {path}")


def choose_indices(groups, count: int, seed: int, log: Callable[[str], None] | None = None) -> list[int]:
    count = min(int(count), len(groups))
    indices = list(range(len(groups)))
    if count == len(groups):
        return indices
    if count < 1:
        raise ValueError("count must be >= 1")
    from sklearn.model_selection import train_test_split

    try:
        _, selected = train_test_split(
            indices,
            test_size=count,
            random_state=seed,
            stratify=groups,
        )
    except ValueError as exc:
        if log is not None:
            log(f"Stratified subsample was not possible ({exc}). Using a seeded sample.")
        import random

        return sorted(random.Random(seed).sample(indices, count))
    return [int(index) for index in selected]


def class_labels(raw, count: int):
    import numpy as np

    array = np.asarray(raw)
    if array.ndim == 2 and array.shape == (count, 2):
        return (array[:, 1] >= array[:, 0]).astype(np.int8)
    flat = np.asarray(array).reshape(count, -1)
    if flat.shape[1] == 1:
        return np.rint(flat[:, 0]).astype(np.int8)
    return np.argmax(flat, axis=1).astype(np.int8)


def release_memory() -> None:
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def make_logger(run_dir: Path) -> Callable[[str], None]:
    log_path = run_dir / "run.log"

    def log(message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def create_run_dir(root: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def import_compile_helpers():
    sibling = Path(__file__).resolve().parent.parent / "cicids2018-kdd-models"
    sibling_text = str(sibling)
    if sibling_text not in sys.path:
        sys.path.insert(0, sibling_text)
    from gpu_utils import compile_for_device, save_fhe_model

    return compile_for_device, save_fhe_model


def reference_row(rows: dict[tuple, dict], feature_set: str, split: str, bits: int) -> dict | None:
    return rows.get(("quantized", feature_set, split, bits))


def recall_text(metrics: dict, prefix: str = "recall_") -> str:
    parts = []
    for name in RECALL_NAMES:
        key = name if name in metrics else f"{prefix}{name}"
        if key in metrics and metrics[key] not in ("", None):
            parts.append(f"{name}={float(metrics[key]):.3f}")
    return " ".join(parts)


def measure_roundtrip(model_dir: Path, rows, log: Callable[[str], None]) -> dict:
    """Time encrypt, inference, and decrypt on each record.

    Inference is only ``server.run``. Encrypt and decrypt have their own
    timers and are not added into that mean. Key generation is reported
    once and is outside both means. Evaluation keys are deserialized once,
    before the per-record loop. The decrypted value is kept so it can be
    compared with the clear label.
    """
    import os

    from concrete import fhe
    from concrete.ml.deployment import FHEModelClient, FHEModelServer

    count = int(len(rows))
    if count < 1:
        raise ValueError("latency sample is empty")
    model_text = str(model_dir)
    log(f"Loading FHE client/server from '{model_text}' for latency measurement...")
    marker = os.path.join(model_text, "fhe_device.txt")
    if os.path.isfile(marker):
        compiled_device = Path(marker).read_text(encoding="utf-8").strip() or "unknown"
        log(f"Model compiled for device='{compiled_device}' (from fhe_device.txt).")
        if compiled_device == "cuda":
            try:
                import concrete.compiler as compiler

                if not compiler.check_gpu_enabled():
                    log(
                        "WARNING: model was compiled for CUDA but Concrete GPU runtime "
                        "is disabled. Install the GPU wheel from https://pypi.zama.ai/gpu"
                    )
            except Exception as exc:
                log(f"WARNING: could not verify GPU runtime: {exc}")

    server = FHEModelServer(model_text)
    server.load()
    client = FHEModelClient(model_text)
    log("Generating evaluation keys...")
    started = time.perf_counter()
    serialized_keys = client.get_serialized_evaluation_keys()
    keygen_seconds = time.perf_counter() - started
    log(
        f"Evaluation keys ready in {keygen_seconds:,.3f}s "
        f"({len(serialized_keys):,} bytes)."
    )
    started = time.perf_counter()
    evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_keys)
    log(f"Evaluation keys deserialized once in {time.perf_counter() - started:,.3f}s.")
    del serialized_keys

    encrypt_times = []
    inference_times = []
    decrypt_times = []
    predictions = []
    report_interval = max(1, count // 10)
    log(f"Measuring encrypt / inference / decrypt on {count:,} records...")
    for index in range(count):
        single = rows[index : index + 1]
        started = time.perf_counter()
        encrypted_input = client.quantize_encrypt_serialize(single)
        encrypt_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        encrypted_output = server.run(encrypted_input, evaluation_keys)
        inference_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        decrypted = client.deserialize_decrypt_dequantize(encrypted_output)
        decrypt_times.append(time.perf_counter() - started)
        predictions.append(label_from_values(flatten_decrypted(decrypted)))
        del encrypted_input, encrypted_output, decrypted

        done = index + 1
        if done % report_interval == 0:
            log(
                f"  {done}/{count} — "
                f"avg encrypt={sum(encrypt_times) / done:.6f}s, "
                f"avg infer={sum(inference_times) / done:.6f}s, "
                f"avg decrypt={sum(decrypt_times) / done:.6f}s"
            )

    del server, client, evaluation_keys
    end_to_end = [
        encrypt + inference + decrypt
        for encrypt, inference, decrypt in zip(encrypt_times, inference_times, decrypt_times)
    ]
    return {
        "n_records": count,
        "keygen_s": keygen_seconds,
        "encrypt_times": encrypt_times,
        "inference_times": inference_times,
        "decrypt_times": decrypt_times,
        "end_to_end_times": end_to_end,
        "predictions": predictions,
        "mean_encrypt_s": sum(encrypt_times) / count,
        "mean_inference_s": sum(inference_times) / count,
        "mean_decrypt_s": sum(decrypt_times) / count,
        "mean_e2e_s": sum(end_to_end) / count,
        "std_encrypt_s": population_std(encrypt_times),
        "std_inference_s": population_std(inference_times),
        "std_decrypt_s": population_std(decrypt_times),
        "std_e2e_s": population_std(end_to_end),
    }


def predict_mode(model, values, mode: str, log: Callable[[str], None], label: str):
    count = int(len(values))
    if mode != "simulate" or count <= SIM_BATCH_ROWS:
        return class_labels(model.predict(values, fhe=mode), count)
    import numpy as np

    out = np.empty(count, dtype=np.int8)
    log(f"{label}: simulate on {count:,} rows...")
    step = max(SIM_BATCH_ROWS, count // 10)
    next_log = step
    for start in range(0, count, SIM_BATCH_ROWS):
        end = min(start + SIM_BATCH_ROWS, count)
        out[start:end] = class_labels(model.predict(values[start:end], fhe="simulate"), end - start)
        if end >= next_log:
            log(f"{label}: simulated {end:,}/{count:,}")
            next_log += step
    return out


def refit_width(screen, matrices, y_binary, family, splits, trees: int, depth: int, n_bits: int):
    model = screen.forest(trees, depth, n_bits)
    started = time.perf_counter()
    model.fit(matrices["train"], y_binary[splits["train"]])
    fit_seconds = time.perf_counter() - started
    scored = {}
    predicted_test = None
    for split_name in ("validation", "test"):
        labels = class_labels(
            model.predict(matrices[split_name], fhe="disable"),
            len(splits[split_name]),
        )
        if split_name == "test":
            predicted_test = labels
        scored[split_name] = screen.score_split(
            y_binary[splits[split_name]],
            family[splits[split_name]],
            labels,
        )
    return model, scored, predicted_test, fit_seconds


def check_reproduction(scored: dict, rows: dict, feature_set: str, n_bits: int) -> list[str]:
    failures = []
    for split_name in ("validation", "test"):
        expected = reference_row(rows, feature_set, split_name, n_bits)
        if expected is None:
            failures.append(f"q{n_bits} {split_name} is not in the source report")
            continue
        observed = scored[split_name]
        for kind, observed_key, expected_key in (
            ("F1", "attack_f1", "f1"),
            ("benign FPR", "benign_fpr", "fpr"),
        ):
            gap = score_gap(float(observed[observed_key]), float(expected[expected_key]))
            if gap is not None:
                failures.append(
                    f"q{n_bits} {split_name} {kind} {float(observed[observed_key]):.4f} "
                    f"differs from the screen {float(expected[expected_key]):.4f} by {gap}"
                )
    return failures


def run_feature_set(
    screen,
    compile_for_device,
    save_fhe_model,
    columns,
    y_binary,
    family,
    splits,
    winner: dict,
    rows: dict[tuple, dict],
    args,
    run_dir: Path,
    sim_index: list[int],
    latency_index: list[int],
    log: Callable[[str], None],
) -> dict:
    import numpy as np

    feature_set = winner["feature_set"]
    trees = int(winner["trees"])
    depth = int(winner["depth"])
    label = f"{feature_set}/trees{trees}_depth{depth}"
    validation_rows = [
        row
        for (phase, name, split, _bits), row in rows.items()
        if phase == "quantized"
        and name == feature_set
        and split == "validation"
        and row["trees"] == trees
        and row["depth"] == depth
        and row["bits"] in BIT_WIDTHS
    ]
    chosen = choose_bit(validation_rows)
    n_bits = int(args.bits) if args.bits is not None else int(chosen["bits"])
    result = {
        "feature_set": feature_set,
        "n_estimators": trees,
        "max_depth": depth,
        "n_bits": n_bits,
        "bit_rule": int(chosen["bits"]),
        "errors": [],
        "reproduction_failures": [],
        "widths": [],
    }
    specs = dict(screen.feature_sets())[feature_set]
    result["columns"] = [spec[1] for spec in specs]
    matrices = screen.scaled_matrices(columns, specs, splits)
    chosen_model = None
    chosen_scores = None
    clear_test_pred = None
    widths = (n_bits,) if args.bits is not None else BIT_WIDTHS
    for width in widths:
        log(f"{label}/q{width}: fitting quantized-clear model on {len(splits['train']):,} rows...")
        try:
            model, scored, test_pred, fit_seconds = refit_width(
                screen, matrices, y_binary, family, splits, trees, depth, width
            )
        except Exception as exc:
            result["errors"].append(f"fit q{width} {type(exc).__name__}: {exc}")
            log(f"{label}/q{width}: FAILED {type(exc).__name__}: {exc}")
            log(traceback.format_exc().rstrip())
            continue
        failures = check_reproduction(scored, rows, feature_set, width)
        result["reproduction_failures"].extend(failures)
        result["widths"].append(
            {
                "bits": width,
                "fit_seconds": fit_seconds,
                "validation_f1": float(scored["validation"]["attack_f1"]),
                "validation_fpr": float(scored["validation"]["benign_fpr"]),
                "test_f1": float(scored["test"]["attack_f1"]),
                "test_fpr": float(scored["test"]["benign_fpr"]),
                "test_precision": float(scored["test"]["attack_precision"]),
                "test_recall": float(scored["test"]["attack_recall"]),
                "test_recalls": {name: float(scored["test"][f"recall_{name}"]) for name in RECALL_NAMES},
            }
        )
        log(
            f"{label}/q{width}: validation F1={scored['validation']['attack_f1']:.4f}, "
            f"test F1={scored['test']['attack_f1']:.4f}, "
            f"test FPR={scored['test']['benign_fpr']:.4f}, fit={fit_seconds:.3f}s"
        )
        for failure in failures:
            log(f"{label}/q{width}: {failure}")
        if width == n_bits:
            chosen_model = model
            chosen_scores = scored
            clear_test_pred = test_pred
        else:
            del model
    if chosen_model is None or chosen_scores is None or clear_test_pred is None:
        result["errors"].append(f"q{n_bits} fit did not produce a model to compile")
        result["gate"], result["gate_reasons"] = decide(result)
        return result

    result["clear_validation_f1"] = float(chosen_scores["validation"]["attack_f1"])
    result["clear_validation_fpr"] = float(chosen_scores["validation"]["benign_fpr"])
    result["clear_test_f1"] = float(chosen_scores["test"]["attack_f1"])
    result["clear_test_fpr"] = float(chosen_scores["test"]["benign_fpr"])
    result["clear_test_precision"] = float(chosen_scores["test"]["attack_precision"])
    result["clear_test_recall"] = float(chosen_scores["test"]["attack_recall"])
    result["clear_test_recalls"] = {
        name: float(chosen_scores["test"][f"recall_{name}"]) for name in RECALL_NAMES
    }
    screen_test = reference_row(rows, feature_set, "test", n_bits)
    if screen_test is not None:
        result["screen_test_f1"] = screen_test["f1"]
        result["screen_test_fpr"] = screen_test["fpr"]
        result["screen_test_recalls"] = {name: screen_test.get(name) for name in RECALL_NAMES}

    log(f"{label}/q{n_bits}: compiling FHE circuit (this can take a long time)...")
    started = time.perf_counter()
    try:
        _circuit, device = compile_for_device(
            chosen_model,
            matrices["train"],
            device=args.device,
            log_fn=log,
            p_error=args.p_error,
        )
    except Exception as exc:
        result["errors"].append(f"compile {type(exc).__name__}: {exc}")
        log(f"{label}/q{n_bits}: compile FAILED {type(exc).__name__}: {exc}")
        log(traceback.format_exc().rstrip())
        result["gate"], result["gate_reasons"] = decide(result)
        del chosen_model
        release_memory()
        return result
    result["compile_seconds"] = time.perf_counter() - started
    result["compiled_device"] = device
    log(f"{label}/q{n_bits}: compiled in {result['compile_seconds']:.1f}s on device={device}.")

    model_dir = run_dir / "fhe" / feature_set / device
    try:
        save_fhe_model(
            str(model_dir),
            chosen_model,
            device,
            meta={
                "feature_set": feature_set,
                "n_estimators": trees,
                "max_depth": depth,
                "n_bits": n_bits,
                "p_error": float(args.p_error),
                "source_run": SOURCE_RUN,
                "compiled_device": device,
                "clear_test_f1": result["clear_test_f1"],
                "clear_test_fpr": result["clear_test_fpr"],
            },
        )
        result["model_dir"] = str(model_dir)
        log(f"{label}/q{n_bits}: saved circuit to '{model_dir}'.")
    except Exception as exc:
        result["errors"].append(f"save {type(exc).__name__}: {exc}")
        log(f"{label}/q{n_bits}: save FAILED {type(exc).__name__}: {exc}")
        log(traceback.format_exc().rstrip())
        model_dir = None

    try:
        sim_values = matrices["test"][sim_index]
        simulated = predict_mode(chosen_model, sim_values, "simulate", log, f"{label}/q{n_bits}")
        sim_family = family[splits["test"]][sim_index]
        sim_truth = y_binary[splits["test"]][sim_index]
        sim_metrics = screen.score_split(sim_truth, sim_family, simulated)
        clear_metrics = screen.score_split(sim_truth, sim_family, clear_test_pred[sim_index])
        result["simulation_rows"] = int(len(sim_index))
        result["simulation_f1"] = float(sim_metrics["attack_f1"])
        result["simulation_fpr"] = float(sim_metrics["benign_fpr"])
        result["simulation_precision"] = float(sim_metrics["attack_precision"])
        result["simulation_recall"] = float(sim_metrics["attack_recall"])
        result["simulation_recalls"] = {name: float(sim_metrics[f"recall_{name}"]) for name in RECALL_NAMES}
        result["simulation_clear_f1"] = float(clear_metrics["attack_f1"])
        result["simulation_clear_fpr"] = float(clear_metrics["benign_fpr"])
        result["simulation_disagree"] = int(np.count_nonzero(simulated != clear_test_pred[sim_index]))
        log(
            f"{label}/q{n_bits}: simulation F1={result['simulation_f1']:.4f}, "
            f"clear F1={result['simulation_clear_f1']:.4f}, "
            f"disagree={result['simulation_disagree']}/{result['simulation_rows']}"
        )
    except Exception as exc:
        result["errors"].append(f"simulation {type(exc).__name__}: {exc}")
        log(f"{label}/q{n_bits}: simulation FAILED {type(exc).__name__}: {exc}")
        log(traceback.format_exc().rstrip())

    del chosen_model
    release_memory()

    if model_dir is not None:
        try:
            latency_rows = np.ascontiguousarray(matrices["test"][latency_index])
            latency = measure_roundtrip(model_dir, latency_rows, log)
            real = np.asarray(latency["predictions"], dtype=np.int8)
            clear_latency = clear_test_pred[latency_index]
            result["real_records"] = int(latency["n_records"])
            result["real_disagree"] = int(np.count_nonzero(real != clear_latency))
            for key in (
                "keygen_s",
                "mean_encrypt_s",
                "mean_inference_s",
                "mean_decrypt_s",
                "mean_e2e_s",
                "std_encrypt_s",
                "std_inference_s",
                "std_decrypt_s",
                "std_e2e_s",
            ):
                result[key] = float(latency[key])
            write_json(
                run_dir / f"latency_{feature_set}.json",
                {
                    "feature_set": feature_set,
                    "n_bits": n_bits,
                    "index": latency_index,
                    "clear_labels": [int(value) for value in clear_latency],
                    "real_labels": [int(value) for value in real],
                    "keygen_s": result["keygen_s"],
                    "mean_inference_s": result["mean_inference_s"],
                    "std_inference_s": result["std_inference_s"],
                    "mean_encrypt_s": result["mean_encrypt_s"],
                    "mean_decrypt_s": result["mean_decrypt_s"],
                    "mean_e2e_s": result["mean_e2e_s"],
                    "encrypt_times": latency["encrypt_times"],
                    "inference_times": latency["inference_times"],
                    "decrypt_times": latency["decrypt_times"],
                    "end_to_end_times": latency["end_to_end_times"],
                },
            )
            log(
                f"{label}/q{n_bits}: mean inference={result['mean_inference_s']:.6f}s "
                f"on {result['real_records']} records "
                f"(encrypt={result['mean_encrypt_s']:.6f}s, "
                f"decrypt={result['mean_decrypt_s']:.6f}s, "
                f"end-to-end={result['mean_e2e_s']:.6f}s), "
                f"real disagreements={result['real_disagree']}."
            )
        except Exception as exc:
            result["errors"].append(f"latency {type(exc).__name__}: {exc}")
            log(f"{label}/q{n_bits}: latency FAILED {type(exc).__name__}: {exc}")
            log(traceback.format_exc().rstrip())

    result["gate"], result["gate_reasons"] = decide(result)
    log(f"{label}/q{n_bits}: gate={result['gate']}")
    for reason in result["gate_reasons"]:
        log(f"{label}/q{n_bits}: {reason}")
    release_memory()
    return result


def format_model(result: dict) -> list[str]:
    bits = result.get("n_bits", "-")
    head = (
        f"{result['feature_set']}  trees={result['n_estimators']} "
        f"depth={result['max_depth']} bits={bits}"
    )
    if result.get("compiled_device"):
        head += f"  device={result['compiled_device']}"
    head += f"  gate={result.get('gate', 'FAIL')}"
    lines = [head]
    if result.get("columns"):
        lines.append("  columns: " + ", ".join(result["columns"]))
    if result.get("bit_rule") is not None and result.get("n_bits") != result.get("bit_rule"):
        lines.append(f"  validation rule chose {result['bit_rule']} bits; this run compiled {result['n_bits']}")
    for width in result.get("widths") or []:
        mark = "*" if width["bits"] == result.get("n_bits") else " "
        lines.append(
            f"  {mark} q{width['bits']} clear validation f1={width['validation_f1']:.4f} "
            f"fpr={width['validation_fpr']:.4f}  test f1={width['test_f1']:.4f} "
            f"fpr={width['test_fpr']:.4f} precision={width['test_precision']:.4f} "
            f"recall={width['test_recall']:.4f}"
        )
    if result.get("clear_test_recalls"):
        lines.append("  clear test recalls " + recall_text(result["clear_test_recalls"], prefix=""))
    if result.get("screen_test_recalls"):
        lines.append("  screen test recalls " + recall_text(result["screen_test_recalls"], prefix=""))
    if result.get("simulation_f1") is not None:
        lines.append(
            f"  simulation rows={result['simulation_rows']} f1={result['simulation_f1']:.4f} "
            f"fpr={result['simulation_fpr']:.4f} precision={result['simulation_precision']:.4f} "
            f"recall={result['simulation_recall']:.4f}  "
            f"clear-on-same-rows f1={result['simulation_clear_f1']:.4f} "
            f"fpr={result['simulation_clear_fpr']:.4f}  "
            f"disagree={result['simulation_disagree']}/{result['simulation_rows']}"
        )
        lines.append("  simulation recalls " + recall_text(result.get("simulation_recalls") or {}, prefix=""))
    if result.get("compile_seconds") is not None:
        lines.append(f"  compile_seconds={result['compile_seconds']:.1f}")
    if result.get("mean_inference_s") is not None:
        lines.append(
            f"  mean inference of {result['real_records']} records="
            f"{result['mean_inference_s']:.6f}s "
            f"std={result['std_inference_s']:.6f}s "
            "(encrypt and decrypt excluded)"
        )
        lines.append(
            f"  mean encrypt={result['mean_encrypt_s']:.6f}s "
            f"std={result['std_encrypt_s']:.6f}s  "
            f"mean decrypt={result['mean_decrypt_s']:.6f}s "
            f"std={result['std_decrypt_s']:.6f}s  "
            f"mean end-to-end={result['mean_e2e_s']:.6f}s "
            f"std={result['std_e2e_s']:.6f}s"
        )
        lines.append(f"  keygen_seconds={result['keygen_s']:.4f} (once, outside both means)")
        lines.append(
            f"  real-fhe label disagreements={result['real_disagree']}/{result['real_records']}"
        )
    for reason in result.get("gate_reasons") or []:
        lines.append(f"  {reason}")
    return lines


def write_report(path: Path, header: list[str], results: list[dict]) -> None:
    lines = list(header)
    lines.append("")
    if not results:
        lines.append("No model has finished.")
    for result in results:
        lines.extend(format_model(result))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, results: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = {field: result.get(field, "") for field in CSV_FIELDS}
            row["error"] = " | ".join(result.get("gate_reasons") or [])
            row["gate"] = result.get("gate", "")
            writer.writerow(row)


def build_header(args, source: Path, counts: dict[str, int], sim_count: int) -> list[str]:
    import os

    sim_text = "every test row" if args.sim_rows == 0 else f"{sim_count} stratified test rows"
    only = "all eight winners" if not args.only else ", ".join(args.only)
    bits = "the validation rule" if args.bits is None else f"{args.bits} bits on every matrix"
    device = args.device or os.environ.get("PPML_FHE_DEVICE") or "auto"
    return [
        "CIC-UNSW-NB15 FHE compile",
        f"source={source}",
        f"seed={args.seed} device={device} p_error={args.p_error}",
        (
            f"train/validation/test={counts['train']}/{counts['validation']}/{counts['test']}. "
            "The split is the screen's stratified row split."
        ),
        "Flows from the same attack can sit on both sides, so these scores are an optimistic screen.",
        f"Models: {only}. Bit width: {bits}.",
        "Clear scores are a new fit at the same seed and shape, predict(..., fhe='disable').",
        f"Simulation scores {sim_text} with fhe='simulate'.",
        (
            f"Inference latency is the mean of the inference phase on {args.fhe_records} "
            "shared test records. Encrypt, decrypt, and their sum are recorded beside it. "
            "Key generation is outside both means."
        ),
        (
            f"A model passes when the refit F1 and benign FPR are within {REPRODUCTION_TOLERANCE:.3f} "
            f"of the source report, clear and simulation F1 are >= {TARGET_F1:.2f}, simulation F1 and "
            f"benign FPR are within {SIMULATION_TOLERANCE:.3f} of the clear score on the same rows, "
            f"and mean inference latency is <= {MAX_INFERENCE_S:.2f}s."
        ),
        "Real-FHE label mismatches on the timed records are printed. They do not change the gate.",
        "",
        "A star marks the compiled bit width.",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=SOURCE_REPORT)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="FHE device. The default follows PPML_FHE_DEVICE, then auto.",
    )
    parser.add_argument("--p-error", type=float, default=P_ERROR)
    parser.add_argument("--bits", type=int, choices=BIT_WIDTHS, default=None)
    parser.add_argument("--only", nargs="+", choices=FEATURE_SETS, default=None)
    parser.add_argument("--fhe-records", type=int, default=100)
    parser.add_argument(
        "--sim-rows",
        type=int,
        default=0,
        help="Test rows for FHE simulation. 0 scores every test row.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("fhe_runs"))
    args = parser.parse_args()
    if not 0 < args.p_error < 1:
        raise SystemExit("--p-error must be between 0 and 1.")
    if args.fhe_records < 1:
        raise SystemExit("--fhe-records must be at least 1.")
    if args.sim_rows < 0:
        raise SystemExit("--sim-rows must be 0 (all test rows) or a positive count.")
    return args


def main() -> None:
    args = parse_args()
    import screen_feature_sets as screen

    names = [name for name, _specs in screen.feature_sets()]
    if names != list(FEATURE_SETS):
        raise SystemExit(f"feature set names changed: {names}")
    if not screen.DATA_PATH.is_file() or not screen.LABEL_PATH.is_file():
        raise SystemExit(
            f"run this from cicunswnb15-models so {screen.DATA_PATH} and {screen.LABEL_PATH} resolve"
        )
    source = resolve_source(args.source)
    winners, rows = parse_report(source.read_text(encoding="utf-8"))
    winner_names = [row["feature_set"] for row in winners]
    if winner_names != list(FEATURE_SETS):
        raise SystemExit(f"source winners are {winner_names}, expected {list(FEATURE_SETS)}")
    selected = [row for row in winners if args.only is None or row["feature_set"] in args.only]
    args.seed = screen.SEED

    compile_for_device, save_fhe_model = import_compile_helpers()
    run_dir = create_run_dir(args.output_root)
    log = make_logger(run_dir)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(
        run_dir / "arguments.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    write_json(run_dir / "status.json", {"status": "running", "started": started})
    log(f"New isolated run directory: {run_dir}")
    log(f"Source report: {source}")

    frame, family = screen.load_frame()
    y_binary = (family != 0).astype(int)
    splits = screen.split_indices(family)
    columns = screen.transformed_columns(frame)
    del frame
    counts = {name: int(len(index)) for name, index in splits.items()}
    log(
        f"Matrices={len(selected)}, bits={list(BIT_WIDTHS) if args.bits is None else [args.bits]}, "
        f"seed={screen.SEED}, train/validation/test="
        f"{counts['train']}/{counts['validation']}/{counts['test']}."
    )
    y_test = y_binary[splits["test"]]
    sim_count = len(y_test) if args.sim_rows == 0 else min(args.sim_rows, len(y_test))
    sim_index = choose_indices(y_test, sim_count, screen.SEED, log)
    latency_index = choose_indices(y_test, args.fhe_records, screen.SEED, log)
    write_json(
        run_dir / "sample_index.json",
        {
            "seed": screen.SEED,
            "latency_index": latency_index,
            "simulation_index": "all" if sim_count == len(y_test) else sim_index,
        },
    )
    header = build_header(args, source, counts, sim_count)
    results: list[dict] = []
    write_report(run_dir / "report.txt", header, results)
    for winner in selected:
        try:
            result = run_feature_set(
                screen,
                compile_for_device,
                save_fhe_model,
                columns,
                y_binary,
                family,
                splits,
                winner,
                rows,
                args,
                run_dir,
                sim_index,
                latency_index,
                log,
            )
        except Exception as exc:
            log(f"{winner['feature_set']}: FAILED {type(exc).__name__}: {exc}")
            log(traceback.format_exc().rstrip())
            result = {
                "feature_set": winner["feature_set"],
                "n_estimators": winner["trees"],
                "max_depth": winner["depth"],
                "n_bits": args.bits,
                "errors": [f"{type(exc).__name__}: {exc}"],
                "reproduction_failures": [],
                "widths": [],
            }
            result["gate"], result["gate_reasons"] = decide(result)
        results.append(result)
        write_report(run_dir / "report.txt", header, results)
        write_csv(run_dir / "results.csv", results)
        release_memory()

    passed = sum(1 for result in results if result.get("gate") == "PASS")
    write_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "started": started,
            "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
            "passed": passed,
            "models": len(results),
        },
    )
    log(f"Experiment complete. {passed}/{len(results)} passed. Read {run_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
