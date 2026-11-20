#!/usr/bin/env python3
"""Plaintext, then quantized, screens for the CIC-UNSW-NB15 forest.

Run from this directory on the dataset machine. This workspace does not
execute the screen. The script does not compile an FHE circuit.
compile_screen_fhe.py compiles the winners.

    python screen_feature_sets.py
    python screen_feature_sets.py --plaintext-only
    python screen_feature_sets.py --quantize-from screen_runs/<timestamp>

Eight feature sets: the four-column core and every subset of the three
add-ons (forward packet length, forward segment size, backward window).
Each set is fit as a plaintext forest at every depth in 3, 4, 5 and every
tree count in 3, 4, 5. The winner of a set is the validation attack F1,
then the lower benign false-positive rate, then the smaller forest. Those
winners are refit at n_bits 8, 7, 6, and 5. Concrete ML 1.9 accepts any
positive bit width; 8 is the tree ciphertext format, and 5 is the floor
of this sweep. Pass --bits-high to walk higher in clear quantization.

Each invocation creates screen_runs/<local timestamp>/. run.log records
every fit as it starts and as it finishes. The csv for that phase grows
by one row per finished score, so a stopped run keeps the fits already done.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SEED = 0
DEPTHS = (3, 4, 5)
TREES = (3, 4, 5)
BITS_HIGH = 8
BITS_LOW = 5
CLIP_AT = 6.0
DATA_PATH = Path("dataset/Data.csv")
LABEL_PATH = Path("dataset/Label.csv")

# name, source column, transform. Transforms are fixed. The min-max is fit
# on the training rows of each feature set.
CORE = (
    ("rate", "Bwd Packets/s", "log1p"),
    ("ratio", "Down/Up Ratio", "clip6"),
    ("iat", "Fwd IAT Mean", "log1p"),
    ("fin", "FIN Flag Count", "identity"),
)
ADDONS = (
    ("length", "Fwd Packet Length Mean", "log1p"),
    ("segment", "Fwd Seg Size Min", "identity"),
    ("window", "Bwd Init Win Bytes", "identity"),
)
COLUMNS = CORE + ADDONS

FAMILY_NAMES = {
    0: "Benign",
    1: "Analysis",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Fuzzers",
    6: "Generic",
    7: "Reconnaissance",
    8: "Shellcode",
    9: "Worms",
}
ATTACK_FAMILIES = [FAMILY_NAMES[code] for code in range(1, 10)]

METRIC_FIELDS = [
    "phase",
    "feature_set",
    "n_features",
    "n_estimators",
    "max_depth",
    "n_bits",
    "split",
    "attack_f1",
    "attack_precision",
    "attack_recall",
    "benign_fpr",
    "fit_seconds",
    "error",
] + [f"recall_{name}" for name in ATTACK_FAMILIES]


def feature_sets() -> list[tuple[str, tuple[tuple[str, str, str], ...]]]:
    sets = [("core", CORE)]
    addon_by_name = {spec[0]: spec for spec in ADDONS}
    names = [spec[0] for spec in ADDONS]
    for width in range(1, len(names) + 1):
        for combo in combinations(names, width):
            label = "core+" + "+".join(combo)
            specs = CORE + tuple(addon_by_name[name] for name in combo)
            sets.append((label, specs))
    return sets


def apply_transform(values: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if kind == "log1p":
        return np.log1p(np.clip(values, 0.0, None))
    if kind == "clip6":
        return np.minimum(values, CLIP_AT)
    if kind == "identity":
        return values
    raise ValueError(f"unknown transform {kind}")


def load_frame() -> tuple[pd.DataFrame, np.ndarray]:
    needed = [spec[1] for spec in COLUMNS]
    frame = pd.read_csv(DATA_PATH, usecols=needed)
    labels = pd.read_csv(LABEL_PATH)
    if len(frame) != len(labels):
        raise SystemExit(
            f"{DATA_PATH} has {len(frame)} rows and {LABEL_PATH} has {len(labels)} rows"
        )
    label_column = labels.columns[0]
    family = labels[label_column].astype(int).to_numpy()
    unknown = sorted(set(np.unique(family)) - set(FAMILY_NAMES))
    if unknown:
        raise SystemExit(f"label codes outside 0-9: {unknown}")
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    return frame, family


def split_indices(family: np.ndarray) -> dict[str, np.ndarray]:
    index = np.arange(len(family))
    train_val, test = train_test_split(
        index, test_size=0.20, random_state=SEED, stratify=family
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=SEED,
        stratify=family[train_val],
    )
    return {"train": train, "validation": validation, "test": test}


def transformed_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        name: apply_transform(frame[source].to_numpy(), kind)
        for name, source, kind in COLUMNS
    }


def scaled_matrices(
    columns: dict[str, np.ndarray],
    specs: tuple[tuple[str, str, str], ...],
    splits: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    stacked = np.column_stack([columns[spec[0]] for spec in specs])
    scaler = MinMaxScaler()
    train = scaler.fit_transform(stacked[splits["train"]])
    validation = np.clip(scaler.transform(stacked[splits["validation"]]), 0.0, 1.0)
    test = np.clip(scaler.transform(stacked[splits["test"]]), 0.0, 1.0)
    return {"train": train, "validation": validation, "test": test}


def forest(n_estimators: int, max_depth: int, n_bits: int | None):
    shared = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=1.0,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    if n_bits is None:
        return RandomForestClassifier(**shared)
    from concrete.ml.sklearn.rf import RandomForestClassifier as ConcreteForest

    return ConcreteForest(n_bits=n_bits, **shared)


def predict_labels(model, values: np.ndarray, quantized: bool) -> np.ndarray:
    if quantized:
        predicted = model.predict(values, fhe="disable")
    else:
        predicted = model.predict(values)
    return np.asarray(predicted).reshape(-1).astype(int)


def score_split(
    y_binary: np.ndarray,
    family: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    truth = y_binary.astype(bool)
    guess = predicted.astype(bool)
    true_positive = int(np.count_nonzero(truth & guess))
    false_negative = int(np.count_nonzero(truth & ~guess))
    false_positive = int(np.count_nonzero(~truth & guess))
    true_negative = int(np.count_nonzero(~truth & ~guess))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    benign = true_negative + false_positive
    row = {
        "attack_f1": f1,
        "attack_precision": precision,
        "attack_recall": recall,
        "benign_fpr": (false_positive / benign) if benign else 0.0,
    }
    for code, name in FAMILY_NAMES.items():
        if code == 0:
            continue
        mask = family == code
        count = int(np.count_nonzero(mask))
        row[f"recall_{name}"] = (
            float(np.count_nonzero(predicted[mask] == 1) / count) if count else float("nan")
        )
    return row


def blank_row() -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in METRIC_FIELDS}
    for name in ATTACK_FAMILIES:
        row[f"recall_{name}"] = ""
    return row


def evaluate(
    model,
    matrices: dict[str, np.ndarray],
    y_binary: np.ndarray,
    family: np.ndarray,
    splits: dict[str, np.ndarray],
    which: tuple[str, ...],
    quantized: bool,
) -> list[dict[str, float]]:
    scored = []
    for split_name in which:
        predicted = predict_labels(model, matrices[split_name], quantized)
        metrics = score_split(y_binary[splits[split_name]], family[splits[split_name]], predicted)
        metrics["split"] = split_name
        scored.append(metrics)
    return scored


def base_fields(feature_set: str, n_features: int, n_estimators: int, max_depth: int, n_bits: int | None, phase: str, fit_seconds: float, error: str) -> dict[str, object]:
    return {
        "phase": phase,
        "feature_set": feature_set,
        "n_features": n_features,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_bits": "" if n_bits is None else n_bits,
        "fit_seconds": f"{fit_seconds:.3f}",
        "error": error,
    }


def run_label(feature_set: str, n_estimators: int, max_depth: int, n_bits: int | None = None) -> str:
    label = f"{feature_set}/trees{n_estimators}_depth{max_depth}"
    if n_bits is not None:
        label = f"{label}/q{n_bits}"
    return label


def metric_line(row: dict[str, object]) -> str:
    if row.get("error"):
        return f"{row['split']} FAILED {row['error']}"
    recalls = " ".join(
        f"{name}={float(row[f'recall_{name}']):.3f}"
        for name in ("Shellcode", "Reconnaissance", "Analysis", "Backdoor", "Worms")
        if row.get(f"recall_{name}") not in ("", None)
    )
    return (
        f"{row['split']} attack F1={float(row['attack_f1']):.4f}, "
        f"benign FPR={float(row['benign_fpr']):.4f}, "
        f"fit={row['fit_seconds']}s, {recalls}"
    )


def record_rows(
    produced: list[dict[str, object]],
    sink: list[dict[str, object]],
    csv_path: Path,
    log: Callable[[str], None],
    label: str,
) -> None:
    sink.extend(produced)
    append_csv(csv_path, produced)
    for row in produced:
        log(f"{label}: {metric_line(row)}")


def run_plaintext(columns, y_binary, family, splits, csv_path: Path, log: Callable[[str], None]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sets = feature_sets()
    total = len(sets) * len(TREES) * len(DEPTHS)
    done = 0
    train_rows = len(splits["train"])
    for feature_set, specs in sets:
        matrices = scaled_matrices(columns, specs, splits)
        for n_estimators in TREES:
            for max_depth in DEPTHS:
                done += 1
                label = run_label(feature_set, n_estimators, max_depth)
                log(f"{label}: fitting plaintext forest {done}/{total} on {train_rows:,} rows...")
                model = forest(n_estimators, max_depth, None)
                started = time.perf_counter()
                model.fit(matrices["train"], y_binary[splits["train"]])
                fit_seconds = time.perf_counter() - started
                shared = base_fields(
                    feature_set, matrices["train"].shape[1], n_estimators, max_depth, None, "plaintext", fit_seconds, ""
                )
                produced = [
                    {**shared, **metrics}
                    for metrics in evaluate(
                        model, matrices, y_binary, family, splits, ("validation",), False
                    )
                ]
                record_rows(produced, rows, csv_path, log, label)
    return rows


def winner_key(row: dict[str, object]) -> tuple:
    return (
        float(row["attack_f1"]),
        -float(row["benign_fpr"]),
        -int(row["n_estimators"]),
        -int(row["max_depth"]),
    )


def select_winners(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    chosen: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["phase"] != "plaintext" or row["split"] != "validation" or row["error"]:
            continue
        name = str(row["feature_set"])
        if name not in chosen or winner_key(row) > winner_key(chosen[name]):
            chosen[name] = row
    order = [name for name, _specs in feature_sets()]
    return [chosen[name] for name in order if name in chosen]


def run_quantized(
    columns,
    y_binary,
    family,
    splits,
    winners: list[dict[str, object]],
    bits: range,
    csv_path: Path,
    log: Callable[[str], None],
) -> list[dict[str, object]]:
    specs_by_name = {name: specs for name, specs in feature_sets()}
    rows: list[dict[str, object]] = []
    jobs = [(winner, n_bits) for winner in winners for n_bits in bits]
    train_rows = len(splits["train"])
    for index, (winner, n_bits) in enumerate(jobs, start=1):
        feature_set = str(winner["feature_set"])
        n_estimators = int(winner["n_estimators"])
        max_depth = int(winner["max_depth"])
        label = run_label(feature_set, n_estimators, max_depth, n_bits)
        log(f"{label}: fitting quantized-clear model {index}/{len(jobs)} on {train_rows:,} rows...")
        matrices = scaled_matrices(columns, specs_by_name[feature_set], splits)
        shared = base_fields(
            feature_set,
            matrices["train"].shape[1],
            n_estimators,
            max_depth,
            n_bits,
            "quantized",
            0.0,
            "",
        )
        try:
            model = forest(n_estimators, max_depth, n_bits)
            started = time.perf_counter()
            model.fit(matrices["train"], y_binary[splits["train"]])
            fit_seconds = time.perf_counter() - started
            shared["fit_seconds"] = f"{fit_seconds:.3f}"
            scored = evaluate(
                model, matrices, y_binary, family, splits, ("validation", "test"), True
            )
        except Exception as exc:
            shared["error"] = f"{type(exc).__name__}: {exc}"
            produced = []
            for split_name in ("validation", "test"):
                failed = blank_row()
                failed.update(shared)
                failed["split"] = split_name
                produced.append(failed)
            record_rows(produced, rows, csv_path, log, label)
            continue
        record_rows([{**shared, **metrics} for metrics in scored], rows, csv_path, log, label)
    return rows


def score_winners_on_test(
    columns,
    y_binary,
    family,
    splits,
    winners: list[dict[str, object]],
    csv_path: Path,
    log: Callable[[str], None],
) -> list[dict[str, object]]:
    specs_by_name = {name: specs for name, specs in feature_sets()}
    rows: list[dict[str, object]] = []
    train_rows = len(splits["train"])
    for winner in winners:
        feature_set = str(winner["feature_set"])
        n_estimators = int(winner["n_estimators"])
        max_depth = int(winner["max_depth"])
        label = run_label(feature_set, n_estimators, max_depth)
        log(f"{label}: refitting plaintext winner for the test split on {train_rows:,} rows...")
        matrices = scaled_matrices(columns, specs_by_name[feature_set], splits)
        model = forest(n_estimators, max_depth, None)
        started = time.perf_counter()
        model.fit(matrices["train"], y_binary[splits["train"]])
        fit_seconds = time.perf_counter() - started
        shared = base_fields(
            feature_set, matrices["train"].shape[1], n_estimators, max_depth, None, "plaintext", fit_seconds, ""
        )
        produced = [
            {**shared, **metrics}
            for metrics in evaluate(model, matrices, y_binary, family, splits, ("test",), False)
        ]
        record_rows(produced, rows, csv_path, log, label)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def make_logger(run_dir: Path) -> Callable[[str], None]:
    log_path = run_dir / "run.log"

    def log(message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def create_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_split_counts(path: Path, family: np.ndarray, splits: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "code", "label", "rows"])
        for split_name, index in splits.items():
            counts = np.bincount(family[index], minlength=10)
            for code, name in FAMILY_NAMES.items():
                writer.writerow([split_name, code, name, int(counts[code])])


def format_row(row: dict[str, object]) -> str:
    bits = row["n_bits"] if row["n_bits"] != "" else "-"
    head = (
        f"{row['phase']:10} {row['split']:10} {str(row['feature_set']):32} "
        f"trees={row['n_estimators']} depth={row['max_depth']} bits={bits}"
    )
    if row.get("error"):
        return f"{head} ERROR {row['error']}"
    recalls = " ".join(
        f"{name}={float(row[f'recall_{name}']):.3f}"
        for name in ("Shellcode", "Reconnaissance", "Analysis", "Backdoor", "Worms")
        if row.get(f"recall_{name}") not in ("", None)
    )
    return (
        f"{head} f1={float(row['attack_f1']):.4f} "
        f"fpr={float(row['benign_fpr']):.4f} {recalls}"
    )


def write_report(path: Path, split_path: Path, winners: list[dict[str, object]], quantized: list[dict[str, object]], test_rows: list[dict[str, object]]) -> None:
    lines = [
        "CIC-UNSW-NB15 feature-set screen",
        f"seed={SEED} depths={list(DEPTHS)} trees={list(TREES)}",
        "Split is stratified on the ten label codes, 60/20/20. It is a row split.",
        "Flows from the same attack can sit on both sides, so these scores are an optimistic screen.",
        "Plaintext validation chooses the depth and tree count. Test rows are scored after that choice.",
        "Quantized rows are a new fit at the same seed and shape, predict(..., fhe='disable').",
        "Bit width for a later compile is chosen from the validation rows, not from the test rows.",
        "",
        "Winners, validation:",
    ]
    lines.extend(format_row(row) for row in winners)
    lines.append("")
    lines.append("Winners, test:")
    lines.extend(format_row(row) for row in test_rows)
    if quantized:
        lines.append("")
        lines.append("Quantized validation and test:")
        lines.extend(format_row(row) for row in quantized if not row.get("error"))
        errors = [row for row in quantized if row.get("error") and row["split"] == "validation"]
        if errors:
            lines.append("")
            lines.append("Quantized fits that failed:")
            lines.extend(format_row(row) for row in errors)
    lines.append("")
    lines.append(f"Family counts: {split_path.name}")
    path.write_text("\n".join(lines) + "\n")


def read_winners(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no winners")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plaintext-only", action="store_true")
    parser.add_argument("--quantize-from", type=Path, default=None, help="run directory that already holds winners.csv")
    parser.add_argument("--bits-high", type=int, default=BITS_HIGH)
    parser.add_argument("--bits-low", type=int, default=BITS_LOW)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.bits_low < 1 or args.bits_high < args.bits_low:
        raise SystemExit("--bits-high must be >= --bits-low, and --bits-low must be >= 1")
    if args.quantize_from and args.plaintext_only:
        raise SystemExit("pass only one of --plaintext-only and --quantize-from")
    return args


def open_run(args: argparse.Namespace) -> tuple[Path, Callable[[str], None]]:
    if args.quantize_from:
        run_dir = args.quantize_from
        if not (run_dir / "winners.csv").is_file():
            raise SystemExit(f"{run_dir} has no winners.csv")
        log = make_logger(run_dir)
        log(f"Resuming quantization in {run_dir}.")
        return run_dir, log
    if args.out is not None:
        run_dir = args.out
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = create_run_dir(Path("screen_runs"))
    log = make_logger(run_dir)
    write_json(
        run_dir / "arguments.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    write_json(
        run_dir / "status.json",
        {"status": "running", "started": datetime.now().astimezone().isoformat(timespec="seconds")},
    )
    log(f"New isolated run directory: {run_dir}")
    return run_dir, log


def main() -> None:
    args = parse_args()
    if not DATA_PATH.is_file() or not LABEL_PATH.is_file():
        raise SystemExit(f"run this from cicunswnb15-models so {DATA_PATH} and {LABEL_PATH} resolve")
    run_dir, log = open_run(args)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    frame, family = load_frame()
    y_binary = (family != 0).astype(int)
    splits = split_indices(family)
    columns = transformed_columns(frame)
    bits = range(args.bits_high, args.bits_low - 1, -1)
    log(
        f"Matrices={len(feature_sets())}, depths={list(DEPTHS)}, trees={list(TREES)}, "
        f"bits={list(bits)}, seed={SEED}, train/validation/test="
        f"{len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])}."
    )

    if args.quantize_from:
        winners = read_winners(run_dir / "winners.csv")
        quantized = run_quantized(
            columns, y_binary, family, splits, winners, bits, run_dir / "quantized.csv", log
        )
        test_path = run_dir / "plaintext_test.csv"
        test_rows = []
        if test_path.is_file():
            with test_path.open(newline="") as handle:
                test_rows = list(csv.DictReader(handle))
        write_report(run_dir / "report.txt", run_dir / "split_counts.csv", winners, quantized, test_rows)
        write_json(
            run_dir / "status.json",
            {
                "status": "complete",
                "started": started,
                "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
        log(f"Quantization complete. Read {run_dir / 'report.txt'}")
        return

    write_split_counts(run_dir / "split_counts.csv", family, splits)
    log(f"Family counts written to {run_dir / 'split_counts.csv'}.")
    plaintext = run_plaintext(
        columns, y_binary, family, splits, run_dir / "plaintext.csv", log
    )
    winners = select_winners(plaintext)
    write_csv(run_dir / "winners.csv", winners)
    for winner in winners:
        log(f"winner {run_label(str(winner['feature_set']), int(winner['n_estimators']), int(winner['max_depth']))}: {metric_line(winner)}")
    test_rows = score_winners_on_test(
        columns, y_binary, family, splits, winners, run_dir / "plaintext_test.csv", log
    )
    quantized: list[dict[str, object]] = []
    if not args.plaintext_only:
        quantized = run_quantized(
            columns, y_binary, family, splits, winners, bits, run_dir / "quantized.csv", log
        )
    write_report(run_dir / "report.txt", run_dir / "split_counts.csv", winners, quantized, test_rows)
    write_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "started": started,
            "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
    log(f"Experiment complete. Read {run_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
