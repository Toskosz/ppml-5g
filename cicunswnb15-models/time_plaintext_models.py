#!/usr/bin/env python3
"""Measure single-record clear (plaintext) inference latency for all 144 CIC-UNSW-NB15 models.

Run from this directory on the dataset machine. This workspace does not
execute the script.

    python time_plaintext_models.py
    python time_plaintext_models.py --bits 5 8 --records 100
    python time_plaintext_models.py --only core core+segment

For each of the 72 (feature_set, trees, depth) shapes and both bit widths
(5 and 8 bits, 144 models total), this script fits the Concrete ML forest at
seed 0 and times ``model.predict(row, fhe="disable")`` one record at a time on
the same 100 stratified test rows used by ``compile_screen_fhe.py``.
Output is written incrementally to ``plaintext_runs/<local timestamp>/``.
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from datetime import datetime
from pathlib import Path

DEFAULT_BITS = (5, 8)
DEFAULT_RECORDS = 100
DEFAULT_WARMUP = 1

CSV_FIELDS = [
    "feature_set",
    "n_estimators",
    "max_depth",
    "n_bits",
    "fit_seconds",
    "clear_test_f1",
    "clear_test_fpr",
    "records",
    "mean_inference_s",
    "std_inference_s",
    "min_inference_s",
    "median_inference_s",
    "p95_inference_s",
    "max_inference_s",
    "mean_inference_ms",
    "error",
]


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def measure_clear_inference(model, rows, warmup: int, population_std) -> dict[str, float]:
    """Time ``model.predict(row, fhe='disable')`` per record."""
    count = int(len(rows))
    if count < 1:
        raise ValueError("latency sample is empty")

    for i in range(warmup):
        model.predict(rows[i % count : (i % count) + 1], fhe="disable")

    times: list[float] = []
    for index in range(count):
        single = rows[index : index + 1]
        started = time.perf_counter()
        model.predict(single, fhe="disable")
        times.append(time.perf_counter() - started)

    ordered = sorted(times)
    mean_s = sum(times) / count
    return {
        "records": count,
        "mean_inference_s": mean_s,
        "std_inference_s": population_std(times),
        "min_inference_s": ordered[0],
        "median_inference_s": percentile(ordered, 0.50),
        "p95_inference_s": percentile(ordered, 0.95),
        "max_inference_s": ordered[-1],
        "mean_inference_ms": mean_s * 1000.0,
    }


def format_result(result: dict) -> list[str]:
    head = (
        f"{result['feature_set']}  trees={result['n_estimators']} "
        f"depth={result['max_depth']} bits={result['n_bits']}"
    )
    if result.get("error"):
        return [head, f"  FAILED: {result['error']}"]
    return [
        head,
        (
            f"  clear test f1={result['clear_test_f1']:.4f} "
            f"fpr={result['clear_test_fpr']:.4f} fit={result['fit_seconds']:.3f}s"
        ),
        (
            f"  mean clear inference of {result['records']} records="
            f"{result['mean_inference_s']:.6f}s ({result['mean_inference_ms']:.3f}ms) "
            f"std={result['std_inference_s']:.6f}s "
            f"min={result['min_inference_s']:.6f}s "
            f"median={result['median_inference_s']:.6f}s "
            f"p95={result['p95_inference_s']:.6f}s "
            f"max={result['max_inference_s']:.6f}s"
        ),
    ]


def write_report(path: Path, header: list[str], results: list[dict]) -> None:
    lines = list(header)
    lines.append("")
    if not results:
        lines.append("No model has finished.")
    for result in results:
        lines.extend(format_result(result))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, results: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    import compile_screen_fhe as compile_fhe

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=list(DEFAULT_BITS),
        help="Bit widths to fit and time in clear mode (default: 5 8).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=compile_fhe.FEATURE_SETS,
        default=None,
        help="Subset of feature sets to run.",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=DEFAULT_RECORDS,
        help="Number of stratified test rows to time (default: 100).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Number of untimed warmup predictions before timing (default: 1).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("plaintext_runs"),
        help="Directory to store timestamped run outputs.",
    )
    args = parser.parse_args()
    if args.records < 1:
        raise SystemExit("--records must be at least 1.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0.")
    if any(b < 1 for b in args.bits):
        raise SystemExit("--bits values must be >= 1.")
    return args


def main() -> None:
    args = parse_args()
    import numpy as np
    import compile_screen_fhe as compile_fhe
    import screen_feature_sets as screen

    if not screen.DATA_PATH.is_file() or not screen.LABEL_PATH.is_file():
        raise SystemExit(
            f"run this from cicunswnb15-models so {screen.DATA_PATH} and {screen.LABEL_PATH} resolve"
        )

    run_dir = compile_fhe.create_run_dir(args.output_root)
    log = compile_fhe.make_logger(run_dir)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    compile_fhe.write_json(
        run_dir / "arguments.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    compile_fhe.write_json(run_dir / "status.json", {"status": "running", "started": started})
    log(f"New isolated run directory: {run_dir}")

    frame, family = screen.load_frame()
    y_binary = (family != 0).astype(int)
    splits = screen.split_indices(family)
    columns = screen.transformed_columns(frame)
    del frame

    counts = {name: int(len(index)) for name, index in splits.items()}
    y_test = y_binary[splits["test"]]
    latency_index = compile_fhe.choose_indices(y_test, args.records, screen.SEED, log)
    compile_fhe.write_json(
        run_dir / "sample_index.json",
        {"seed": screen.SEED, "latency_index": latency_index},
    )

    selected_sets = [
        (name, specs)
        for name, specs in screen.feature_sets()
        if args.only is None or name in args.only
    ]
    total = len(selected_sets) * len(screen.TREES) * len(screen.DEPTHS) * len(args.bits)
    log(
        f"Timing {total} clear models (bits={args.bits}, records={len(latency_index)}, "
        f"warmup={args.warmup}, seed={screen.SEED})."
    )

    header = [
        "CIC-UNSW-NB15 plaintext (clear) inference timing",
        f"seed={screen.SEED} bits={args.bits} records={len(latency_index)} warmup={args.warmup}",
        (
            f"train/validation/test={counts['train']}/{counts['validation']}/{counts['test']}. "
            "Same stratified split and 100-record test sample as compile_screen_fhe.py."
        ),
        "Models are Concrete ML RandomForestClassifier(..., n_bits=b) evaluated with fhe='disable'.",
    ]
    results: list[dict] = []
    write_report(run_dir / "report.txt", header, results)

    done = 0
    for feature_set, specs in selected_sets:
        matrices = screen.scaled_matrices(columns, specs, splits)
        latency_rows = np.ascontiguousarray(matrices["test"][latency_index])

        for n_bits in args.bits:
            for trees in screen.TREES:
                for depth in screen.DEPTHS:
                    done += 1
                    label = f"[{done}/{total}] {feature_set}/trees{trees}_depth{depth}/q{n_bits}"
                    result = {
                        "feature_set": feature_set,
                        "n_estimators": trees,
                        "max_depth": depth,
                        "n_bits": n_bits,
                    }
                    try:
                        model = screen.forest(trees, depth, n_bits)
                        t0 = time.perf_counter()
                        model.fit(matrices["train"], y_binary[splits["train"]])
                        fit_seconds = time.perf_counter() - t0

                        test_pred = compile_fhe.class_labels(
                            model.predict(matrices["test"], fhe="disable"),
                            len(splits["test"]),
                        )
                        scored = screen.score_split(
                            y_binary[splits["test"]],
                            family[splits["test"]],
                            test_pred,
                        )
                        timing = measure_clear_inference(
                            model,
                            latency_rows,
                            args.warmup,
                            compile_fhe.population_std,
                        )
                        result.update(
                            {
                                "fit_seconds": fit_seconds,
                                "clear_test_f1": float(scored["attack_f1"]),
                                "clear_test_fpr": float(scored["benign_fpr"]),
                                **timing,
                                "error": "",
                            }
                        )
                        log(
                            f"{label}: mean={result['mean_inference_ms']:.3f}ms "
                            f"(std={result['std_inference_s'] * 1000.0:.3f}ms, "
                            f"median={result['median_inference_s'] * 1000.0:.3f}ms), "
                            f"F1={result['clear_test_f1']:.4f}, fit={fit_seconds:.3f}s"
                        )
                        del model
                    except Exception as exc:
                        result["error"] = f"{type(exc).__name__}: {exc}"
                        log(f"{label}: FAILED {result['error']}")
                        log(traceback.format_exc().rstrip())

                    results.append(result)
                    write_report(run_dir / "report.txt", header, results)
                    write_csv(run_dir / "results.csv", results)
                    compile_fhe.release_memory()

    failed = sum(1 for r in results if r.get("error"))
    compile_fhe.write_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "started": started,
            "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
            "models": len(results),
            "failed": failed,
        },
    )
    log(
        f"Timing complete: {len(results) - failed}/{len(results)} succeeded. "
        f"Read {run_dir / 'report.txt'}"
    )


if __name__ == "__main__":
    main()
