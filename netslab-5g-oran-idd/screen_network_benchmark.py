"""Accuracy benchmark for the O-CU Zeek table.

This is not the xApp model. Network flows are not on the E2 path, so
--compile-fhe is refused. The protocol is EXPERIMENT.md.

    python screen_network_benchmark.py --csv data/Network_Dataset.csv

A row-random split is refused. Features are reduced to the top 16 and top
32 mutual-information columns before the same radio model grid is scored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from screen_radio_xapp import (
    HIGH_RATE_CATEGORIES,
    LOW_RATE_CATEGORIES,
    RADIO3_CLASS,
    RADIO6_CLASS,
    SEED,
    ScenarioSplitError,
    apply_split,
    assign_scenarios,
    assign_time_gap_scenarios,
    binary_model_configs,
    choose_label_column,
    choose_scenario_column,
    create_run_dir,
    make_logger,
    masks_for,
    multiclass_model_configs,
    normalize_category,
    normalize_key,
    read_sqlite,
    refuse_row_random,
    screen_matrices,
    to_seconds,
    write_json,
    write_tables,
)


IDENTIFIER_KEYS = {"uid", "id", "srcip", "dstip", "saddr", "daddr", "flowid"}
TIMESTAMP_KEYS = {"timestamp", "ts", "starttime", "frametime"}
MI_ROW_CAP = 200_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Zeek flow features without compiling an xApp circuit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default="data/Network_Dataset.csv")
    parser.add_argument(
        "--split",
        choices=("scenario",),
        default="scenario",
        help="Only whole capture scenarios are accepted.",
    )
    parser.add_argument("--scenario-gap-seconds", type=float, default=5.0)
    parser.add_argument("--top-k", nargs="+", type=int, default=(16, 32))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-root", default="network_screen_runs")
    parser.add_argument(
        "--compile-fhe",
        action="store_true",
        help="Refused. Zeek columns are not the E2 features an xApp reads.",
    )
    parser.add_argument("--skip-quantized-clear", action="store_true")
    parser.add_argument("--quantized-bits", nargs="*", type=int, default=None)
    parser.add_argument("--target-f1", type=float, default=0.70)
    parser.add_argument("--max-benign-fpr", type=float, default=0.05)
    parser.add_argument("--max-quant-gap", type=float, default=0.05)
    return parser.parse_args(argv)


def is_identifier(name: object) -> bool:
    key = normalize_key(name)
    return key in IDENTIFIER_KEYS or key.endswith("ip")


def load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Network table not found: {path}")
    if path.suffix.lower() == ".db":
        frame, _info = read_sqlite(path)
        return frame
    return pd.read_csv(path, low_memory=False)


def prepare_network(raw: pd.DataFrame, gap_seconds: float, log) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_column = choose_label_column(raw)
    if label_column is None:
        raise SystemExit(
            "No Benign/Attack label column was found. Columns: "
            + ", ".join(map(str, raw.columns))
        )
    working = raw.drop(columns=[column for column in raw.columns if is_identifier(column)]).copy()
    if label_column not in working.columns:
        working[label_column] = raw[label_column]
    working["category"] = working[label_column].map(normalize_category)
    known = working["category"].isin(("Benign", "DoS", "DDoS", "Probe", "BruteForce", "Web", "Attack"))
    dropped_unknown = int((~known).sum())
    working = working.loc[known].copy()
    working["binary_label"] = (working["category"] != "Benign").astype(np.int8)
    working["radio6_label"] = working["category"].map(RADIO6_CLASS)
    working["radio3_label"] = working["category"].map(RADIO3_CLASS)
    raw_text = working[label_column].astype(str).str.strip()
    coarse = raw_text.map(normalize_key).isin(
        {"benign", "dos", "ddos", "probe", "bruteforce", "web", "attack", "normal", "malicious"}
    )
    working["subtype"] = raw_text.where(~coarse, working["category"])

    timestamp_column = next(
        (column for column in working.columns if normalize_key(column) in TIMESTAMP_KEYS),
        None,
    )
    scenario_column = choose_scenario_column(working, label_column)
    if timestamp_column is not None:
        seconds = to_seconds(working[timestamp_column])
        good_time = seconds.notna()
        working = working.loc[good_time].copy()
        working["timestamp_seconds"] = np.round(seconds.loc[good_time].to_numpy(dtype=np.float64), 3)
        timestamp_note = timestamp_column
    else:
        timestamp_note = "missing"
        working["timestamp_seconds"] = np.nan
    if scenario_column is not None:
        working["scenario_id"] = working[scenario_column].astype(str)
        scenario_source = scenario_column
    elif timestamp_column is not None:
        working["scenario_id"] = assign_time_gap_scenarios(working, gap_seconds)
        scenario_source = "time_gap"
    else:
        raise ScenarioSplitError(
            "Network table has no scenario id and no timestamp. "
            "A row-random split is not used."
        )
    drop_columns = []
    if label_column not in {"category", "scenario_id"}:
        drop_columns.append(label_column)
    if timestamp_column is not None and timestamp_column not in {"scenario_id", "category"}:
        drop_columns.append(timestamp_column)
    if scenario_column is not None and scenario_column not in {"scenario_id", "category"}:
        drop_columns.append(scenario_column)
    working = working.drop(columns=[column for column in drop_columns if column in working.columns])
    working = working.reset_index(drop=True)
    schema = {
        "rows": int(len(working)),
        "label_column": label_column,
        "timestamp": timestamp_note,
        "scenario_source": scenario_source,
        "rows_dropped_unknown_label": dropped_unknown,
        "class_counts": {
            str(name): int(count)
            for name, count in working["category"].value_counts().sort_index().items()
        },
        "scenarios": int(working["scenario_id"].nunique()),
    }
    log(
        f"Loaded {schema['rows']:,} Zeek rows, {schema['scenarios']} scenarios, "
        f"label '{label_column}', scenario source '{scenario_source}'."
    )
    return working, schema


def encode_candidates(frame: pd.DataFrame, train_mask: pd.Series, log) -> tuple[pd.DataFrame, list[str]]:
    reserved = {
        "category",
        "subtype",
        "binary_label",
        "radio3_label",
        "radio6_label",
        "scenario_id",
        "split",
        "timestamp_seconds",
    }
    encoded = pd.DataFrame(index=frame.index)
    logged: list[str] = []
    for column in frame.columns:
        if column in reserved or is_identifier(column):
            continue
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.9:
            train_values = numeric.loc[train_mask]
            fill = float(train_values.median()) if train_values.notna().any() else 0.0
            values = numeric.fillna(fill).to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            train_finite = train_mask.to_numpy() & finite
            if train_finite.any() and float(np.min(values[train_finite])) >= 0:
                median = float(np.median(values[train_finite]))
                tail = float(np.quantile(values[train_finite], 0.99))
                if median > 0 and tail > 50 * median:
                    values = np.log1p(np.clip(values, 0, None))
                    logged.append(str(column))
            encoded[str(column)] = np.where(np.isfinite(values), values, fill)
            continue
        levels = series.loc[train_mask].astype(str)
        distinct = int(levels.nunique(dropna=True))
        if distinct < 2 or distinct > 16:
            continue
        kept_levels = sorted(levels.value_counts().head(16).index)
        text = series.astype(str)
        for level in kept_levels:
            encoded[f"{column}={level}"] = text.eq(level).astype(np.float64)
    if encoded.empty:
        raise SystemExit("No numeric or low-cardinality Zeek features remained after identifier removal.")
    log(f"Candidate features: {encoded.shape[1]:,}. log1p columns: {logged or 'none'}.")
    return encoded, logged


def mutual_information_rank(
    encoded: pd.DataFrame,
    y: np.ndarray,
    train_index: np.ndarray,
) -> list[dict[str, Any]]:
    index = train_index
    if len(index) > MI_ROW_CAP:
        rng = np.random.default_rng(SEED)
        benign = index[y[index] == 0]
        attack = index[y[index] == 1]
        benign_take = min(len(benign), MI_ROW_CAP // 2)
        attack_take = min(len(attack), MI_ROW_CAP - benign_take)
        index = np.sort(
            np.concatenate(
                [
                    rng.choice(benign, size=benign_take, replace=False),
                    rng.choice(attack, size=attack_take, replace=False),
                ]
            )
        )
    scores = mutual_info_classif(
        encoded.to_numpy(dtype=np.float64)[index],
        y[index],
        random_state=SEED,
        discrete_features=False,
    )
    order = np.argsort(scores)[::-1]
    return [
        {"feature": encoded.columns[int(position)], "mutual_information": float(scores[int(position)])}
        for position in order
    ]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    refuse_row_random(argv)
    args = parse_args(argv[1:])
    run_dir = create_run_dir(args.output_root)
    log = make_logger(run_dir)
    write_json(run_dir / "arguments.json", vars(args))
    if args.compile_fhe:
        log("Refusing FHE compilation. Zeek features are not the E2 xApp input.")
    try:
        raw = load_table(Path(args.csv))
        frame, schema = prepare_network(raw, args.scenario_gap_seconds, log)
        split_summary = assign_scenarios(frame)
        frame = apply_split(frame, split_summary["assignment"])
        train_mask, _validation, _test = masks_for(frame, "scenario_holdout")
        encoded, logged = encode_candidates(frame, train_mask, log)
        y = frame["binary_label"].to_numpy(dtype=np.int64)
        ranking = mutual_information_rank(encoded, y, np.flatnonzero(train_mask.to_numpy()))
        write_json(run_dir / "mutual_information.json", ranking)
        write_json(run_dir / "schema.json", {**schema, "log1p_columns": logged})
        write_json(
            run_dir / "split.json",
            {
                "split_counts": split_summary["split_counts"],
                "unheld_categories": split_summary["unheld_categories"],
                "assignment": split_summary["assignment"],
            },
        )
        float_rows: list[dict[str, Any]] = []
        quant_rows: list[dict[str, Any]] = []
        breakdown: list[dict[str, Any]] = []
        for width in args.top_k:
            chosen = [row["feature"] for row in ranking[:width]]
            if len(chosen) < width:
                log(f"Only {len(chosen)} features available for top-{width}.")
            view = frame[
                [
                    "category",
                    "binary_label",
                    "radio3_label",
                    "radio6_label",
                    "scenario_id",
                    "split",
                ]
            ].copy()
            for feature in chosen:
                view[feature] = encoded[feature].to_numpy(dtype=np.float64)
            feature_set = f"mi_top_{width}"
            columns = tuple(chosen)
            binary_configs = binary_model_configs()
            screen_one(
                view, feature_set, columns, "binary_label", binary_configs,
                args, log, run_dir, float_rows, quant_rows, breakdown,
            )
            for task_name, label_column in (("radio3", "radio3_label"), ("radio6", "radio6_label")):
                try:
                    screen_one(
                        view,
                        feature_set,
                        columns,
                        label_column,
                        multiclass_model_configs(task_name),
                        args,
                        log,
                        run_dir,
                        float_rows,
                        quant_rows,
                        breakdown,
                    )
                except ScenarioSplitError as exc:
                    log(f"Skipping {task_name}/{feature_set}: {exc}")
        write_tables(run_dir, float_rows, quant_rows, breakdown)
        lines = [
            "NetsLab Zeek benchmark",
            "======================",
            "This table is not compiled. An xApp reads E2 radio KPIs.",
            f"Rows: {schema['rows']}",
            f"Scenarios: {schema['scenarios']}",
            f"Class counts: {schema['class_counts']}",
            f"High-rate categories: {HIGH_RATE_CATEGORIES}",
            f"Low-rate categories: {LOW_RATE_CATEGORIES}",
            "Top mutual-information features are in mutual_information.json.",
            "Model scores are in results.csv and quantized_results.csv.",
            "FHE compilation: refused.",
        ]
        (run_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"Report written to {run_dir / 'report.txt'}")
        return 0
    except (ScenarioSplitError, SystemExit) as exc:
        message = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, str) else str(exc)
        if isinstance(exc, SystemExit) and not isinstance(exc.code, str):
            raise
        log(str(message))
        write_json(run_dir / "status.json", {"status": "failed", "error": message})
        return 2
    except Exception as exc:
        import traceback

        log(traceback.format_exc())
        write_json(run_dir / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return 1


def screen_one(
    frame: pd.DataFrame,
    feature_set: str,
    columns: tuple[str, ...],
    label_column: str,
    configs: list[dict[str, Any]],
    args: argparse.Namespace,
    log,
    run_dir: Path,
    float_rows: list[dict[str, Any]],
    quant_rows: list[dict[str, Any]],
    breakdown: list[dict[str, Any]],
) -> None:
    train, validation, test = masks_for(frame, "scenario_holdout")
    produced_float, produced_quant, produced_breakdown, _kept = screen_matrices(
        frame,
        feature_set,
        columns,
        "scenario_holdout",
        train,
        validation,
        test,
        configs,
        args,
        log,
        run_dir,
        label_column=label_column,
    )
    float_rows.extend(produced_float)
    quant_rows.extend(produced_quant)
    breakdown.extend(produced_breakdown)
    write_tables(run_dir, float_rows, quant_rows, breakdown)


if __name__ == "__main__":
    sys.exit(main())
