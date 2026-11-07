"""Plaintext and quantized-clear screen for a radio-KPI xApp model.

The protocol is EXPERIMENT.md. Run this on the dataset machine from this
directory. The default command does not compile an FHE circuit:

    python screen_radio_xapp.py --db data/Lower_Layer_Data.db

Task A is binary. Task B scores a 3-class radio grouping and the six
Version 2 categories. A row-random split is refused. --compile-fhe compiles
one load or delta linear model, or one single tree of depth at most 4, after
the binary scenario-holdout gate passes. Wider ensembles and the quantized
MLP are accuracy screens.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import joblib
except ImportError:  # pragma: no cover - present on the dataset machine
    joblib = None


SEED = 42
TARGET_F1 = 0.70
MAX_BENIGN_FPR = 0.05
MAX_QUANT_GAP = 0.05
PREFERRED_E2E_S = 0.25
HARD_E2E_S = 0.70
VERSION2_CATEGORIES = ("Benign", "DoS", "DDoS", "Probe", "BruteForce", "Web")
HIGH_RATE_CATEGORIES = ("DoS", "DDoS", "Probe")
LOW_RATE_CATEGORIES = ("BruteForce", "Web")
RADIO3_CLASS = {
    "Benign": 0,
    **{name: 1 for name in HIGH_RATE_CATEGORIES},
    **{name: 2 for name in LOW_RATE_CATEGORIES},
}
RADIO6_CLASS = {name: index for index, name in enumerate(VERSION2_CATEGORIES)}
PAPER_DROP = ("pucchSnr", "puschSnr", "cqi")
BYTE_COLUMNS = ("dlBytes", "ulBytes")
DELTA_SPECS = (
    ("dlBytes", "d_dlBytes", True),
    ("ulBytes", "d_ulBytes", True),
    ("dlBler", "d_dlBler", False),
    ("ulBler", "d_ulBler", False),
)
LOAD_FEATURES = (
    "dlBytes",
    "ulBytes",
    "dlBler",
    "ulBler",
    "dlMcs",
    "ulMcs",
    "phr",
    "pcmax",
    "ri",
    "pmi",
    "in_sync",
)
CHANNEL_FEATURES = (
    "rsrp",
    "rsrq",
    "rssi",
    "sinr",
    "cqi",
    "pucchSnr",
    "puschSnr",
)
FULL_FEATURES = LOAD_FEATURES + CHANNEL_FEATURES
ID_COLUMNS = ("timestamp", "ue_id", "cellid")

FEATURE_ALIASES = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "frametime": "timestamp",
    "ueid": "ue_id",
    "rnti": "ue_id",
    "cellid": "cellid",
    "cell": "cellid",
    "cqi": "cqi",
    "dlcqi": "cqi",
    "dlmcs": "dlMcs",
    "ulmcs": "ulMcs",
    "dlbytes": "dlBytes",
    "ulbytes": "ulBytes",
    "dlbler": "dlBler",
    "ulbler": "ulBler",
    "rsrp": "rsrp",
    "rsrq": "rsrq",
    "rssi": "rssi",
    "sinr": "sinr",
    "pcmax": "pcmax",
    "pucchsnr": "pucchSnr",
    "puschsnr": "puschSnr",
    "phr": "phr",
    "powerheadroom": "phr",
    "ri": "ri",
    "rankindicator": "ri",
    "pmi": "pmi",
    "insync": "in_sync",
    "syncstatus": "in_sync",
}
LABEL_ALIASES = {
    "label",
    "class",
    "category",
    "attack",
    "attacktype",
    "attackcategory",
    "attackname",
    "target",
    "type",
}
SUBTYPE_ALIASES = {
    "attacksubcategory",
    "subcategory",
    "attacksubtype",
    "subtype",
}
SCENARIO_ALIASES = {
    "sourcefile",
    "filename",
    "file",
    "filepath",
    "path",
    "source",
    "scenario",
    "scenarioid",
    "capture",
    "captureid",
    "trace",
    "experiment",
}
VERSION_ALIASES = {"version", "datasetversion", "release"}
LEGACY_TEXT = re.compile(r"legacy|version\s*1|\bv1\b", re.IGNORECASE)
BOOL_TEXT = {
    "true": 1.0,
    "false": 0.0,
    "yes": 1.0,
    "no": 0.0,
    "y": 1.0,
    "n": 0.0,
    "in": 1.0,
    "out": 0.0,
    "in_sync": 1.0,
    "insync": 1.0,
    "out_of_sync": 0.0,
    "outofsync": 0.0,
}


class ScenarioSplitError(RuntimeError):
    """Raised when whole scenarios cannot form train, validation, and test."""


def normalize_key(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def normalize_category(value: object) -> str:
    text = normalize_key(value)
    if text in {"", "nan", "none", "null"}:
        return "Unknown"
    if text in {"benign", "normal", "background"} or text.startswith("benign"):
        return "Benign"
    if "ddos" in text or "distributeddenial" in text:
        return "DDoS"
    if "dos" in text or "slowloris" in text or "flood" in text:
        return "DoS"
    if any(token in text for token in ("probe", "scan", "nmap", "masscan", "xprobe", "fingerprint")):
        return "Probe"
    if any(token in text for token in ("brute", "hydra", "medusa", "ssh", "ftp")):
        return "BruteForce"
    if any(token in text for token in ("web", "sql", "xss", "gobuster", "dvwa", "http")):
        return "Web"
    if text in {"attack", "malicious", "1"}:
        return "Attack"
    return "Unknown"


def refuse_row_random(argv: list[str]) -> None:
    """Stop before argparse when the caller asks to shuffle or subsample rows."""
    if any(arg in {"-h", "--help"} for arg in argv):
        return
    normalized = [arg.lower().replace("_", "-") for arg in argv[1:]]
    for index, arg in enumerate(normalized):
        if arg == "--sample-rows" or arg.startswith("--sample-rows="):
            raise SystemExit(
                "Refusing a row sample. The screen uses every row in the "
                "selected scenarios. See EXPERIMENT.md."
            )
        if arg in {"--random-split", "--row-random", "--shuffle-split"}:
            raise SystemExit(
                "Refusing a row-random split. Capture scenarios stay intact. "
                "See EXPERIMENT.md."
            )
        if "random" in arg and "split" in arg:
            raise SystemExit(
                "Refusing a row-random split. Capture scenarios stay intact. "
                "See EXPERIMENT.md."
            )
        if arg in {"--split", "--split-strategy"}:
            nxt = normalized[index + 1] if index + 1 < len(normalized) else ""
            if nxt != "scenario":
                raise SystemExit(
                    "Refusing a row-random split. The only split is "
                    "--split scenario. See EXPERIMENT.md."
                )
        if arg.startswith("--split=") or arg.startswith("--split-strategy="):
            value = arg.split("=", 1)[1]
            if value != "scenario":
                raise SystemExit(
                    "Refusing a row-random split. The only split is "
                    "--split scenario. See EXPERIMENT.md."
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen radio-KPI models for a one-second xApp loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--db",
        default=None,
        help="Path to Lower_Layer_Data.db. Default when --lower-layer-dir is omitted.",
    )
    source.add_argument(
        "--lower-layer-dir",
        default=None,
        help="Directory of per-scenario lower_layer .txt/.csv files. One file is one scenario.",
    )
    parser.add_argument(
        "--split",
        choices=("scenario",),
        default="scenario",
        help="Only whole capture scenarios are accepted.",
    )
    parser.add_argument("--scenario-gap-seconds", type=float, default=5.0)
    parser.add_argument(
        "--quantized-bits",
        nargs="*",
        type=int,
        default=None,
        help="Override every model's bit list. The default is the per-model list in EXPERIMENT.md.",
    )
    parser.add_argument("--target-f1", type=float, default=TARGET_F1)
    parser.add_argument("--max-benign-fpr", type=float, default=MAX_BENIGN_FPR)
    parser.add_argument("--max-quant-gap", type=float, default=MAX_QUANT_GAP)
    parser.add_argument(
        "--skip-quantized-clear",
        action="store_true",
        help="Float models only. Quantized-clear scores and FHE compilation are skipped.",
    )
    parser.add_argument(
        "--compile-fhe",
        action="store_true",
        help="Compile the smallest eligible load or delta model (linear, or one tree of depth <= 4).",
    )
    parser.add_argument(
        "--compile-wide",
        action="store_true",
        help="Also allow the single depth-6 tree when it passes the binary gate.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--p-error", type=float, default=0.01)
    parser.add_argument("--fhe-records", type=int, default=100)
    parser.add_argument(
        "--fhe-sim-rows",
        type=int,
        default=5000,
        help="Stratified FHE-simulation rows. 0 uses every scenario-holdout test row.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-root", default="screen_runs")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Check scenario assignment and category mapping, then exit.",
    )
    args = parser.parse_args(argv)
    if not args.self_check and args.db is None and args.lower_layer_dir is None:
        args.db = "data/Lower_Layer_Data.db"
    return args


def create_run_dir(output_root: str) -> Path:
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_logger(run_dir: Path):
    log_path = run_dir / "run.log"

    def log(message: str) -> None:
        line = f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def feature_hit_count(columns: Any) -> int:
    found = set()
    for column in columns:
        canonical = FEATURE_ALIASES.get(normalize_key(column))
        if canonical in FULL_FEATURES or canonical in ID_COLUMNS:
            found.add(canonical)
    return len(found)


def rename_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Map known aliases onto canonical names.

    A table that already has the canonical column keeps it. Extra aliases,
    such as ``rnti`` beside ``ue_id``, are dropped so they are not features
    and are not collapsed onto the identifier. Two aliases and no canonical
    column still stop the run.
    """
    grouped: dict[str, list[str]] = {}
    for column in frame.columns:
        canonical = FEATURE_ALIASES.get(normalize_key(column))
        if canonical is None:
            continue
        grouped.setdefault(canonical, []).append(str(column))

    mapping: dict[str, str] = {}
    ignored: list[str] = []
    for canonical, columns in grouped.items():
        if len(columns) == 1:
            mapping[columns[0]] = canonical
            continue
        exact_key = normalize_key(canonical)
        exact = [column for column in columns if normalize_key(column) == exact_key]
        if len(exact) != 1:
            names = "', '".join(columns)
            raise SystemExit(
                f"Columns '{names}' all map to {canonical}, "
                "and more than one could be that field."
            )
        mapping[exact[0]] = canonical
        ignored.extend(column for column in columns if column != exact[0])

    renamed = frame.rename(columns=mapping)
    if ignored:
        renamed = renamed.drop(columns=ignored)
    return renamed, mapping, ignored


def coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(np.float64)
    numeric = pd.to_numeric(series, errors="coerce")
    if float(numeric.notna().mean()) >= 0.9:
        return numeric.astype(np.float64)
    text = series.astype(str).str.strip().str.lower().map(BOOL_TEXT)
    filled = numeric.fillna(text)
    if float(filled.notna().mean()) >= 0.9:
        return filled.astype(np.float64)
    # OAI writes PMI as a pair, for example "(0, 0)". The feature is the first number.
    pairs = series.astype(str).str.extract(
        r"^\s*[\[\(]\s*([+-]?\d+(?:\.\d+)?)",
        expand=False,
    )
    filled = filled.fillna(pd.to_numeric(pairs, errors="coerce"))
    return filled.astype(np.float64)


def to_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        median = float(numeric.dropna().median())
        if median > 1e14:
            return numeric / 1e9
        if median > 1e11:
            return numeric / 1e3
        return numeric
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    seconds = pd.Series(np.nan, index=series.index, dtype=np.float64)
    ok = parsed.notna()
    if ok.any():
        seconds.loc[ok] = parsed.loc[ok].astype("int64").to_numpy() / 1e9
    return seconds


def choose_label_column(frame: pd.DataFrame) -> str | None:
    banned = set(FULL_FEATURES) | set(ID_COLUMNS) | {"folder_category", "source_file"}
    best_name = None
    best_score = (-1.0, -1, -1)
    for column in frame.columns:
        if column in banned:
            continue
        key = normalize_key(column)
        series = frame[column]
        values = series.astype(str)
        categories = values.map(normalize_category)
        known = categories.isin(VERSION2_CATEGORIES + ("Attack",))
        fraction = float(known.mean()) if len(series) else 0.0
        distinct_categories = int(categories[known].nunique())
        distinct_raw = int(values[known].nunique())
        alias_bonus = 0.05 if key in LABEL_ALIASES else 0.0
        if fraction < 0.5 or distinct_categories < 2:
            continue
        score = (fraction + alias_bonus, distinct_categories, distinct_raw)
        if score > best_score:
            best_score = score
            best_name = str(column)
    return best_name


def choose_subtype_column(frame: pd.DataFrame, label_column: str | None) -> str | None:
    banned = set(FULL_FEATURES) | set(ID_COLUMNS)
    if label_column is not None:
        banned.add(label_column)
    for column in frame.columns:
        if str(column) in banned:
            continue
        if normalize_key(column) in SUBTYPE_ALIASES:
            return str(column)
    return None


def choose_scenario_column(frame: pd.DataFrame, label_column: str | None) -> str | None:
    banned = set(FULL_FEATURES) | set(ID_COLUMNS)
    if label_column is not None:
        banned.add(label_column)
    candidates: list[tuple[int, str]] = []
    for column in frame.columns:
        if column in banned:
            continue
        if normalize_key(column) not in SCENARIO_ALIASES:
            continue
        series = frame[column].astype(str)
        distinct = int(series.nunique(dropna=True))
        if distinct < 2 or distinct > 500:
            continue
        median_rows = float(series.value_counts().median())
        if median_rows < 5:
            continue
        candidates.append((distinct, str(column)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def drop_legacy_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = pd.Series(False, index=frame.index)
    matched: list[str] = []
    for column in frame.columns:
        key = normalize_key(column)
        if key in VERSION_ALIASES:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            version_hit = numeric.eq(1)
            mask = mask | version_hit.fillna(False)
            if version_hit.any():
                matched.append(str(column))
        if key not in VERSION_ALIASES | SCENARIO_ALIASES | LABEL_ALIASES:
            continue
        if not (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
        ):
            continue
        text_hit = frame[column].astype(str).str.contains(LEGACY_TEXT, na=False)
        mask = mask | text_hit
        if text_hit.any():
            matched.append(str(column))
    kept = frame.loc[~mask].copy()
    return kept, {
        "legacy_rows_removed": int(mask.sum()),
        "legacy_columns": sorted(set(matched)),
        "legacy_filter": "applied" if matched else "not_identifiable",
    }


def read_sqlite(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Radio database not found: {path}")
    connection = sqlite3.connect(path)
    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            connection,
        )
        best_frame = None
        best_name = None
        best_hits = -1
        for name in tables["name"].tolist():
            quoted = str(name).replace('"', '""')
            frame = pd.read_sql(f'SELECT * FROM "{quoted}"', connection)
            hits = feature_hit_count(frame.columns)
            if hits > best_hits:
                best_frame = frame
                best_name = str(name)
                best_hits = hits
    finally:
        connection.close()
    if best_frame is None:
        raise SystemExit(f"No tables found in {path}")
    info = {"source": str(path), "sqlite_table": best_name, "feature_hits": best_hits}
    return best_frame, info


def read_text_table(path: Path) -> pd.DataFrame:
    head = path.read_bytes()[:400].lstrip()
    if head.startswith(b"{") or head.startswith(b"["):
        frame = pd.read_json(path, lines=head.startswith(b"{"))
        if feature_hit_count(frame.columns) > 0:
            return frame
    attempts = (
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": "\t"},
        {"sep": r"\s+", "engine": "python"},
    )
    last_error = "no readable header"
    for kwargs in attempts:
        try:
            frame = pd.read_csv(path, **kwargs)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if feature_hit_count(frame.columns) >= 8:
            return frame
    raise SystemExit(
        f"Unrecognized radio table '{path}' ({last_error}). "
        "Expected a header row with the published PHY/MAC names."
    )


def category_from_path(path: Path) -> str | None:
    for part in path.parts:
        if LEGACY_TEXT.search(part):
            return None
        mapped = normalize_category(part)
        if mapped in VERSION2_CATEGORIES:
            return mapped
    return None


def read_lower_layer_dir(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not root.is_dir():
        raise SystemExit(f"Lower-layer directory not found: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".csv"}
    )
    frames: list[pd.DataFrame] = []
    skipped_legacy = 0
    for path in files:
        if any(LEGACY_TEXT.search(part) for part in path.parts):
            skipped_legacy += 1
            continue
        category = category_from_path(path)
        if category is None:
            continue
        frame = read_text_table(path)
        frame["source_file"] = str(path.relative_to(root))
        frame["folder_category"] = category
        frames.append(frame)
    if not frames:
        raise SystemExit(
            f"No Version 2 lower_layer tables under {root}. "
            "Expected .txt or .csv files inside Benign/DoS/DDoS/Probe/BruteForce/Web."
        )
    combined = pd.concat(frames, ignore_index=True)
    info = {
        "source": str(root),
        "scenario_files": len(frames),
        "legacy_files_skipped": skipped_legacy,
    }
    return combined, info


def prepare_frame(raw: pd.DataFrame, source_info: dict[str, Any], gap_seconds: float, log) -> tuple[pd.DataFrame, dict[str, Any]]:
    renamed, mapping, ignored_aliases = rename_columns(raw)
    if ignored_aliases:
        log(
            "Dropped identifier aliases already covered by a canonical column: "
            + ", ".join(ignored_aliases)
            + "."
        )
    missing = [name for name in FULL_FEATURES if name not in renamed.columns]
    if missing:
        raise SystemExit(
            "Radio table is missing required KPIs: "
            + ", ".join(missing)
            + ". Columns present: "
            + ", ".join(map(str, renamed.columns))
        )
    filtered, legacy_info = drop_legacy_rows(renamed)
    label_column = choose_label_column(filtered)
    if label_column is None and "folder_category" not in filtered.columns:
        columns = ", ".join(map(str, filtered.columns))
        raise SystemExit("No Benign/Attack label column was found. Columns: " + columns)
    scenario_column = "source_file" if "source_file" in filtered.columns else None
    if scenario_column is None:
        scenario_column = choose_scenario_column(filtered, label_column)

    working = filtered.copy()
    for column in FULL_FEATURES:
        working[column] = coerce_numeric(working[column])
    finite = np.ones(len(working), dtype=bool)
    negative_bytes = 0
    nonfinite_columns: list[str] = []
    for column in FULL_FEATURES:
        values = working[column].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(values)
        if bad.any():
            nonfinite_columns.append(f"{column} ({int(bad.sum())})")
        finite &= ~bad
        if column in BYTE_COLUMNS:
            negative = np.isfinite(values) & (values < 0)
            negative_bytes += int(negative.sum())
            clipped = values.copy()
            clipped[negative] = 0.0
            working[column] = clipped
    dropped_nonfinite = int((~finite).sum())
    working = working.iloc[np.flatnonzero(finite)].copy()

    folder_label_disagreements = 0
    if "folder_category" in working.columns:
        working["category"] = working["folder_category"].map(normalize_category)
        if label_column is not None:
            from_column = working[label_column].map(normalize_category)
            comparable = from_column.isin(VERSION2_CATEGORIES)
            folder_label_disagreements = int(
                (comparable & from_column.ne(working["category"])).sum()
            )
            raw_text = working[label_column].astype(str).str.strip()
        else:
            raw_text = working["category"].astype(str)
            label_column = "folder_category"
    else:
        raw_text = working[label_column].astype(str).str.strip()
        working["category"] = raw_text.map(normalize_category)
    unknown = ~working["category"].isin(VERSION2_CATEGORIES + ("Attack",))
    dropped_unknown = int(unknown.sum())
    working = working.loc[~unknown].copy()
    raw_text = raw_text.loc[working.index]
    working["binary_label"] = (working["category"] != "Benign").astype(np.int8)
    working["radio6_label"] = working["category"].map(RADIO6_CLASS)
    working["radio3_label"] = working["category"].map(RADIO3_CLASS)
    coarse = raw_text.map(normalize_key).isin(
        {"benign", "dos", "ddos", "probe", "bruteforce", "web", "attack", "normal", "malicious"}
    )
    # A file stem or a fine attack name is the subtype. A bare category is not.
    subtype_column = choose_subtype_column(working, label_column)
    if subtype_column is not None:
        working["subtype"] = working[subtype_column].astype(str).str.strip()
        empty_subtype = working["subtype"].map(normalize_key).isin({"", "nan", "none", "null"})
        working.loc[empty_subtype, "subtype"] = working.loc[empty_subtype, "category"]
    elif scenario_column == "source_file":
        working["subtype"] = (
            raw_text.where(~coarse, working["source_file"].map(lambda path: Path(str(path)).stem))
        )
    else:
        working["subtype"] = raw_text.where(~coarse, working["category"])

    timestamp_note = "parsed"
    if "timestamp" in working.columns:
        seconds = to_seconds(working["timestamp"])
        if float(seconds.notna().mean()) < 0.5:
            timestamp_note = "synthetic_row_order"
            seconds = pd.Series(np.nan, index=working.index)
    else:
        timestamp_note = "missing_synthetic_row_order"
        seconds = pd.Series(np.nan, index=working.index)
    if "ue_id" not in working.columns:
        working["ue_id"] = "unknown"
        ue_note = "missing_constant"
    else:
        working["ue_id"] = working["ue_id"].astype(str)
        ue_note = "present"
    if timestamp_note == "parsed":
        bad_time = seconds.isna()
        dropped_time = int(bad_time.sum())
        working = working.loc[~bad_time].copy()
        seconds = seconds.loc[~bad_time]
        working["timestamp_seconds"] = np.round(seconds.to_numpy(dtype=np.float64), 3)
    else:
        dropped_time = 0
        working = working.sort_values(["category", "subtype", "ue_id"], kind="mergesort")
        working["timestamp_seconds"] = working.groupby(["category", "subtype", "ue_id"], sort=False).cumcount().astype(float)

    if scenario_column is not None:
        working["scenario_id"] = working[scenario_column].astype(str)
        scenario_source = scenario_column
    else:
        working["scenario_id"] = assign_time_gap_scenarios(working, gap_seconds)
        scenario_source = "time_gap"

    working = working.reset_index(drop=True)
    schema = {
        **source_info,
        **legacy_info,
        "renamed_columns": mapping,
        "ignored_alias_columns": ignored_aliases,
        "label_column": label_column,
        "scenario_source": scenario_source,
        "timestamp": timestamp_note,
        "ue_id": ue_note,
        "rows_after_legacy_filter": int(len(filtered)),
        "rows_dropped_nonfinite": dropped_nonfinite,
        "nonfinite_columns": nonfinite_columns,
        "subtype_column": subtype_column,
        "negative_byte_values_clipped": negative_bytes,
        "rows_dropped_unknown_label": dropped_unknown,
        "folder_label_disagreements": folder_label_disagreements,
        "rows_dropped_bad_timestamp": dropped_time,
        "rows": int(len(working)),
        "class_counts": {
            str(name): int(count)
            for name, count in working["category"].value_counts().sort_index().items()
        },
        "binary_counts": {
            "benign": int((working["binary_label"] == 0).sum()),
            "attack": int((working["binary_label"] == 1).sum()),
        },
        "radio3_rows": int(working["radio3_label"].notna().sum()),
        "radio6_rows": int(working["radio6_label"].notna().sum()),
        "scenarios": int(working["scenario_id"].nunique()),
    }
    log(
        f"Loaded {schema['rows']:,} rows, {schema['scenarios']} scenarios, "
        f"label column '{label_column}', scenario source '{scenario_source}'."
    )
    log(f"Class counts: {schema['class_counts']}")
    if schema["binary_counts"]["benign"] == 0 or schema["binary_counts"]["attack"] == 0:
        detail = (
            f"rows after legacy filter={schema['rows_after_legacy_filter']}, "
            f"dropped nonfinite={schema['rows_dropped_nonfinite']}, "
            f"dropped unknown label={schema['rows_dropped_unknown_label']}, "
            f"dropped bad timestamp={schema['rows_dropped_bad_timestamp']}"
        )
        if nonfinite_columns:
            detail += ". Non-finite columns: " + ", ".join(nonfinite_columns)
        raise SystemExit("Need both benign and attack rows after cleaning. " + detail)
    return working, schema


def scenario_gap_threshold(times: np.ndarray, gap_seconds: float) -> float:
    """Separate captures without splitting a slow but continuous sample.

    A series whose samples are about a second apart uses ``gap_seconds``.
    A slower series, such as DoS rows a few seconds apart, starts a new run
    only after a gap of at least 15 times its median spacing.
    """
    if len(times) < 2:
        return gap_seconds
    spacing = np.diff(np.sort(times.astype(np.float64)))
    median = float(np.median(spacing))
    if not np.isfinite(median) or median <= 1.5:
        return gap_seconds
    return max(gap_seconds, 15.0 * median)


def assign_time_gap_scenarios(frame: pd.DataFrame, gap_seconds: float) -> pd.Series:
    scenario = pd.Series(index=frame.index, dtype=object)
    for (category, subtype), group in frame.groupby(["category", "subtype"], sort=False):
        safe_subtype = re.sub(r"\s+", "_", str(subtype)).replace("|", "/")
        times = np.unique(group["timestamp_seconds"].to_numpy(dtype=np.float64))
        times.sort()
        threshold = scenario_gap_threshold(times, gap_seconds)
        run_of_time: dict[float, int] = {}
        run = 0
        previous = None
        for stamp in times:
            if previous is not None and float(stamp - previous) > threshold:
                run += 1
            run_of_time[float(stamp)] = run
            previous = float(stamp)
        scenario.loc[group.index] = [
            f"{category}|{safe_subtype}|run{run_of_time[float(stamp)]:04d}"
            for stamp in group["timestamp_seconds"].to_numpy(dtype=np.float64)
        ]
    return scenario


def assign_scenarios(frame: pd.DataFrame) -> dict[str, Any]:
    grouped = (
        frame.groupby(["scenario_id", "category"], sort=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    if grouped["scenario_id"].duplicated().any():
        raise ScenarioSplitError(
            "A scenario id maps to more than one category. Refusing to split it."
        )
    assignment: dict[str, str] = {}
    by_category: dict[str, list[str]] = {}
    for category, group in grouped.groupby("category", sort=False):
        ordered = sorted(group["scenario_id"].astype(str).tolist())
        by_category[str(category)] = ordered
        count = len(ordered)
        if count >= 3:
            for scenario_id in ordered[:-2]:
                assignment[scenario_id] = "train"
            assignment[ordered[-2]] = "validation"
            assignment[ordered[-1]] = "test"
        elif count == 2:
            assignment[ordered[0]] = "train"
            assignment[ordered[1]] = "test"
        else:
            assignment[ordered[0]] = "train"
    if "validation" not in assignment.values():
        raise ScenarioSplitError(
            "No category has three or more scenarios, so there is no validation "
            "scenario to leave intact. Pass --lower-layer-dir with one file per "
            "capture. A row-random split is not used."
        )
    if "test" not in assignment.values():
        raise ScenarioSplitError(
            "No scenario was held out for test. Pass --lower-layer-dir with one "
            "file per capture."
        )
    splits = set(assignment.values())
    if splits != {"train", "validation", "test"} and "train" not in splits:
        raise ScenarioSplitError("Scenario assignment produced no training scenarios.")
    unheld = sorted(
        category
        for category, scenarios in by_category.items()
        if not any(assignment[scenario_id] == "test" for scenario_id in scenarios)
    )
    summary = {
        "assignment": assignment,
        "by_category": by_category,
        "unheld_categories": unheld,
        "split_counts": {
            split: int(sum(1 for value in assignment.values() if value == split))
            for split in ("train", "validation", "test")
        },
    }
    return summary


def apply_split(frame: pd.DataFrame, assignment: dict[str, str]) -> pd.DataFrame:
    mapped = frame["scenario_id"].map(assignment)
    if mapped.isna().any():
        missing = sorted(set(frame.loc[mapped.isna(), "scenario_id"].astype(str)))
        raise ScenarioSplitError(f"Scenarios missing from the assignment: {missing[:10]}")
    assigned = frame.copy()
    assigned["split"] = mapped.astype(str)
    overlap = (
        assigned.groupby("scenario_id")["split"].nunique().loc[lambda counts: counts > 1]
    )
    if not overlap.empty:
        raise ScenarioSplitError(
            "A scenario landed in more than one split: "
            + ", ".join(overlap.index.astype(str).tolist())
        )
    return assigned


def attach_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["scenario_id", "ue_id", "timestamp_seconds"],
        kind="mergesort",
    ).copy()
    grouped = ordered.groupby(["scenario_id", "ue_id"], sort=False)
    previous_time = grouped["timestamp_seconds"].shift(1)
    adjacent = (ordered["timestamp_seconds"] - previous_time).between(0.5, 1.5)
    for source, destination, use_log in DELTA_SPECS:
        current = ordered[source].to_numpy(dtype=np.float64)
        previous = grouped[source].shift(1).to_numpy(dtype=np.float64)
        if use_log:
            change = np.log1p(current) - np.log1p(previous)
        else:
            change = current - previous
        valid = adjacent.to_numpy() & np.isfinite(change)
        ordered[destination] = np.where(valid, change, 0.0)
    return ordered.sort_index()


def design_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    blocks = []
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float64)
        if column in BYTE_COLUMNS:
            values = np.log1p(np.clip(values, 0, None))
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values remain in {column}.")
        blocks.append(values)
    return np.column_stack(blocks)


def scale_matrix(matrix: np.ndarray, train_index: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaler.fit(matrix[train_index])
    scaled = scaler.transform(matrix)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32), scaler


def mutual_information(
    matrix: np.ndarray, y: np.ndarray, names: tuple[str, ...]
) -> tuple[list[dict[str, Any]], bool]:
    scores = mutual_info_classif(matrix, y, random_state=SEED, discrete_features=False)
    order = np.argsort(scores)
    ranking = [
        {"feature": names[int(index)], "mutual_information": float(scores[int(index)])}
        for index in order[::-1]
    ]
    lowest = {names[int(index)] for index in order[: len(PAPER_DROP)]}
    return ranking, lowest == set(PAPER_DROP)


def model_config(
    model_id: str,
    family: str,
    bits: tuple[int, ...],
    *,
    C: float | None = None,
    max_depth: int | None = None,
    n_estimators: int | None = None,
    compile_shape: bool = False,
    binary_only: bool = False,
) -> dict[str, Any]:
    return {
        "id": model_id,
        "family": family,
        "task": "binary",
        "bits": bits,
        "compile_shape": compile_shape,
        "binary_only": binary_only,
        "params": {"C": C, "max_depth": max_depth, "n_estimators": n_estimators},
    }


def binary_model_configs() -> list[dict[str, Any]]:
    """Task A grid. Bit widths are the quantized-clear settings for that shape."""
    return [
        model_config("lr_c1", "lr", (6, 8), C=1.0, compile_shape=True),
        model_config("lr_c10", "lr", (6,), C=10.0, compile_shape=True),
        model_config("linear_svc", "linear_svc", (6, 8), C=1.0, compile_shape=True),
        model_config("tree_depth_3", "tree", (6,), max_depth=3, n_estimators=1, compile_shape=True),
        model_config("tree_depth_4", "tree", (5, 6), max_depth=4, n_estimators=1, compile_shape=True),
        model_config("tree_depth_6", "tree", (6,), max_depth=6, n_estimators=1, compile_shape=False),
        model_config("rf_10_depth_3", "forest", (5,), max_depth=3, n_estimators=10),
        model_config("rf_20_depth_4", "forest", (5,), max_depth=4, n_estimators=20),
        model_config(
            "xgb_10_depth_3", "xgb", (5,), max_depth=3, n_estimators=10, binary_only=True
        ),
        model_config(
            "xgb_20_depth_4", "xgb", (5,), max_depth=4, n_estimators=20, binary_only=True
        ),
        model_config("qmlp_32", "mlp", (4,)),
    ]


def multiclass_model_configs(task: str) -> list[dict[str, Any]]:
    """Task B. XGBClassifier is omitted because multi-class uses one-vs-rest."""
    selected = []
    for config in binary_model_configs():
        if config["binary_only"] or config["family"] == "xgb":
            continue
        if config["family"] not in {"tree", "forest", "mlp", "lr"}:
            continue
        item = {
            **config,
            "task": task,
            "compile_shape": False,
            "params": dict(config["params"]),
        }
        selected.append(item)
    return selected


def with_task(configs: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    tagged = []
    for config in configs:
        item = {**config, "task": task, "params": dict(config["params"])}
        tagged.append(item)
    return tagged


def instantiate_float(config: dict[str, Any], jobs: int, n_features: int):
    family = config["family"]
    params = config["params"]
    if family == "lr":
        return LogisticRegression(
            C=params["C"],
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=SEED,
        )
    if family == "linear_svc":
        from sklearn.svm import LinearSVC

        return LinearSVC(
            C=params["C"],
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )
    if family == "tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"], random_state=SEED)
    if family == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=0.3,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=jobs,
            random_state=SEED,
            use_label_encoder=False,
            verbosity=0,
        )
    if family == "mlp":
        from sklearn.neural_network import MLPClassifier

        width = max(8, int(round(1.5 * n_features)))
        return MLPClassifier(
            hidden_layer_sizes=(width,),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=30,
            random_state=SEED,
            early_stopping=False,
        )
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=SEED,
        n_jobs=jobs,
    )


def instantiate_quantized(config: dict[str, Any], n_bits: int, jobs: int):
    family = config["family"]
    params = config["params"]
    if family == "lr":
        from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression

        return ConcreteLogisticRegression(
            n_bits=n_bits,
            C=params["C"],
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=SEED,
        )
    if family == "linear_svc":
        from concrete.ml.sklearn import LinearSVC as ConcreteLinearSVC

        return ConcreteLinearSVC(
            n_bits=n_bits,
            C=params["C"],
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )
    if family == "tree":
        from concrete.ml.sklearn import DecisionTreeClassifier as ConcreteDecisionTreeClassifier

        return ConcreteDecisionTreeClassifier(
            n_bits=n_bits,
            max_depth=params["max_depth"],
            random_state=SEED,
        )
    if family == "xgb":
        from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier

        return ConcreteXGBClassifier(
            n_bits=n_bits,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=0.3,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=jobs,
            random_state=SEED,
            use_label_encoder=False,
            verbosity=0,
        )
    if family == "mlp":
        from concrete.ml.sklearn import NeuralNetClassifier as ConcreteNeuralNetClassifier

        return ConcreteNeuralNetClassifier(
            module__n_layers=2,
            module__n_hidden_neurons_multiplier=1.5,
            module__n_w_bits=n_bits,
            module__n_a_bits=n_bits,
            module__n_accum_bits=8,
            max_epochs=30,
            batch_size=256,
            lr=1e-3,
            train_split=None,
            callbacks="disable",
            verbose=0,
            device="cpu",
        )
    from concrete.ml.sklearn.rf import RandomForestClassifier as ConcreteRandomForestClassifier

    return ConcreteRandomForestClassifier(
        n_bits=n_bits,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=SEED,
    )


def model_call(model, method_name: str, values: np.ndarray, quantized: bool):
    method = getattr(model, method_name)
    if quantized:
        try:
            return method(values, fhe="disable")
        except TypeError:
            return method(values)
    return method(values)


def positive_scores(model, values: np.ndarray, quantized: bool) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(model_call(model, "predict_proba", values, quantized))
        except Exception:
            probabilities = None
        if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] == 2:
            classes = list(getattr(model, "classes_", [0, 1]))
            column = classes.index(1) if 1 in classes else 1
            return probabilities[:, column].astype(np.float64)
    scores = np.asarray(model_call(model, "decision_function", values, quantized))
    if scores.ndim == 2:
        scores = scores[:, -1]
    return scores.ravel().astype(np.float64)


def predicted_classes(model, values: np.ndarray, quantized: bool) -> np.ndarray:
    return np.asarray(model_call(model, "predict", values, quantized)).ravel()


def choose_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        raise RuntimeError("Validation split has a single class, so F1 cannot choose a threshold.")
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.nanargmax(f1_values))
    return float(thresholds[best]), float(f1_values[best])


def safe_metric(function, y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(function(y_true, scores))


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (scores >= threshold).astype(np.int8)
    benign = y_true == 0
    false_positive_rate = float(prediction[benign].mean()) if benign.any() else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "average_precision": safe_metric(average_precision_score, y_true, scores),
        "roc_auc": safe_metric(roc_auc_score, y_true, scores),
        "benign_fpr": false_positive_rate,
        "benign_support": int(benign.sum()),
    }


def quality_met(metrics: dict[str, float], target_f1: float, max_fpr: float) -> bool:
    fpr = metrics["benign_fpr"]
    return bool(
        metrics["f1"] >= target_f1
        and metrics["benign_support"] > 0
        and not math.isnan(fpr)
        and fpr <= max_fpr
    )


def multiclass_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    benign = y_true == 0
    false_positive_rate = float((predicted[benign] != 0).mean()) if benign.any() else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "average_precision": float("nan"),
        "roc_auc": float("nan"),
        "benign_fpr": false_positive_rate,
        "benign_support": int(benign.sum()),
    }


def per_category_multiclass(
    categories: np.ndarray,
    predicted: np.ndarray,
    task: str,
    feature_set: str,
    model_id: str,
    mode: str,
    protocol: str,
    n_bits: int | None,
) -> list[dict[str, Any]]:
    mapping = RADIO3_CLASS if task == "radio3" else RADIO6_CLASS
    rows = []
    for label, expected in mapping.items():
        mask = categories == label
        if not mask.any():
            continue
        if label == "Benign":
            rate = float((predicted[mask] != 0).mean())
            metric_name = "false_positive_rate"
        else:
            rate = float((predicted[mask] == expected).mean())
            metric_name = "detection_recall"
        rows.append(
            {
                "protocol": protocol,
                "task": task,
                "feature_set": feature_set,
                "model_id": model_id,
                "mode": mode,
                "n_bits": n_bits,
                "category": label,
                "support": int(mask.sum()),
                "predicted_attack_rate": rate,
                "metric_name": metric_name,
            }
        )
    return rows


def per_category_rows(
    categories: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    feature_set: str,
    model_id: str,
    mode: str,
    protocol: str,
    n_bits: int | None,
) -> list[dict[str, Any]]:
    prediction = scores >= threshold
    present = [str(label) for label in pd.unique(categories)]
    ordered = [label for label in VERSION2_CATEGORIES if label in present]
    ordered.extend(sorted(set(present) - set(ordered)))
    rows = []
    for label in ordered:
        mask = categories == label
        rate = float(prediction[mask].mean()) if mask.any() else float("nan")
        rows.append(
            {
                "protocol": protocol,
                "feature_set": feature_set,
                "model_id": model_id,
                "mode": mode,
                "n_bits": n_bits,
                "category": label,
                "support": int(mask.sum()),
                "predicted_attack_rate": rate,
                "metric_name": "false_positive_rate" if label == "Benign" else "detection_recall",
                "task": "binary",
            }
        )
    return rows


def result_row(
    config: dict[str, Any],
    feature_set: str,
    protocol: str,
    mode: str,
    n_bits: int | None,
    y_train: np.ndarray,
    metrics: dict[str, float],
    default_metrics: dict[str, float],
    threshold: float,
    validation_f1: float,
    fit_seconds: float,
    input_dimensions: int,
    target_f1: float,
    max_fpr: float,
) -> dict[str, Any]:
    params = config["params"]
    return {
        "status": "ok",
        "protocol": protocol,
        "feature_set": feature_set,
        "model_id": config["id"],
        "family": config["family"],
        "mode": mode,
        "n_bits": n_bits,
        "C": params["C"],
        "max_depth": params["max_depth"],
        "n_estimators": params["n_estimators"],
        "input_dimensions": input_dimensions,
        "train_rows": int(len(y_train)),
        "train_benign": int((y_train == 0).sum()),
        "train_attack": int((y_train == 1).sum()),
        "fit_seconds": fit_seconds,
        "selected_threshold": threshold,
        "validation_f1": validation_f1,
        "test_f1_default_0_5": default_metrics["f1"],
        "test_f1": metrics["f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_accuracy": metrics["accuracy"],
        "test_average_precision": metrics["average_precision"],
        "test_roc_auc": metrics["roc_auc"],
        "benign_fpr": metrics["benign_fpr"],
        "benign_support": metrics["benign_support"],
        "quality_target_met": quality_met(metrics, target_f1, max_fpr),
        "task": config.get("task", "binary"),
        "compile_shape": bool(config.get("compile_shape", False)),
    }


def fit_one(
    config: dict[str, Any],
    feature_set: str,
    protocol: str,
    quantized: bool,
    n_bits: int | None,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_categories: np.ndarray,
    args: argparse.Namespace,
    log,
):
    import time

    task = config.get("task", "binary")
    multiclass = task != "binary"
    model = (
        instantiate_quantized(config, int(n_bits), args.jobs)
        if quantized
        else instantiate_float(config, args.jobs, int(x_train.shape[1]))
    )
    label = f"{protocol}/{task}/{feature_set}/{config['id']}"
    if quantized:
        label += f"/q{n_bits}"
    log(f"{label}: fitting on {len(y_train):,} rows...")
    started = time.perf_counter()
    model.fit(x_train, y_train.astype(np.int64))
    fit_seconds = time.perf_counter() - started
    if multiclass:
        validation_pred = predicted_classes(model, x_validation, quantized)
        test_pred = predicted_classes(model, x_test, quantized)
        metrics = multiclass_metrics(y_test, test_pred)
        validation_f1 = float(f1_score(y_validation, validation_pred, average="macro", zero_division=0))
        threshold = float("nan")
        default_metrics = metrics
        breakdown = per_category_multiclass(
            test_categories,
            test_pred,
            task,
            feature_set,
            config["id"],
            "quantized_clear" if quantized else "float",
            protocol,
            n_bits,
        )
    else:
        validation_scores = positive_scores(model, x_validation, quantized)
        threshold, validation_f1 = choose_f1_threshold(y_validation, validation_scores)
        test_scores = positive_scores(model, x_test, quantized)
        metrics = classification_metrics(y_test, test_scores, threshold)
        default_metrics = classification_metrics(y_test, test_scores, 0.5)
        breakdown = per_category_rows(
            test_categories,
            test_scores,
            threshold,
            feature_set,
            config["id"],
            "quantized_clear" if quantized else "float",
            protocol,
            n_bits,
        )
    row = result_row(
        config,
        feature_set,
        protocol,
        "quantized_clear" if quantized else "float",
        n_bits,
        y_train,
        metrics,
        default_metrics,
        threshold,
        validation_f1,
        fit_seconds,
        int(x_train.shape[1]),
        args.target_f1,
        args.max_benign_fpr,
    )
    log(
        f"{label}: F1={row['test_f1']:.4f}, benign FPR={row['benign_fpr']:.4f}, "
        f"threshold={threshold}."
    )
    return row, breakdown, model


def error_row(config: dict[str, Any], feature_set: str, protocol: str, n_bits: int | None, exc: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "protocol": protocol,
        "feature_set": feature_set,
        "model_id": config["id"],
        "family": config["family"],
        "n_bits": n_bits,
        "error": f"{type(exc).__name__}: {exc}",
        "quality_target_met": False,
        "compile_eligible": False,
    }


def indices_for(mask: pd.Series) -> np.ndarray:
    return np.flatnonzero(mask.to_numpy())


def save_joblib(payload: Any, path: Path, log) -> None:
    """Save a fitted model when the runtime can pickle it.

    A Concrete ML quantized tree carries a local lambda in its ONNX forward
    pass. joblib cannot store that object. The screen scores are already
    recorded, so a failed save must not stop the run.
    """
    if joblib is None:
        return
    try:
        joblib.dump(payload, path)
    except Exception as exc:
        log(f"Could not save {path.name}: {type(exc).__name__}: {exc}")


def screen_matrices(
    frame: pd.DataFrame,
    feature_set: str,
    columns: tuple[str, ...],
    protocol: str,
    train_mask: pd.Series,
    validation_mask: pd.Series,
    test_mask: pd.Series,
    configs: list[dict[str, Any]],
    args: argparse.Namespace,
    log,
    run_dir: Path,
    label_column: str = "binary_label",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], Any]]:
    matrix = design_matrix(frame, columns)
    label_values = pd.to_numeric(frame[label_column], errors="coerce").to_numpy(dtype=np.float64)
    valid = pd.Series(np.isfinite(label_values), index=frame.index)
    train_index = indices_for(train_mask & valid)
    validation_index = indices_for(validation_mask & valid)
    test_index = indices_for(test_mask & valid)
    for name, index in ("train", train_index), ("validation", validation_index), ("test", test_index):
        if len(index) == 0 or len(np.unique(label_values[index])) < 2:
            raise ScenarioSplitError(
                f"{protocol}/{feature_set}/{label_column}: the {name} split has a single class. "
                "Held-out scenarios must include both benign and attack traffic. "
                "A row-random split is not used."
            )
    scaled, scaler = scale_matrix(matrix, train_index)
    x_train = scaled[train_index]
    x_validation = scaled[validation_index]
    x_test = scaled[test_index]
    y_train = label_values[train_index].astype(np.int64)
    y_validation = label_values[validation_index].astype(np.int64)
    y_test = label_values[test_index].astype(np.int64)
    categories = frame["category"].to_numpy(dtype=str)[test_index]
    task_name = configs[0].get("task", "binary") if configs else "binary"
    model_dir = run_dir / "models" / protocol / task_name / feature_set
    model_dir.mkdir(parents=True, exist_ok=True)
    save_joblib(
        {"scaler": scaler, "columns": columns, "feature_set": feature_set},
        model_dir / "scaler.joblib",
        log,
    )

    float_rows: list[dict[str, Any]] = []
    quant_rows: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    kept: dict[tuple[str, int], Any] = {}
    float_by_id: dict[str, dict[str, Any]] = {}
    for config in configs:
        if config.get("binary_only") and config.get("task", "binary") != "binary":
            float_rows.append(
                {
                    **error_row(
                        config,
                        feature_set,
                        protocol,
                        None,
                        RuntimeError(
                            "Skipped. Multi-class XGBClassifier evaluates one tree "
                            "per class per boosting round."
                        ),
                    ),
                    "status": "skipped",
                    "task": config.get("task", "binary"),
                }
            )
            continue
        try:
            row, rows, model = fit_one(
                config,
                feature_set,
                protocol,
                False,
                None,
                x_train,
                y_train,
                x_validation,
                y_validation,
                x_test,
                y_test,
                categories,
                args,
                log,
            )
        except Exception as exc:
            log(f"{protocol}/{feature_set}/{config['id']} failed: {exc}")
            log(traceback.format_exc())
            float_rows.append(error_row(config, feature_set, protocol, None, exc))
            continue
        float_rows.append(row)
        float_by_id[config["id"]] = row
        breakdown.extend(rows)
        save_joblib(model, model_dir / f"{config['id']}.joblib", log)

    if args.skip_quantized_clear:
        return float_rows, quant_rows, breakdown, kept

    for config in configs:
        if config.get("binary_only") and config.get("task", "binary") != "binary":
            continue
        partner = float_by_id.get(config["id"])
        bit_list = list(args.quantized_bits) if args.quantized_bits else list(config["bits"])
        for n_bits in bit_list:
            try:
                row, rows, model = fit_one(
                    config,
                    feature_set,
                    protocol,
                    True,
                    int(n_bits),
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    x_test,
                    y_test,
                    categories,
                    args,
                    log,
                )
            except Exception as exc:
                log(f"{protocol}/{feature_set}/{config['id']}/q{n_bits} failed: {exc}")
                log(traceback.format_exc())
                quant_rows.append(error_row(config, feature_set, protocol, int(n_bits), exc))
                continue
            if partner is None or partner.get("status") != "ok":
                gap = float("nan")
                gap_ok = False
                float_ok = False
            else:
                gap = abs(float(row["test_f1"]) - float(partner["test_f1"]))
                gap_ok = gap <= args.max_quant_gap
                float_ok = bool(partner["quality_target_met"])
            row["float_test_f1"] = None if partner is None else partner.get("test_f1")
            row["f1_gap"] = gap
            row["quantized_gap_ok"] = gap_ok
            shape_ok = bool(config.get("compile_shape")) or (
                bool(getattr(args, "compile_wide", False)) and config["id"] == "tree_depth_6"
            )
            row["compile_eligible"] = bool(
                shape_ok
                and config.get("task", "binary") == "binary"
                and protocol == "scenario_holdout"
                and feature_set in {"load", "delta"}
                and row["quality_target_met"]
                and gap_ok
                and float_ok
            )
            quant_rows.append(row)
            breakdown.extend(rows)
            if row["compile_eligible"]:
                kept[(config["id"], int(n_bits))] = {
                    "model": model,
                    "x_train": x_train,
                    "x_test": x_test,
                    "y_test": y_test,
                    "categories": categories,
                    "result": row,
                }
            save_joblib(
                {"model": model, "threshold": row["selected_threshold"], "result": row},
                model_dir / f"{config['id']}_q{n_bits}.joblib",
                log,
            )
    return float_rows, quant_rows, breakdown, kept


def misses_float_gate(float_rows: list[dict[str, Any]], feature_set: str) -> bool:
    rows = [
        row
        for row in float_rows
        if row.get("feature_set") == feature_set
        and row.get("protocol") == "scenario_holdout"
        and row.get("task", "binary") == "binary"
        and row.get("status") == "ok"
    ]
    return not any(row.get("quality_target_met") for row in rows)


def masks_for(frame: pd.DataFrame, protocol: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    if protocol == "scenario_holdout":
        return (
            frame["split"].eq("train"),
            frame["split"].eq("validation"),
            frame["split"].eq("test"),
        )
    held_out = "Web" if protocol == "family_holdout_web" else "BruteForce"
    train = frame["split"].eq("train") & frame["category"].ne(held_out)
    validation = frame["split"].eq("validation") & frame["category"].ne(held_out)
    test = frame["category"].eq(held_out) | (
        frame["category"].eq("Benign") & frame["split"].eq("test")
    )
    if bool((train & test).any()):
        raise ScenarioSplitError(f"{protocol} leaked training rows into the score set.")
    return train, validation, test


def write_tables(run_dir: Path, float_rows: list[dict], quant_rows: list[dict], breakdown: list[dict]) -> None:
    pd.DataFrame(float_rows).to_csv(run_dir / "results.csv", index=False)
    pd.DataFrame(quant_rows).to_csv(run_dir / "quantized_results.csv", index=False)
    pd.DataFrame(breakdown).to_csv(run_dir / "category_recall.csv", index=False)
    write_json(run_dir / "results.json", float_rows)
    write_json(run_dir / "quantized_results.json", quant_rows)
    write_json(run_dir / "category_recall.json", breakdown)


def select_compile_target(quant_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in quant_rows if row.get("compile_eligible")]

    def rank(row: dict[str, Any]) -> tuple:
        return (
            0 if row["feature_set"] == "load" else 1,
            {"lr": 0, "linear_svc": 1, "tree": 2}.get(row["family"], 9),
            int(row["max_depth"] or 0),
            int(row["n_estimators"] or 1),
            int(row["n_bits"]),
            {"lr_c1": 0, "lr_c10": 1, "linear_svc": 2}.get(row["model_id"], 3),
        )

    if not eligible:
        return None
    return sorted(eligible, key=rank)[0]


def simulated_positive_scores(model, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(model.predict_proba(values, fhe="simulate"))
        except Exception:
            probabilities = None
        if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            classes = list(getattr(model, "classes_", [0, 1]))
            column = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
            return probabilities[:, column].astype(np.float64)
    scores = np.asarray(model.decision_function(values, fhe="simulate"))
    if scores.ndim == 2:
        scores = scores[:, -1]
    return scores.ravel().astype(np.float64)


def simulation_indices(y: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or limit >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(SEED)
    chosen = []
    for label in (0, 1):
        pool = np.flatnonzero(y == label)
        if len(pool) == 0:
            continue
        take = min(len(pool), max(1, int(round(limit * len(pool) / len(y)))))
        chosen.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def compile_target(bundle: dict[str, Any], args: argparse.Namespace, run_dir: Path, log) -> dict[str, Any]:
    import time

    cic_dir = Path(__file__).resolve().parents[1] / "cicids2018-kdd-models"
    sys.path.insert(0, str(cic_dir))
    from fhe_latency import measure_fhe_roundtrip
    from gpu_utils import compile_for_device, save_fhe_model

    result = bundle["result"]
    model = bundle["model"]
    log(
        f"Compiling {result['feature_set']}/{result['model_id']}/q{result['n_bits']} "
        f"on device={args.device}."
    )
    started = time.perf_counter()
    _, compiled_device = compile_for_device(
        model,
        bundle["x_train"],
        device=args.device,
        log_fn=log,
        p_error=args.p_error,
    )
    compile_seconds = time.perf_counter() - started
    model_dir = run_dir / "fhe" / f"{result['feature_set']}_{result['model_id']}_q{result['n_bits']}" / compiled_device
    save_fhe_model(
        str(model_dir),
        model,
        compiled_device,
        meta={
            "feature_set": result["feature_set"],
            "model_id": result["model_id"],
            "n_bits": result["n_bits"],
            "selected_threshold": result["selected_threshold"],
            "quantized_clear_f1": result["test_f1"],
            "quantized_clear_benign_fpr": result["benign_fpr"],
        },
    )
    y_test = bundle["y_test"]
    chosen = simulation_indices(y_test, args.fhe_sim_rows)
    started = time.perf_counter()
    simulation_scores = simulated_positive_scores(model, bundle["x_test"][chosen])
    simulation_seconds = time.perf_counter() - started
    simulation_metrics = classification_metrics(
        y_test[chosen],
        simulation_scores,
        float(result["selected_threshold"]),
    )
    rng = np.random.default_rng(SEED)
    latency_order = rng.permutation(len(bundle["x_test"]))
    latency = measure_fhe_roundtrip(
        str(model_dir),
        bundle["x_test"][latency_order],
        n_records=min(args.fhe_records, len(bundle["x_test"])),
        log_fn=log,
    )
    mean_e2e = float(latency["mean_e2e_s"])
    summary = {
        "status": "ok",
        "feature_set": result["feature_set"],
        "model_id": result["model_id"],
        "n_bits": result["n_bits"],
        "compiled_device": compiled_device,
        "compile_seconds": compile_seconds,
        "simulation_rows": int(len(chosen)),
        "simulation_seconds": simulation_seconds,
        "simulation_f1": simulation_metrics["f1"],
        "simulation_benign_fpr": simulation_metrics["benign_fpr"],
        "selected_threshold": result["selected_threshold"],
        "keygen_s": latency["keygen_s"],
        "mean_encrypt_s": latency["mean_encrypt_s"],
        "mean_inference_s": latency["mean_inference_s"],
        "mean_decrypt_s": latency["mean_decrypt_s"],
        "mean_e2e_s": mean_e2e,
        "preferred_latency_met": mean_e2e <= PREFERRED_E2E_S,
        "hard_latency_met": mean_e2e <= HARD_E2E_S,
        "simulation_quality_met": quality_met(
            simulation_metrics, args.target_f1, args.max_benign_fpr
        ),
    }
    summary["circuit_pass"] = bool(
        summary["simulation_quality_met"] and summary["hard_latency_met"]
    )
    log(
        f"FHE simulation F1={summary['simulation_f1']:.4f}, "
        f"mean end-to-end={mean_e2e:.4f}s, circuit_pass={summary['circuit_pass']}."
    )
    return summary


def claim_text(breakdown: list[dict[str, Any]], float_rows: list[dict[str, Any]]) -> str:
    passing = {
        (row["feature_set"], row["model_id"])
        for row in float_rows
        if row.get("protocol") == "scenario_holdout"
        and row.get("task", "binary") == "binary"
        and row.get("status") == "ok"
        and row.get("quality_target_met")
    }
    if not passing:
        return (
            "No float model cleared the scenario-holdout gate. "
            "Per-category recall is in category_recall.csv."
        )
    weak = []
    for row in breakdown:
        key = (row.get("feature_set"), row.get("model_id"))
        if key not in passing or row.get("protocol") != "scenario_holdout":
            continue
        if row.get("task", "binary") != "binary":
            continue
        if row.get("mode") != "float" or row.get("metric_name") != "detection_recall":
            continue
        if row.get("category") not in {"Web", "BruteForce"} or int(row.get("support") or 0) < 20:
            continue
        rate = row.get("predicted_attack_rate")
        if rate is not None and not (isinstance(rate, float) and math.isnan(rate)) and float(rate) < 0.10:
            weak.append(f"{row['feature_set']}/{row['model_id']} {row['category']}")
    if weak:
        return (
            "These passing float models have Web or BruteForce recall below 0.10: "
            + ", ".join(weak)
            + ". Describe them as flood/probe detection on radio KPIs."
        )
    return (
        "Passing float models are not below 0.10 Web or BruteForce recall "
        "at support >= 20. Confirm the rows in category_recall.csv before a broader claim."
    )


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "None."
    frame = pd.DataFrame(rows)
    present = [column for column in columns if column in frame.columns]
    view = frame[present].copy()
    for column in present:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return view.to_string(index=False)


def write_report(
    run_dir: Path,
    args: argparse.Namespace,
    schema: dict[str, Any],
    split_summary: dict[str, Any],
    mi_ranking: list[dict[str, Any]],
    paper_drop_agrees: bool,
    float_rows: list[dict[str, Any]],
    quant_rows: list[dict[str, Any]],
    breakdown: list[dict[str, Any]],
    fhe_result: dict[str, Any] | None,
    delta_ran: bool,
    channel_confound: bool,
) -> None:
    scenario_float = [row for row in float_rows if row.get("protocol") == "scenario_holdout"]
    lines = [
        "NetsLab radio xApp screen",
        "=========================",
        f"Run directory: {run_dir}",
        f"Generated: {dt.datetime.now().astimezone().isoformat()}",
        f"Arguments: {json.dumps(vars(args), sort_keys=True, default=str)}",
        "",
        "Schema",
        "------",
    ]
    for key, value in schema.items():
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "Scenario split",
            "--------------",
            f"Train/validation/test scenarios: {split_summary['split_counts']}",
            f"Categories without a held-out scenario: {split_summary['unheld_categories'] or 'none'}",
            "Scenario ids are sorted within each category. With three or more, the last is test and the second-to-last is validation.",
            "",
            "Mutual information on training scenarios",
            "---------------------------------------",
        ]
    )
    lines.append(format_table(mi_ranking, ["feature", "mutual_information"]))
    lines.append(
        "Paper drop of pucchSnr, puschSnr, and cqi agrees with the lowest three."
        if paper_drop_agrees
        else "Paper drop of pucchSnr, puschSnr, and cqi does not match the lowest three. full_reduced was not trained."
    )
    lines.extend(["", "Float scenario holdout", "----------------------"])
    lines.append(
        format_table(
            scenario_float,
            [
                "feature_set",
                "task",
                "model_id",
                "test_f1",
                "benign_fpr",
                "test_precision",
                "test_recall",
                "selected_threshold",
                "quality_target_met",
            ],
        )
    )
    lines.extend(["", "Quantized-clear scenario holdout", "--------------------------------"])
    scenario_quant = [row for row in quant_rows if row.get("protocol") == "scenario_holdout"]
    if args.skip_quantized_clear:
        lines.append("Not requested.")
    else:
        lines.append(
            format_table(
                scenario_quant,
                [
                    "feature_set",
                    "task",
                    "model_id",
                    "n_bits",
                    "test_f1",
                    "benign_fpr",
                    "f1_gap",
                    "quantized_gap_ok",
                    "quality_target_met",
                    "compile_eligible",
                ],
            )
        )
    if channel_confound:
        confound_line = (
            "Full passes the float gate and load does not. Channel KPIs are "
            "carrying the split. No full model is compiled."
        )
    elif delta_ran:
        confound_line = "Load and full missed the float gate. The delta set was trained."
    else:
        confound_line = "Load or full met the float gate. Delta was not required."
    lines.extend(
        [
            "",
            "Channel confound",
            "----------------",
            confound_line,
            f"Delta feature set trained: {delta_ran}",
            "",
            "Family holdout",
            "--------------",
            "Each family model is trained without that family and scored on all of its rows plus held-out benign test scenarios.",
        ]
    )
    family_float = [row for row in float_rows if str(row.get("protocol", "")).startswith("family_")]
    lines.append(
        format_table(
            family_float,
            ["protocol", "feature_set", "model_id", "test_f1", "benign_fpr", "test_recall", "quality_target_met"],
        )
    )
    lines.extend(["", "Claim bound", "-----------", claim_text(breakdown, float_rows)])
    lines.extend(["", "FHE compile", "-----------"])
    if not args.compile_fhe:
        lines.append("Not requested. Re-run with --compile-fhe after reading this report.")
    elif fhe_result is None:
        lines.append("No load or delta model passed the quantized-clear gate, so nothing was compiled.")
    elif fhe_result.get("status") != "ok":
        lines.append(f"{fhe_result.get('status')}: {fhe_result.get('error')}")
    else:
        lines.extend(
            [
                f"Model: {fhe_result['feature_set']}/{fhe_result['model_id']} q{fhe_result['n_bits']}",
                f"Compiled device: {fhe_result['compiled_device']}",
                f"Compile seconds: {fhe_result['compile_seconds']:.1f}",
                f"Simulation rows: {fhe_result['simulation_rows']}",
                f"Simulation F1: {fhe_result['simulation_f1']:.4f}",
                f"Simulation benign FPR: {fhe_result['simulation_benign_fpr']:.4f}",
                f"Key generation seconds: {fhe_result['keygen_s']:.4f}",
                f"Mean encrypt seconds: {fhe_result['mean_encrypt_s']:.6f}",
                f"Mean inference seconds: {fhe_result['mean_inference_s']:.6f}",
                f"Mean decrypt seconds: {fhe_result['mean_decrypt_s']:.6f}",
                f"Mean end-to-end seconds: {fhe_result['mean_e2e_s']:.6f}",
                f"Preferred latency <= {PREFERRED_E2E_S:.2f}s: {fhe_result['preferred_latency_met']}",
                f"Hard latency <= {HARD_E2E_S:.2f}s: {fhe_result['hard_latency_met']}",
                f"Circuit pass: {fhe_result['circuit_pass']}",
            ]
        )
    lines.extend(
        [
            "",
            "Gates",
            "-----",
            f"Float and quantized-clear F1 >= {args.target_f1:.2f}.",
            f"Benign false-positive rate <= {args.max_benign_fpr:.0%}.",
            f"Quantized-clear F1 within {args.max_quant_gap:.2f} of float.",
            "Compile candidates are binary load and delta models: logistic regression, "
            "LinearSVC, and one tree of depth 3 or 4. --compile-wide adds the depth-6 tree.",
            "Random forests, XGBoost, and the quantized MLP are accuracy screens. "
            "A 4-tree depth-4 forest on this stack measured about 1.8–2.2 s, so a "
            "20-tree model is not compiled from the 150–850 ms estimate.",
            "Six-class XGBClassifier is not trained. One-vs-rest would multiply the tree count by 6.",
            "Key generation is outside the end-to-end mean.",
        ]
    )
    (run_dir / "report.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def radio_group(category: str) -> str | None:
    if category == "Benign":
        return "Benign"
    if category in HIGH_RATE_CATEGORIES:
        return "HighRate"
    if category in LOW_RATE_CATEGORIES:
        return "LowRate"
    return None


def run_self_check() -> None:
    categories = {
        "benign": "Benign",
        "DoS_SYN": "DoS",
        "DDOS_UDP": "DDoS",
        "portscan_masscan": "Probe",
        "ssh_hydra": "BruteForce",
        "sqlmap": "Web",
        "Web Attacks": "Web",
        "http_flood": "DoS",
    }
    for raw, expected in categories.items():
        got = normalize_category(raw)
        if got != expected:
            raise SystemExit(f"Category mapping failed for {raw}: {got} != {expected}")

    rows = []
    for category, count, prefix in (
        ("Benign", 4, "b"),
        ("DoS", 2, "d"),
        ("Probe", 1, "p"),
    ):
        for index in range(count):
            rows.append(
                {
                    "scenario_id": f"{prefix}{index}",
                    "category": category,
                    "binary_label": 0 if category == "Benign" else 1,
                }
            )
    frame = pd.DataFrame(rows)
    summary = assign_scenarios(frame)
    assignment = summary["assignment"]
    if assignment["b2"] != "validation" or assignment["b3"] != "test" or assignment["b0"] != "train":
        raise SystemExit(f"Benign assignment failed: {assignment}")
    if assignment["d0"] != "train" or assignment["d1"] != "test":
        raise SystemExit(f"DoS assignment failed: {assignment}")
    if assignment["p0"] != "train" or "Probe" not in summary["unheld_categories"]:
        raise SystemExit(f"Single-scenario category failed: {summary}")
    assigned = apply_split(frame, assignment)
    if assigned.groupby("scenario_id")["split"].nunique().max() != 1:
        raise SystemExit("Scenario overlap detected.")

    tiny = pd.DataFrame(
        [
            {"scenario_id": "a", "category": "Benign"},
            {"scenario_id": "b", "category": "Benign"},
            {"scenario_id": "c", "category": "DoS"},
            {"scenario_id": "d", "category": "DoS"},
        ]
    )
    try:
        assign_scenarios(tiny)
    except ScenarioSplitError:
        pass
    else:
        raise SystemExit("Two-scenario categories should not invent a validation split.")

    series = pd.DataFrame(
        {
            "scenario_id": ["s", "s", "s", "t"],
            "ue_id": ["u", "u", "u", "u"],
            "timestamp_seconds": [0.0, 1.0, 3.0, 0.0],
            "dlBytes": [0.0, 10.0, 10.0, 5.0],
            "ulBytes": [0.0, 0.0, 0.0, 0.0],
            "dlBler": [0.0, 0.2, 0.2, 0.0],
            "ulBler": [0.0, 0.0, 0.0, 0.0],
        }
    )
    delta = attach_deltas(series).sort_values(["scenario_id", "timestamp_seconds"])
    first = float(delta.loc[delta["timestamp_seconds"].eq(0.0) & delta["scenario_id"].eq("s"), "d_dlBytes"].iloc[0])
    second = float(delta.loc[delta["timestamp_seconds"].eq(1.0), "d_dlBytes"].iloc[0])
    third = float(delta.loc[delta["timestamp_seconds"].eq(3.0), "d_dlBytes"].iloc[0])
    if first != 0.0 or third != 0.0 or abs(second - (np.log1p(10.0) - np.log1p(0.0))) > 1e-9:
        raise SystemExit(f"Delta boundary failed: {first}, {second}, {third}")
    try:
        refuse_row_random(["screen_radio_xapp.py", "--split", "random"])
    except SystemExit:
        pass
    else:
        raise SystemExit("Row-random split was accepted.")

    rows = []
    for start, label in (
        (0, "Benign"),
        (100, "Benign"),
        (200, "Benign"),
        (300, "DoS_SYN"),
        (400, "DoS_SYN"),
        (500, "portscan"),
    ):
        for offset in range(5):
            row = {name: 0.0 for name in FULL_FEATURES}
            row["timestamp"] = start + offset
            row["ue_id"] = "ue-1"
            row["Label"] = label
            row["dlBytes"] = 10.0 if label == "Benign" else 1000.0
            rows.append(row)
    prepared, schema = prepare_frame(
        pd.DataFrame(rows),
        {"source": "self-check"},
        5.0,
        lambda _message: None,
    )
    if schema["scenarios"] != 6:
        raise SystemExit(f"Expected 6 time-gap scenarios, got {schema['scenarios']}.")
    if schema["class_counts"].get("DoS") != 10 or schema["class_counts"].get("Probe") != 5:
        raise SystemExit(f"Category counts failed: {schema['class_counts']}")
    split_summary = assign_scenarios(prepared)
    if split_summary["split_counts"]["validation"] < 1 or split_summary["split_counts"]["test"] < 1:
        raise SystemExit(f"Prepared split failed: {split_summary['split_counts']}")
    binary = {config["id"]: config for config in binary_model_configs()}
    for required in ("lr_c1", "linear_svc", "tree_depth_6", "rf_20_depth_4", "xgb_20_depth_4", "qmlp_32"):
        if required not in binary:
            raise SystemExit(f"Binary grid is missing {required}.")
    if not binary["lr_c1"]["compile_shape"] or binary["rf_20_depth_4"]["compile_shape"]:
        raise SystemExit("Compile shapes do not match the measured latency policy.")
    if binary["xgb_20_depth_4"]["bits"] != (5,) or binary["lr_c1"]["bits"] != (6, 8):
        raise SystemExit("Per-model bit widths drifted from EXPERIMENT.md.")
    multi = multiclass_model_configs("radio6")
    if any(config["family"] == "xgb" or config["compile_shape"] for config in multi):
        raise SystemExit("6-class grid must not compile XGBClassifier or any other shape.")
    if radio_group("Web") != "LowRate" or radio_group("DoS") != "HighRate" or radio_group("Benign") != "Benign":
        raise SystemExit("3-class radio grouping failed.")
    both, both_mapping, both_ignored = rename_columns(
        pd.DataFrame({"ue_id": ["ue-1"], "rnti": [99], "timestamp": [0.0]})
    )
    if (
        list(both["ue_id"]) != ["ue-1"]
        or "rnti" in both.columns
        or both_ignored != ["rnti"]
        or both_mapping.get("ue_id") != "ue_id"
    ):
        raise SystemExit(f"ue_id should win over rnti: {both_mapping}, {both_ignored}")
    rnti_only, rnti_mapping, rnti_ignored = rename_columns(pd.DataFrame({"rnti": [99]}))
    if rnti_mapping != {"rnti": "ue_id"} or rnti_ignored or "ue_id" not in rnti_only.columns:
        raise SystemExit("A table with only rnti should still use it as ue_id.")
    try:
        rename_columns(pd.DataFrame({"time": [1.0], "frametime": [2.0]}))
    except SystemExit:
        pass
    else:
        raise SystemExit("Ambiguous timestamp aliases were accepted.")
    pmi_values = coerce_numeric(pd.Series(["(0, 0)", "(1, 2)", "3"]))
    if pmi_values.tolist() != [0.0, 1.0, 3.0]:
        raise SystemExit(f"PMI pair was not read as its first number: {pmi_values.tolist()}")
    slow = pd.DataFrame(
        {
            "category": ["DoS", "DoS", "DoS", "DoS"],
            "subtype": ["DoS_ACK", "DoS_ACK", "DoS_ACK", "DoS_ACK"],
            "timestamp_seconds": [0.0, 6.0, 12.0, 400.0],
        }
    )
    if assign_time_gap_scenarios(slow, 5.0).nunique() != 2:
        raise SystemExit("A slow DoS capture was split on its sample spacing.")
    fast = pd.DataFrame(
        {
            "category": ["Benign"] * 5,
            "subtype": ["Benign"] * 5,
            "timestamp_seconds": [0.0, 1.0, 2.0, 3.0, 100.0],
        }
    )
    if assign_time_gap_scenarios(fast, 5.0).nunique() != 2:
        raise SystemExit("A one-second capture did not split at the 5 second gap.")
    pmi_rows = []
    for start, label, subtype in (
        (0, "Benign", "Benign"),
        (100, "DoS", "DoS_SYN"),
        (200, "Web Attacks", "Web_XSS"),
    ):
        for offset in range(5):
            row = {name: 0.0 for name in FULL_FEATURES}
            row["pmi"] = "(0, 0)"
            row["timestamp"] = start + offset
            row["ue_id"] = "1"
            row["rnti"] = "9999"
            row["attack_category"] = label
            row["attack_subcategory"] = subtype
            pmi_rows.append(row)
    prepared_pmi, pmi_schema = prepare_frame(
        pd.DataFrame(pmi_rows),
        {"source": "self-check-pmi"},
        5.0,
        lambda _message: None,
    )
    if pmi_schema["rows"] != 15 or pmi_schema["class_counts"].get("Web") != 5:
        raise SystemExit(f"PMI text rows were dropped: {pmi_schema['class_counts']}")
    if set(prepared_pmi["subtype"]) != {"Benign", "DoS_SYN", "Web_XSS"}:
        raise SystemExit(f"Subtype column was ignored: {set(prepared_pmi['subtype'])}")
    if pmi_schema["ignored_alias_columns"] != ["rnti"]:
        raise SystemExit(f"rnti was not dropped beside ue_id: {pmi_schema['ignored_alias_columns']}")
    print("self-check ok")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    refuse_row_random(argv)
    args = parse_args(argv[1:])
    if args.self_check:
        run_self_check()
        return 0

    run_dir = create_run_dir(args.output_root)
    log = make_logger(run_dir)
    write_json(run_dir / "arguments.json", vars(args))
    write_json(run_dir / "status.json", {"status": "running"})
    log(f"New run directory: {run_dir}")
    try:
        if args.lower_layer_dir is not None:
            raw, source_info = read_lower_layer_dir(Path(args.lower_layer_dir))
        else:
            raw, source_info = read_sqlite(Path(args.db))
        frame, schema = prepare_frame(raw, source_info, args.scenario_gap_seconds, log)
        split_summary = assign_scenarios(frame)
        frame = apply_split(frame, split_summary["assignment"])
        frame = attach_deltas(frame)
        write_json(run_dir / "schema.json", schema)
        write_json(
            run_dir / "split.json",
            {
                "split_counts": split_summary["split_counts"],
                "unheld_categories": split_summary["unheld_categories"],
                "by_category": split_summary["by_category"],
                "assignment": split_summary["assignment"],
            },
        )

        train_mask, validation_mask, test_mask = masks_for(frame, "scenario_holdout")
        full_matrix = design_matrix(frame, FULL_FEATURES)
        scaled_full, _ = scale_matrix(full_matrix, indices_for(train_mask))
        y_train = frame.loc[train_mask, "binary_label"].to_numpy(dtype=np.int8)
        mi_ranking, paper_drop_agrees = mutual_information(
            scaled_full[indices_for(train_mask)],
            y_train,
            FULL_FEATURES,
        )
        write_json(
            run_dir / "mutual_information.json",
            {"ranking": mi_ranking, "paper_drop_agrees": paper_drop_agrees},
        )
        feature_sets: list[tuple[str, tuple[str, ...]]] = [
            ("load", LOAD_FEATURES),
            ("full", FULL_FEATURES),
        ]
        if paper_drop_agrees:
            feature_sets.append(
                ("full_reduced", tuple(column for column in FULL_FEATURES if column not in PAPER_DROP))
            )
            log("Training-split mutual information matches the paper drop. Adding full_reduced.")
        else:
            log("Training-split mutual information does not match the paper drop.")

        float_rows: list[dict[str, Any]] = []
        quant_rows: list[dict[str, Any]] = []
        breakdown: list[dict[str, Any]] = []
        kept: dict[tuple[str, str, int], Any] = {}
        base_configs = binary_model_configs()
        family_configs = [config for config in base_configs if config["compile_shape"]]

        def run_set(
            protocol: str,
            feature_set: str,
            columns: tuple[str, ...],
            configs: list[dict[str, Any]],
            label_column: str = "binary_label",
        ) -> None:
            train, validation, test = masks_for(frame, protocol)
            produced_float, produced_quant, produced_breakdown, produced_kept = screen_matrices(
                frame,
                feature_set,
                columns,
                protocol,
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
            for (model_id, n_bits), bundle in produced_kept.items():
                kept[(feature_set, model_id, n_bits)] = bundle
            write_tables(run_dir, float_rows, quant_rows, breakdown)

        for feature_set, columns in feature_sets:
            run_set("scenario_holdout", feature_set, columns, base_configs)

        delta_ran = False
        if misses_float_gate(float_rows, "load") and misses_float_gate(float_rows, "full"):
            delta_ran = True
            log("Load and full missed the float gate. Training the delta feature set.")
            delta_columns = LOAD_FEATURES + tuple(destination for _, destination, _ in DELTA_SPECS)
            run_set("scenario_holdout", "delta", delta_columns, base_configs)
        else:
            log("Delta feature set not trained. Load or full met the float gate.")

        for task_name, label_column in (("radio3", "radio3_label"), ("radio6", "radio6_label")):
            task_configs = multiclass_model_configs(task_name)
            for feature_set, columns in (("load", LOAD_FEATURES), ("full", FULL_FEATURES)):
                try:
                    run_set(
                        "scenario_holdout",
                        feature_set,
                        columns,
                        task_configs,
                        label_column=label_column,
                    )
                except ScenarioSplitError as exc:
                    log(f"Skipping {task_name}/{feature_set}: {exc}")

        channel_confound = misses_float_gate(float_rows, "load") and not misses_float_gate(float_rows, "full")
        if channel_confound:
            log("Channel confound: full meets the float gate and load does not.")

        screened_sets = list(feature_sets)
        if delta_ran:
            screened_sets.append(
                ("delta", LOAD_FEATURES + tuple(destination for _, destination, _ in DELTA_SPECS))
            )
        for protocol, category in (("family_holdout_web", "Web"), ("family_holdout_bruteforce", "BruteForce")):
            if category not in set(frame["category"]):
                log(f"Skipping {protocol}: no {category} rows.")
                continue
            for feature_set, columns in screened_sets:
                try:
                    run_set(protocol, feature_set, columns, family_configs)
                except ScenarioSplitError as exc:
                    log(f"Skipping {protocol}/{feature_set}: {exc}")

        fhe_result = None
        target = select_compile_target(quant_rows)
        if args.compile_fhe and target is None:
            fhe_result = {
                "status": "not_compiled",
                "error": "No load or delta model passed the quantized-clear gate.",
            }
        elif args.compile_fhe and target is not None:
            bundle = kept[(target["feature_set"], target["model_id"], int(target["n_bits"]))]
            try:
                fhe_result = compile_target(bundle, args, run_dir, log)
            except Exception as exc:
                log(traceback.format_exc())
                fhe_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if fhe_result is not None:
            write_json(run_dir / "fhe_result.json", fhe_result)

        write_report(
            run_dir,
            args,
            schema,
            split_summary,
            mi_ranking,
            paper_drop_agrees,
            float_rows,
            quant_rows,
            breakdown,
            fhe_result,
            delta_ran,
            channel_confound,
        )
        write_json(run_dir / "status.json", {"status": "complete", "run_dir": str(run_dir)})
        log(f"Report written to {run_dir / 'report.txt'}")
        return 0
    except (ScenarioSplitError, SystemExit) as exc:
        message = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, str) else str(exc)
        if isinstance(exc, SystemExit) and not isinstance(exc.code, str):
            raise
        log(message)
        write_json(run_dir / "status.json", {"status": "failed", "error": message})
        return 2
    except Exception as exc:
        log(traceback.format_exc())
        write_json(run_dir / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
