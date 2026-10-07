"""Memory-conscious CIC-IDS-2018 CSV loading helpers."""

import glob
import os

import numpy as np
import pandas as pd


def normalize_col_name(col):
    return col.strip().replace(" ", "_").replace("/", "_").lower()


def clean_col_names(df):
    """Cleans column names to be Python-friendly."""
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def resolve_usecols(csv_path, needed_cols):
    """Map normalized column names to the raw header names in *csv_path*."""
    header = pd.read_csv(csv_path, nrows=0)
    cleaned_to_raw = {normalize_col_name(c): c for c in header.columns}
    missing = [c for c in needed_cols if c not in cleaned_to_raw]
    if missing:
        raise ValueError(
            f"{csv_path}: missing columns {missing}. "
            f"Available (normalized): {sorted(cleaned_to_raw)}"
        )
    return [cleaned_to_raw[c] for c in needed_cols]


def load_or_assemble_cic_ids_2018(
    data_folder,
    combined_csv_path,
    numerical_features,
    categorical_features,
    log_time,
):
    """Load the slim combined CSV, or build it from day files with low peak RAM.

    Each day file is read with ``usecols`` only (not all ~80 CICFlowMeter columns)
    and appended to a partial combined file so previous days are not kept in memory.
    """
    needed_cols = numerical_features + categorical_features + ["label"]

    if os.path.exists(combined_csv_path):
        log_time(f"Loading existing combined file '{combined_csv_path}'...")
        df = pd.read_csv(combined_csv_path, usecols=needed_cols)
        log_time(f"Loaded {len(df)} rows from '{combined_csv_path}'.")
        return df

    log_time(f"No combined file found. Assembling data from '{data_folder}' folder...")
    all_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not all_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_folder}'. "
            f"Download with: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ {data_folder}/"
        )
    log_time(
        f"Found {len(all_files)} CSV files. "
        "Reading one at a time (needed columns only)..."
    )

    partial_path = combined_csv_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    total_rows = 0
    wrote_header = False
    for i, f in enumerate(all_files):
        log_time(f"  Reading file {i+1}/{len(all_files)}: {os.path.basename(f)}...")
        usecols = resolve_usecols(f, needed_cols)
        chunk = pd.read_csv(f, usecols=usecols, low_memory=False)
        chunk = clean_col_names(chunk)
        for col in numerical_features:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(subset=needed_cols, inplace=True)
        for col in categorical_features:
            chunk[col] = chunk[col].astype(str)
        slim = chunk[needed_cols]
        slim.to_csv(
            partial_path,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
            index=False,
        )
        total_rows += len(slim)
        wrote_header = True
        del chunk, slim

    os.replace(partial_path, combined_csv_path)
    log_time(f"Combined {len(all_files)} files → {total_rows} rows.")
    log_time(f"Saved combined data to '{combined_csv_path}'.")
    log_time(f"Loading combined file '{combined_csv_path}'...")
    df = pd.read_csv(combined_csv_path, usecols=needed_cols)
    log_time(f"Loaded {len(df)} rows from '{combined_csv_path}'.")
    return df
