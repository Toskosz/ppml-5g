# Dataset: CSE-CIC-IDS-2018
# Replaces: KDD Cup 1999 / NSL-KDD (removed from CIC servers, deprecated)
# Download: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/
# Features: CICFlowMeter-V3 (83 features); model uses 5 numerical + 2 categorical
# This script: SVD 15, 25, 40, and 100 components (optimized: n_bits=5, p_error=0.01)
# FHE simulation uses 50,000 test rows. Model files use a _fast suffix so they
# stay separate from svd_model.py. Preprocessor and SVD files are shared.
# Reruns reuse saved preprocessor, SVD, plaintext models, and FHE circuits
# compiled for the requested device (PPML_FHE_DEVICE / --device).
# Circuits are stored in <model_dir>/cpu and <model_dir>/cuda.

from concrete.ml.sklearn.rf import RandomForestClassifier
from concrete.ml.common.serialization.dumpers import dump
from concrete.ml.common.serialization.loaders import load
import datetime
import json
import numpy as np
import os
import pandas as pd
import pickle
import sklearn
import sys
import time as _time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from zoneinfo import ZoneInfo

from data_load import clean_col_names, load_or_assemble_cic_ids_2018
from fhe_latency import measure_fhe_roundtrip, print_latency_summary
from gpu_utils import (
    any_fhe_assets,
    circuit_reuse_status,
    circuit_save_dir,
    compile_for_device,
    compiled_circuit_dir,
    fhe_circuit_dirs,
    fhe_assets_ready,
    read_fhe_device,
    recover_interrupted_fhe_save,
    release_memory,
    resolve_fhe_device,
    save_fhe_model,
    save_projection,
)

FHE_SIM_SAMPLE_SIZE = 50_000
# Real FHE encrypt/infer/decrypt timing sample size (separate from the simulate subset).
FHE_LATENCY_SAMPLE_SIZE = 100
# FHE device: auto | cpu | cuda  (override with PPML_FHE_DEVICE)
FHE_DEVICE = os.environ.get("PPML_FHE_DEVICE", "auto")

_wall_start = _time.time()

def log_time(msg=None):
    brasilia_tz = ZoneInfo("America/Sao_Paulo")
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    brasilia_now = utc_now.astimezone(brasilia_tz)
    formatted_time = brasilia_now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    elapsed = _time.time() - _wall_start
    if msg:
        print(f"[LOG {formatted_time}] (elapsed {elapsed:,.1f}s) {msg}")
    else:
        print(f"[LOG {formatted_time}] (elapsed {elapsed:,.1f}s)")
    sys.stdout.flush()


def log_model_metrics(y_test, y_pred):
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
    }
    print_stored_metrics(metrics)
    return metrics


def print_stored_metrics(metrics):
    print("--- Model Evaluation ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("\n--- Confusion Matrix ---")
    print(np.array(metrics["confusion_matrix"]))
    print("\n--- Classification Report ---")
    print(metrics["classification_report"])


def fmt_opt(value, width, decimals):
    if value is None:
        return f"{'cached':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def plaintext_json_path(n_estimators, max_depth, n_components):
    return (
        f"plaintext_model_{n_estimators}_estimators_{max_depth}_depth_svd_{n_components}_fast.json"
    )


def plaintext_meta_path(json_path):
    if not json_path.endswith(".json"):
        raise ValueError(json_path)
    return json_path[:-5] + ".meta.json"


def fhe_model_dir(n_estimators, max_depth, n_components):
    return (
        f"./fhe_model_{n_estimators}_estimators_{max_depth}_depth_svd_{n_components}_components_fast"
    )


def plaintext_ready(json_path):
    return os.path.isfile(json_path) and os.path.getsize(json_path) > 0


def write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def latency_meta(latency):
    keys = (
        "n_records",
        "keygen_s",
        "total_encrypt_s",
        "total_inference_s",
        "total_decrypt_s",
        "total_e2e_s",
        "mean_encrypt_s",
        "mean_inference_s",
        "mean_decrypt_s",
        "mean_e2e_s",
        "std_encrypt_s",
        "std_inference_s",
        "std_decrypt_s",
    )
    return {key: latency[key] for key in keys}


def new_config_record(idx, n_estimators, max_depth, n_components):
    json_path = plaintext_json_path(n_estimators, max_depth, n_components)
    model_base = fhe_model_dir(n_estimators, max_depth, n_components)
    matched = compiled_circuit_dir(model_base, RESOLVED_FHE_DEVICE, log_fn=log_time)
    return {
        "idx": idx,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_components_svd": n_components,
        "config_tag": f"n={n_estimators}, d={max_depth}, svd={n_components}",
        "json_path": json_path,
        "model_base": model_base,
        "model_dir": matched or model_base,
        "train_dur": None,
        "pred_dur": None,
        "compile_dur": None,
        "fhe_dur": None,
        "fhe_device": None,
        "mean_encrypt_s": None,
        "mean_inference_s": None,
        "mean_decrypt_s": None,
        "mean_e2e_s": None,
        "plaintext_metrics": None,
        "fhe_metrics": None,
        "n_sim": None,
        "latency": None,
    }


def fill_saved_meta(cfg):
    meta_path = plaintext_meta_path(cfg["json_path"])
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        cfg["train_dur"] = meta.get("train_dur")
        cfg["pred_dur"] = meta.get("pred_dur")
        cfg["plaintext_metrics"] = meta.get("metrics")
    fhe_meta_path = os.path.join(cfg["model_dir"], "run_meta.json")
    if fhe_assets_ready(cfg["model_dir"]):
        cfg["fhe_device"] = read_fhe_device(cfg["model_dir"])
    if os.path.isfile(fhe_meta_path):
        with open(fhe_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        cfg["compile_dur"] = meta.get("compile_dur")
        cfg["fhe_dur"] = meta.get("fhe_dur")
        cfg["fhe_metrics"] = meta.get("metrics")
        cfg["n_sim"] = meta.get("n_sim")
        if meta.get("fhe_device"):
            cfg["fhe_device"] = meta["fhe_device"]
        latency = meta.get("latency") or None
        cfg["latency"] = latency
        if latency:
            cfg["mean_encrypt_s"] = latency.get("mean_encrypt_s")
            cfg["mean_inference_s"] = latency.get("mean_inference_s")
            cfg["mean_decrypt_s"] = latency.get("mean_decrypt_s")
            cfg["mean_e2e_s"] = latency.get("mean_e2e_s")
    return cfg


def replay_plaintext(cfg):
    tag = cfg["config_tag"]
    if cfg["train_dur"] is not None:
        log_time(f"[{tag}] Training completed in {cfg['train_dur']:,.1f}s")
    if cfg["pred_dur"] is not None:
        log_time(f"[{tag}] Clear prediction completed in {cfg['pred_dur']:,.1f}s")
    if cfg["plaintext_metrics"]:
        log_time(f"[{tag}] Plaintext metrics:")
        print_stored_metrics(cfg["plaintext_metrics"])
    log_time(f"[{tag}] Reusing plaintext model '{cfg['json_path']}'.")


def replay_fhe(cfg):
    tag = cfg["config_tag"]
    if cfg["compile_dur"] is not None and cfg["fhe_device"]:
        log_time(
            f"[{tag}] FHE compilation completed in {cfg['compile_dur']:,.1f}s "
            f"on device='{cfg['fhe_device']}'"
        )
    if cfg["fhe_dur"] is not None:
        log_time(f"[{tag}] FHE simulation completed in {cfg['fhe_dur']:,.1f}s")
    if cfg["fhe_metrics"]:
        if cfg.get("n_sim"):
            log_time(f"[{tag}] FHE metrics (on {int(cfg['n_sim']):,} sample subset):")
        else:
            log_time(f"[{tag}] FHE metrics:")
        print_stored_metrics(cfg["fhe_metrics"])
    log_time(
        f"[{tag}] Reusing compiled FHE circuit '{cfg['model_dir']}' "
        f"(device={cfg['fhe_device']})."
    )
    if cfg["latency"]:
        print_latency_summary(cfg["latency"], title=f"FHE LATENCY ({tag})")


def print_plaintext_summary(trained_configs, n_components_svd):
    print("\n" + "=" * 70)
    print(f"  PLAINTEXT SUMMARY  (SVD {n_components_svd})")
    print("=" * 70)
    print(f"{'Config':>30s} | {'Train (s)':>10s} | {'Predict (s)':>12s}")
    print("-" * 60)
    for cfg in trained_configs:
        if cfg["n_components_svd"] == n_components_svd:
            print(
                f"{cfg['config_tag']:>30s} | {fmt_opt(cfg['train_dur'], 10, 1)} | "
                f"{fmt_opt(cfg['pred_dur'], 12, 1)}"
            )
    print()


def print_fhe_summary(trained_configs, n_components_svd):
    print("\n" + "=" * 70)
    print(f"  FHE SUMMARY  (SVD {n_components_svd})")
    print("=" * 70)
    print(
        f"{'Config':>30s} | {'Device':>6s} | {'Train (s)':>10s} | {'Predict (s)':>12s} | "
        f"{'Compile (s)':>12s} | {'FHE Sim (s)':>12s} | {'Enc (s)':>10s} | {'Inf (s)':>10s} | {'Dec (s)':>10s}"
    )
    print("-" * 140)
    for cfg in trained_configs:
        if cfg["n_components_svd"] == n_components_svd:
            print(
                f"{cfg['config_tag']:>30s} | {(cfg['fhe_device'] or 'n/a'):>6s} | "
                f"{fmt_opt(cfg['train_dur'], 10, 1)} | {fmt_opt(cfg['pred_dur'], 12, 1)} | "
                f"{fmt_opt(cfg['compile_dur'], 12, 1)} | {fmt_opt(cfg['fhe_dur'], 12, 1)} | "
                f"{fmt_opt(cfg['mean_encrypt_s'], 10, 6)} | "
                f"{fmt_opt(cfg['mean_inference_s'], 10, 6)} | "
                f"{fmt_opt(cfg['mean_decrypt_s'], 10, 6)}"
            )
    print()


def print_global_summary(trained_configs):
    print("\n" + "#" * 70)
    print("#  GLOBAL SUMMARY")
    print("#" * 70)
    print(
        f"{'Config':>30s} | {'Device':>6s} | {'Train (s)':>10s} | {'Predict (s)':>12s} | "
        f"{'Compile (s)':>12s} | {'FHE Sim (s)':>12s} | {'Enc (s)':>10s} | {'Inf (s)':>10s} | "
        f"{'Dec (s)':>10s} | {'E2E (s)':>10s}"
    )
    print("-" * 155)
    for cfg in trained_configs:
        print(
            f"{cfg['config_tag']:>30s} | {(cfg['fhe_device'] or 'n/a'):>6s} | "
            f"{fmt_opt(cfg['train_dur'], 10, 1)} | {fmt_opt(cfg['pred_dur'], 12, 1)} | "
            f"{fmt_opt(cfg['compile_dur'], 12, 1)} | {fmt_opt(cfg['fhe_dur'], 12, 1)} | "
            f"{fmt_opt(cfg['mean_encrypt_s'], 10, 6)} | "
            f"{fmt_opt(cfg['mean_inference_s'], 10, 6)} | "
            f"{fmt_opt(cfg['mean_decrypt_s'], 10, 6)} | {fmt_opt(cfg['mean_e2e_s'], 10, 6)}"
        )
    print()


def config_is_complete(n_estimators, max_depth, n_components):
    return plaintext_ready(
        plaintext_json_path(n_estimators, max_depth, n_components)
    ) and compiled_circuit_dir(
        fhe_model_dir(n_estimators, max_depth, n_components),
        RESOLVED_FHE_DEVICE,
        log_fn=log_time,
    ) is not None


# (n_estimators, max_depth, n_components_svd)
# SVD 100 only fits n_estimators=2, max_depth=2. The next config (2, 4, 100)
# is killed during plaintext training (exit -9).
_tree_configs = ((2, 2), (2, 4), (4, 2), (4, 4))
_svd_components = (15, 25, 40)
configs = [
    (n_estimators, max_depth, n_components)
    for n_components in _svd_components
    for n_estimators, max_depth in _tree_configs
] + [(2, 2, 100)]
total_configs = len(configs)

log_time(f"Starting svd_model_fast.py — scikit-learn {sklearn.__version__}")
RESOLVED_FHE_DEVICE = resolve_fhe_device(FHE_DEVICE, log_fn=log_time)

for n_estimators, max_depth, n_components in configs:
    model_base = fhe_model_dir(n_estimators, max_depth, n_components)
    for model_dir in fhe_circuit_dirs(model_base):
        recovered = recover_interrupted_fhe_save(model_dir)
        if recovered:
            log_time(f"Recovered interrupted FHE save at '{model_dir}' ({recovered}).")

if all(config_is_complete(*cfg) for cfg in configs):
    log_time(
        "Every config already has a plaintext model and an FHE circuit "
        f"compiled for device='{RESOLVED_FHE_DEVICE}'."
    )
    for n_components in sorted({c[2] for c in configs}):
        for path in (f"X_train_svd_{n_components}.npy", f"X_test_svd_{n_components}.npy"):
            if os.path.isfile(path):
                os.remove(path)
                log_time(f"Deleted leftover projection '{path}'.")
    trained_configs = []
    for idx, (n_estimators, max_depth, n_components) in enumerate(configs, 1):
        cfg = fill_saved_meta(new_config_record(idx, n_estimators, max_depth, n_components))
        replay_plaintext(cfg)
        replay_fhe(cfg)
        trained_configs.append(cfg)
    print_global_summary(trained_configs)
    log_time("All configurations complete.")
    sys.exit(0)

numerical_features = ['syn_flag_cnt', 'ack_flag_cnt', 'fin_flag_cnt', 'rst_flag_cnt', 'totlen_fwd_pkts']
categorical_features = ['protocol', 'dst_port']

data_folder = 'CIC-IDS-2018'
combined_csv_path = 'CIC-IDS-2018-Combined.csv'

df = load_or_assemble_cic_ids_2018(
    data_folder,
    combined_csv_path,
    numerical_features,
    categorical_features,
    log_time,
)

log_time("Cleaning column names and dropping NaN/Inf rows...")
df = clean_col_names(df)
rows_before = len(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
for col in numerical_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=numerical_features, inplace=True)
for col in categorical_features:
    df[col] = df[col].astype(str)
log_time(f"Dropped {rows_before - len(df)} rows with NaN/Inf. Remaining: {len(df)} rows.")

df['label'] = df['label'].str.strip().str.lower()
df['binary_label'] = (df['label'] != 'benign').astype(int)
log_time(f"Label distribution — benign: {(df['binary_label']==0).sum()}, attack: {(df['binary_label']==1).sum()}")

log_time("Splitting data 80/20 (stratified)...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
log_time(f"Train: {len(train_df)}, Test: {len(test_df)}.")

preprocessor_path = "preprocessor_cic_ids_2018.pkl"
print("Applying preprocessing (MinMaxScaler and OneHotEncoder)...")
if os.path.isfile(preprocessor_path):
    log_time(f"Loading existing preprocessor from '{preprocessor_path}'.")
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    X_train_sparse = preprocessor.transform(train_df)
    X_test_sparse = preprocessor.transform(test_df)
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    log_time("Fitting preprocessor on training data...")
    X_train_sparse = preprocessor.fit_transform(train_df)
    X_test_sparse = preprocessor.transform(test_df)
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    log_time(f"Preprocessor saved to '{preprocessor_path}'.")
log_time(f"Preprocessing done. X_train: {X_train_sparse.shape}, X_test: {X_test_sparse.shape} (sparse)")

y_train_final = train_df['binary_label'].to_numpy(dtype=np.int32, copy=True)
y_test_final = test_df['binary_label'].to_numpy(dtype=np.int32, copy=True)

del train_df
del test_df
del df
release_memory()

del preprocessor
release_memory()
log_time("Released raw tables and fitted preprocessor.")

print(f"\n{'Config':>30s} | {'Phase':>20s} | {'Wall Time':>12s}")
print("-" * 70)

trained_configs = []
svd_groups = sorted(set(c[2] for c in configs))
projection_paths = {}

for svd_idx, n_components_svd in enumerate(svd_groups, 1):
    group_configs = [c for c in configs if c[2] == n_components_svd]
    if all(config_is_complete(*cfg) for cfg in group_configs):
        log_time(
            f"SVD {n_components_svd} artifacts already complete for device='{RESOLVED_FHE_DEVICE}'. "
            "Skipping projection."
        )
        continue

    svd_pkl_path = f"svd_cic_ids_2018_{n_components_svd}.pkl"
    train_path = f"X_train_svd_{n_components_svd}.npy"
    test_path = f"X_test_svd_{n_components_svd}.npy"
    projection_paths[n_components_svd] = (train_path, test_path)
    saved_models = any(
        plaintext_ready(plaintext_json_path(*cfg)) or any_fhe_assets(fhe_model_dir(*cfg))
        for cfg in group_configs
    )
    if os.path.isfile(svd_pkl_path):
        log_time(f"Loading existing SVD from '{svd_pkl_path}'.")
        with open(svd_pkl_path, "rb") as f:
            svd = pickle.load(f)
    elif saved_models:
        raise SystemExit(
            f"SVD file '{svd_pkl_path}' is missing but saved models for SVD "
            f"{n_components_svd} exist. Refusing to refit, which would change "
            "the feature space of those models."
        )
    else:
        log_time(
            f"Applying TruncatedSVD: {X_train_sparse.shape[1]} → {n_components_svd} components  "
            f"[SVD group {svd_idx}/{len(svd_groups)}]..."
        )
        svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
        svd_sample_size = 500_000
        n_rows = X_train_sparse.shape[0]
        sample_idx = None
        if n_rows <= svd_sample_size:
            X_svd_fit = X_train_sparse
        else:
            log_time(f"Subsampling {svd_sample_size:,} rows from {n_rows:,} for SVD fitting...")
            sample_idx = np.random.default_rng(42).choice(n_rows, svd_sample_size, replace=False)
            X_svd_fit = X_train_sparse[sample_idx]
        svd.fit(X_svd_fit)
        del X_svd_fit, sample_idx
        release_memory()
        with open(svd_pkl_path, "wb") as f:
            pickle.dump(svd, f)
        log_time(f"SVD saved to '{svd_pkl_path}'.")

    explained = float(svd.explained_variance_ratio_.sum())
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        log_time(
            f"Reusing SVD {n_components_svd} projections ('{train_path}', '{test_path}'). "
            f"Explained variance ratio sum: {explained:.4f}"
        )
    else:
        save_projection(svd, X_train_sparse, train_path, log_fn=log_time)
        save_projection(svd, X_test_sparse, test_path, log_fn=log_time)
        log_time(
            f"SVD {n_components_svd} projections stored. "
            f"Explained variance ratio sum: {explained:.4f}"
        )
    del svd
    release_memory()

del X_train_sparse
del X_test_sparse
release_memory()
log_time("Released one-hot matrices before training and FHE compilation.")

for svd_idx, n_components_svd in enumerate(svd_groups, 1):
    group_configs = [c for c in configs if c[2] == n_components_svd]
    if all(config_is_complete(*cfg) for cfg in group_configs):
        log_time(
            f"SVD {n_components_svd} artifacts already complete for device='{RESOLVED_FHE_DEVICE}'. "
            "Skipping plaintext training and FHE compilation."
        )
        print("\n" + "#" * 70)
        print(f"#  PHASE 1: PLAINTEXT TRAINING & EVALUATION  (SVD {n_components_svd})")
        print("#" * 70)
        for n_estimators, max_depth, _ in group_configs:
            idx = configs.index((n_estimators, max_depth, n_components_svd)) + 1
            cfg = fill_saved_meta(new_config_record(idx, n_estimators, max_depth, n_components_svd))
            print(f"\n{'='*60}")
            print(
                f"  CONFIG {cfg['idx']}/{total_configs}: n_estimators={n_estimators}, "
                f"max_depth={max_depth}, svd={n_components_svd}"
            )
            print(f"{'='*60}")
            sys.stdout.flush()
            replay_plaintext(cfg)
            trained_configs.append(cfg)
        print_plaintext_summary(trained_configs, n_components_svd)
        log_time(f"Phase 1 complete for SVD {n_components_svd} — reused saved plaintext models.")
        print("\n" + "#" * 70)
        print(f"#  PHASE 2: FHE COMPILATION & SIMULATION  (SVD {n_components_svd})")
        print("#" * 70)
        for cfg in trained_configs:
            if cfg["n_components_svd"] != n_components_svd:
                continue
            print(f"\n{'='*60}")
            print(
                f"  FHE for CONFIG {cfg['idx']}/{total_configs}: "
                f"n_estimators={cfg['n_estimators']}, max_depth={cfg['max_depth']}, "
                f"svd={n_components_svd}"
            )
            print(f"{'='*60}")
            sys.stdout.flush()
            replay_fhe(cfg)
        print_fhe_summary(trained_configs, n_components_svd)
        log_time(f"FHE phase complete for SVD {n_components_svd}.")
        for path in (
            f"X_train_svd_{n_components_svd}.npy",
            f"X_test_svd_{n_components_svd}.npy",
        ):
            if os.path.isfile(path):
                os.remove(path)
                log_time(f"Deleted leftover projection '{path}'.")
        continue

    train_path, test_path = projection_paths[n_components_svd]
    needs_plaintext = any(
        not plaintext_ready(plaintext_json_path(n_estimators, max_depth, n_components_svd))
        for n_estimators, max_depth, _ in group_configs
    )
    X_train_final = None
    X_test_final = None
    if needs_plaintext:
        log_time(f"Loading SVD {n_components_svd} projections for plaintext training...")
        X_train_final = np.load(train_path)
        X_test_final = np.load(test_path)
        log_time(f"Final training data: {X_train_final.shape}, testing data: {X_test_final.shape}")
    else:
        log_time(f"SVD {n_components_svd} plaintext models already saved. Skipping training.")

    print("\n" + "#" * 70)
    print(f"#  PHASE 1: PLAINTEXT TRAINING & EVALUATION  (SVD {n_components_svd})")
    print("#" * 70)

    for n_estimators, max_depth, _ in group_configs:
        idx = configs.index((n_estimators, max_depth, n_components_svd)) + 1
        cfg = new_config_record(idx, n_estimators, max_depth, n_components_svd)
        config_tag = cfg["config_tag"]
        print(f"\n{'='*60}")
        print(f"  CONFIG {idx}/{total_configs}: n_estimators={n_estimators}, max_depth={max_depth}, svd={n_components_svd}")
        print(f"{'='*60}")
        sys.stdout.flush()

        if plaintext_ready(cfg["json_path"]):
            fill_saved_meta(cfg)
            replay_plaintext(cfg)
            trained_configs.append(cfg)
            continue

        t0 = _time.time()
        log_time(f"[{config_tag}] Training RandomForestClassifier...")
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_bits=5,
        )
        classifier.fit(X_train_final, y_train_final)
        train_dur = _time.time() - t0
        log_time(f"[{config_tag}] Training completed in {train_dur:,.1f}s")

        t0 = _time.time()
        log_time(f"[{config_tag}] Starting clear (plaintext) prediction on {X_test_final.shape[0]} samples...")
        y_pred = classifier.predict(X_test_final)
        pred_dur = _time.time() - t0
        log_time(f"[{config_tag}] Clear prediction completed in {pred_dur:,.1f}s")

        log_time(f"[{config_tag}] Plaintext metrics:")
        plaintext_metrics = log_model_metrics(y_test_final, y_pred)

        json_tmp = cfg["json_path"] + ".tmp"
        with open(json_tmp, "w") as f:
            dump(classifier, f)
        os.replace(json_tmp, cfg["json_path"])
        write_json_atomic(
            plaintext_meta_path(cfg["json_path"]),
            {
                "train_dur": train_dur,
                "pred_dur": pred_dur,
                "metrics": plaintext_metrics,
            },
        )
        log_time(f"[{config_tag}] Plaintext model saved to '{cfg['json_path']}'.")

        cfg["train_dur"] = train_dur
        cfg["pred_dur"] = pred_dur
        cfg["plaintext_metrics"] = plaintext_metrics
        trained_configs.append(cfg)
        del classifier
        del y_pred
        release_memory()

    print_plaintext_summary(trained_configs, n_components_svd)

    log_time(f"Phase 1 complete for SVD {n_components_svd} — all plaintext variants trained and evaluated.")

    if X_train_final is not None:
        del X_train_final
        del X_test_final
        release_memory()
        log_time(f"Released SVD {n_components_svd} projections before FHE compilation.")

    print("\n" + "#" * 70)
    print(f"#  PHASE 2: FHE COMPILATION & SIMULATION  (SVD {n_components_svd})")
    print("#" * 70)

    for cfg in trained_configs:
        if cfg['n_components_svd'] != n_components_svd:
            continue

        config_tag = cfg['config_tag']
        n_estimators = cfg['n_estimators']
        max_depth = cfg['max_depth']

        print(f"\n{'='*60}")
        print(f"  FHE for CONFIG {cfg['idx']}/{total_configs}: n_estimators={n_estimators}, max_depth={max_depth}, svd={n_components_svd}")
        print(f"{'='*60}")
        sys.stdout.flush()

        matched, mismatch = circuit_reuse_status(
            cfg["model_base"], RESOLVED_FHE_DEVICE, log_fn=log_time
        )
        if matched:
            cfg["model_dir"] = matched
            fill_saved_meta(cfg)
            replay_fhe(cfg)
            continue
        if mismatch:
            log_time(
                f"[{config_tag}] {mismatch} Compiling for device='{RESOLVED_FHE_DEVICE}'."
            )

        log_time(f"[{config_tag}] Loading plaintext model from '{cfg['json_path']}'...")
        with open(cfg['json_path'], "r") as f:
            classifier = load(f)

        log_time(f"[{config_tag}] Loading train projection for FHE compilation...")
        X_train_final = np.load(train_path)
        t0 = _time.time()
        log_time(f"[{config_tag}] Compiling FHE circuit (this may take a long time)...")
        _, fhe_device = compile_for_device(
            classifier,
            X_train_final,
            device=FHE_DEVICE,
            p_error=0.01,
            log_fn=lambda msg: log_time(f"[{config_tag}] {msg}"),
        )
        compile_dur = _time.time() - t0
        log_time(f"[{config_tag}] FHE compilation completed in {compile_dur:,.1f}s on device='{fhe_device}'")
        del X_train_final
        release_memory()

        log_time(f"[{config_tag}] Loading test projection for FHE simulation...")
        X_test_final = np.load(test_path)
        n_sim = min(FHE_SIM_SAMPLE_SIZE, X_test_final.shape[0])
        sim_idx = np.random.default_rng(42).choice(X_test_final.shape[0], n_sim, replace=False)
        X_test_sim = np.ascontiguousarray(X_test_final[sim_idx])
        y_test_sim = y_test_final[sim_idx]
        n_latency = min(FHE_LATENCY_SAMPLE_SIZE, X_test_final.shape[0])
        X_latency = np.array(X_test_final[:n_latency], dtype=np.float32, copy=True)
        del X_test_final, sim_idx
        release_memory()
        log_time(f"[{config_tag}] Subsampled {n_sim:,} rows for FHE simulation...")

        t0 = _time.time()
        log_time(f"[{config_tag}] Starting FHE simulation prediction on {n_sim:,} samples...")
        y_pred_fhe = classifier.predict(X_test_sim, fhe="simulate")
        fhe_dur = _time.time() - t0
        log_time(f"[{config_tag}] FHE simulation completed in {fhe_dur:,.1f}s")

        log_time(f"[{config_tag}] FHE metrics (on {n_sim:,} sample subset):")
        fhe_metrics = log_model_metrics(y_test_sim, y_pred_fhe)
        del X_test_sim, y_test_sim, y_pred_fhe
        release_memory()

        model_dir = circuit_save_dir(cfg["model_base"], fhe_device, log_fn=log_time)
        cfg["model_dir"] = model_dir
        log_time(f"[{config_tag}] Saving compiled FHE circuit to '{model_dir}'...")
        run_meta = {
            "compile_dur": compile_dur,
            "fhe_dur": fhe_dur,
            "fhe_device": fhe_device,
            "n_sim": int(n_sim),
            "metrics": fhe_metrics,
        }
        save_fhe_model(model_dir, classifier, fhe_device, meta=run_meta)
        del classifier
        train_s = "cached" if cfg["train_dur"] is None else f"{cfg['train_dur']:,.1f}s"
        pred_s = "cached" if cfg["pred_dur"] is None else f"{cfg['pred_dur']:,.1f}s"
        log_time(
            f"[{config_tag}] FHE assets saved (device={fhe_device}). Summary — "
            f"train: {train_s}, predict: {pred_s}, compile: {compile_dur:,.1f}s, "
            f"fhe_sim: {fhe_dur:,.1f}s"
        )
        release_memory()

        log_time(f"[{config_tag}] Measuring real FHE encrypt/inference/decrypt on {n_latency} records...")
        latency = measure_fhe_roundtrip(
            model_dir,
            X_latency,
            n_records=n_latency,
            log_fn=lambda msg: log_time(f"[{config_tag}] {msg}"),
        )
        print_latency_summary(latency, title=f"FHE LATENCY ({config_tag})")
        run_meta["latency"] = latency_meta(latency)
        write_json_atomic(os.path.join(model_dir, "run_meta.json"), run_meta)

        cfg['compile_dur'] = compile_dur
        cfg['fhe_dur'] = fhe_dur
        cfg['fhe_device'] = fhe_device
        cfg['fhe_metrics'] = fhe_metrics
        cfg['n_sim'] = int(n_sim)
        cfg['latency'] = run_meta["latency"]
        cfg['mean_encrypt_s'] = latency['mean_encrypt_s']
        cfg['mean_inference_s'] = latency['mean_inference_s']
        cfg['mean_decrypt_s'] = latency['mean_decrypt_s']
        cfg['mean_e2e_s'] = latency['mean_e2e_s']
        del X_latency, latency
        release_memory()

    print_fhe_summary(trained_configs, n_components_svd)

    log_time(f"FHE phase complete for SVD {n_components_svd}.")
    for projection_path in (train_path, test_path):
        os.remove(projection_path)
    log_time(f"Deleted SVD {n_components_svd} projection files.")
    release_memory()

print_global_summary(trained_configs)

log_time("All configurations complete.")
