# Dataset: CSE-CIC-IDS-2018
# Replaces: KDD Cup 1999 / NSL-KDD (removed from CIC servers, deprecated)
# Download: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/
# Features: CICFlowMeter-V3; model uses 5 numerical + 2 categorical

from concrete.ml.common.serialization.loaders import load
import datetime
import numpy as np
import os
import pandas as pd
import pickle
import sklearn
import sys
import time as _time
from sklearn.model_selection import train_test_split
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
    print("--- Model Evaluation ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.4f}")

    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall:.4f}")

    f1 = f1_score(y_test, y_pred)
    print(f"F1-Score: {f1:.4f}")

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred)
    print(report)


NUMERICAL_FEATURES = ['syn_flag_cnt', 'ack_flag_cnt', 'fin_flag_cnt', 'rst_flag_cnt', 'totlen_fwd_pkts']
CATEGORICAL_FEATURES = ['protocol', 'dst_port']


def predict_single_record_plaintext(estimators, depth, n_components_svd=100):
    config_tag = f"n={estimators}, d={depth}, svd={n_components_svd}"
    print(f"\n{'='*60}")
    print(f"  PLAINTEXT CONFIG: n_estimators={estimators}, max_depth={depth}, svd={n_components_svd}")
    print(f"{'='*60}")
    sys.stdout.flush()
    log_time(f"[{config_tag}] Starting plaintext inference benchmark")

    json_path = f"plaintext_model_{estimators}_estimators_{depth}_depth_svd_{n_components_svd}.json"
    if not os.path.exists(json_path):
        log_time(f"[{config_tag}] ERROR: Serialized model not found at '{json_path}'.")
        print(f"Please run model.py (or svd200_model.py) first to generate the model artifacts.")
        return None

    log_time(f"[{config_tag}] Loading Concrete ML model from '{json_path}'...")
    with open(json_path, "r") as f:
        classifier = load(f)
    log_time(f"[{config_tag}] Model loaded.")

    X_test_final, y_test_full, _ = prepare_test_data_with_saved_artifacts(n_components_svd)

    t0 = _time.time()
    log_time(f"[{config_tag}] Starting clear (plaintext) prediction on {X_test_final.shape[0]} samples...")
    y_pred = classifier.predict(X_test_final)
    pred_dur = _time.time() - t0
    log_time(f"[{config_tag}] Clear prediction completed in {pred_dur:,.1f}s")

    log_model_metrics(y_test_full, y_pred)

    n_single = min(1000, X_test_final.shape[0])
    log_time(f"[{config_tag}] Running single-record inference on {n_single} records...")
    inference_times = []
    report_interval = max(1, n_single // 5)

    for i in range(n_single):
        single_record = X_test_final[i:i+1]

        start_time = _time.time()
        classifier.predict(single_record)
        end_time = _time.time()

        inference_times.append(end_time - start_time)

        if (i + 1) % report_interval == 0:
            avg_so_far = np.mean(inference_times)
            log_time(f"[{config_tag}] Processed {i+1}/{n_single} records (avg {avg_so_far:.6f}s/record)")

    log_time(f"[{config_tag}] All {n_single} single-record inferences complete.")

    total_inference_time = float(np.sum(inference_times))
    mean_inference_time = float(np.mean(inference_times))
    std_inference_time = float(np.std(inference_times))

    print("\n" + "="*20 + " PLAINTEXT LATENCY SUMMARY " + "="*20)
    print(f"Records measured: {n_single}")
    print(f"Total inference time: {total_inference_time:.4f} s")
    print(f"Mean inference time/record: {mean_inference_time:.6f} s")
    print(f"Std inference time/record:  {std_inference_time:.6f} s")
    print("="*70 + "\n")

    return {
        'config_tag': config_tag,
        'pred_dur': pred_dur,
        'mean_inference_s': mean_inference_time,
        'std_inference_s': std_inference_time,
        'n_records': n_single,
    }


def prepare_test_data_with_saved_artifacts(n_components_svd=100):
    log_time(f"Loading test data using saved preprocessor and SVD artifacts (svd={n_components_svd})...")

    with open('preprocessor_cic_ids_2018.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    log_time("Loaded saved preprocessor.")

    svd_pkl_path = f'svd_cic_ids_2018_{n_components_svd}.pkl'
    with open(svd_pkl_path, 'rb') as f:
        svd = pickle.load(f)
    log_time(f"Loaded saved SVD from '{svd_pkl_path}'.")

    combined_csv_path = 'CIC-IDS-2018-Combined.csv'
    data_folder = 'CIC-IDS-2018'

    df = load_or_assemble_cic_ids_2018(
        data_folder,
        combined_csv_path,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        log_time,
    )

    log_time("Cleaning column names and dropping NaN/Inf rows...")
    df = clean_col_names(df)
    rows_before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=NUMERICAL_FEATURES, inplace=True)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    log_time(f"Dropped {rows_before - len(df)} rows with NaN/Inf. Remaining: {len(df)} rows.")

    df['label'] = df['label'].str.strip().str.lower()
    df['binary_label'] = (df['label'] != 'benign').astype(int)
    log_time(f"Label distribution — benign: {(df['binary_label']==0).sum()}, attack: {(df['binary_label']==1).sum()}")

    log_time("Splitting data 80/20 (stratified)...")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    log_time(f"Test: {len(test_df)}.")
    test_labels = test_df['label'].reset_index(drop=True)
    y_test_full = test_df['binary_label']

    X_test_sparse = preprocessor.transform(test_df)
    X_test_final = svd.transform(X_test_sparse)
    log_time(f"Test data ready — {X_test_final.shape}, using saved preprocessor and SVD.")

    del df
    return X_test_final, y_test_full, test_labels


def predict_single_record_with_comparison(estimators, depth, records=1000, n_components_svd=100):
    config_tag = f"n={estimators}, d={depth}, svd={n_components_svd}"
    print(f"\n{'='*60}")
    print(f"  FHE CONFIG: n_estimators={estimators}, max_depth={depth}, svd={n_components_svd}")
    print(f"{'='*60}")
    sys.stdout.flush()
    log_time(f"[{config_tag}] Starting FHE inference benchmark")

    model_dir = f"./fhe_model_{estimators}_estimators_{depth}_depth_svd_{n_components_svd}_components/"
    if not os.path.isdir(model_dir):
        log_time(f"[{config_tag}] ERROR: FHE model directory not found at '{model_dir}'.")
        print("Please run a training script first to generate the FHE assets.")
        return

    log_time(f"[{config_tag}] [STEP 1/3] Preparing data using saved artifacts...")
    X_test_final, y_test_full, test_labels = prepare_test_data_with_saved_artifacts(n_components_svd)
    n_records = min(records, len(test_labels))
    log_time(f"[{config_tag}] {len(test_labels)} test records available, processing {n_records}.")

    log_time(f"[{config_tag}] [STEP 2/3] Measuring encrypt / inference / decrypt...")
    try:
        latency = measure_fhe_roundtrip(
            model_dir,
            X_test_final,
            n_records=n_records,
            log_fn=lambda msg: log_time(f"[{config_tag}] {msg}"),
            return_predictions=True,
        )
    except FileNotFoundError as e:
        log_time(f"[{config_tag}] ERROR loading model files: {e}")
        print("Please run a training script first to generate the FHE assets.")
        return

    log_time(f"[{config_tag}] [STEP 3/3] Computing FHE metrics...")
    true_labels = []
    for i in range(latency["n_records"]):
        true_label_text = test_labels.iloc[i]
        true_labels.append(0 if true_label_text == 'benign' else 1)

    log_time(f"[{config_tag}] FHE metrics:")
    log_model_metrics(np.array(true_labels), np.array(latency["predictions"]))
    print_latency_summary(latency, title=f"FHE LATENCY ({config_tag})")

    return {
        'config_tag': config_tag,
        'n_records': latency["n_records"],
        'keygen_s': latency["keygen_s"],
        'mean_encrypt_s': latency["mean_encrypt_s"],
        'mean_inference_s': latency["mean_inference_s"],
        'mean_decrypt_s': latency["mean_decrypt_s"],
        'mean_e2e_s': latency["mean_e2e_s"],
        'std_encrypt_s': latency["std_encrypt_s"],
        'std_inference_s': latency["std_inference_s"],
        'std_decrypt_s': latency["std_decrypt_s"],
        'total_encrypt_s': latency["total_encrypt_s"],
        'total_inference_s': latency["total_inference_s"],
        'total_decrypt_s': latency["total_decrypt_s"],
        'total_e2e_s': latency["total_e2e_s"],
    }


if __name__ == "__main__":
    log_time(f"Starting best_model_single_record.py — scikit-learn {sklearn.__version__}")

    configs = [
        ("plaintext", 2, 2, 100),
        ("plaintext", 2, 4, 100),
        ("plaintext", 4, 2, 100),
        ("plaintext", 4, 4, 100),
        ("plaintext", 2, 2, 200),
        ("plaintext", 2, 4, 200),
        ("plaintext", 4, 2, 200),
        ("plaintext", 4, 4, 200),
        ("fhe", 2, 2, 1000, 100),
        ("fhe", 2, 4, 1000, 100),
        ("fhe", 4, 2, 1000, 100),
        ("fhe", 4, 4, 1000, 100),
        ("fhe", 2, 2, 1000, 200),
        ("fhe", 2, 4, 1000, 200),
        ("fhe", 4, 2, 1000, 200),
        ("fhe", 4, 4, 1000, 200),
    ]
    total = len(configs)
    log_time(f"Starting benchmark suite — {total} configurations to run")

    print("\n" + "#" * 70)
    print("#  PHASE 1: PLAINTEXT INFERENCE BENCHMARK")
    print("#" * 70)

    plaintext_results = []
    for idx, cfg in enumerate(configs, 1):
        log_time(f"=== Configuration {idx}/{total} ===")
        if cfg[0] == "plaintext":
            result = predict_single_record_plaintext(cfg[1], cfg[2], n_components_svd=cfg[3])
            if result:
                plaintext_results.append(result)

    if plaintext_results:
        print("\n" + "=" * 70)
        print("  PLAINTEXT SUMMARY")
        print("=" * 70)
        print(f"{'Config':>30s} | {'Batch Pred (s)':>14s} | {'Mean Inf (s)':>12s} | {'Std Inf (s)':>12s}")
        print("-" * 76)
        for r in plaintext_results:
            print(
                f"{r['config_tag']:>30s} | {r['pred_dur']:>14.1f} | "
                f"{r['mean_inference_s']:>12.6f} | {r['std_inference_s']:>12.6f}"
            )
        print()

    log_time("Phase 1 complete — all plaintext variants evaluated.")

    print("\n" + "#" * 70)
    print("#  PHASE 2: FHE INFERENCE BENCHMARK")
    print("#" * 70)

    fhe_results = []
    for idx, cfg in enumerate(configs, 1):
        if cfg[0] == "fhe":
            log_time(f"=== FHE Configuration ===")
            result = predict_single_record_with_comparison(cfg[1], cfg[2], cfg[3], n_components_svd=cfg[4])
            if result:
                fhe_results.append(result)

    if fhe_results:
        print("\n" + "=" * 70)
        print("  FHE LATENCY SUMMARY (per-record means)")
        print("=" * 70)
        print(
            f"{'Config':>30s} | {'Encrypt (s)':>12s} | {'Infer (s)':>12s} | "
            f"{'Decrypt (s)':>12s} | {'E2E (s)':>12s}"
        )
        print("-" * 90)
        for r in fhe_results:
            print(
                f"{r['config_tag']:>30s} | {r['mean_encrypt_s']:>12.6f} | "
                f"{r['mean_inference_s']:>12.6f} | {r['mean_decrypt_s']:>12.6f} | "
                f"{r['mean_e2e_s']:>12.6f}"
            )
        print()

    log_time(f"All {total} configurations complete.")
