# Dataset: CSE-CIC-IDS-2018
# Replaces: KDD Cup 1999 / NSL-KDD (removed from CIC servers, deprecated)
# Download: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/
# Features: CICFlowMeter-V3; model uses 5 numerical + 2 categorical

from concrete.ml.deployment import FHEModelClient, FHEModelServer
from concrete.ml.common.serialization.loaders import load
import datetime
import glob
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


def clean_col_names(df):
    """Cleans column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.strip().replace(' ', '_').replace('/', '_').lower() for col in cols]
    df.columns = new_cols
    return df


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

    log_time(f"[{config_tag}] Running single-record inference on 1000 records...")
    inference_times = []
    report_interval = 200

    for i in range(1000):
        single_record = X_test_final[i:i+1]

        start_time = _time.time()
        output = classifier.predict(single_record)
        end_time = _time.time()

        duration = end_time - start_time
        inference_times.append(duration)

        if (i + 1) % report_interval == 0:
            avg_so_far = np.mean(inference_times)
            log_time(f"[{config_tag}] Processed {i+1}/1000 records (avg {avg_so_far:.6f}s/record)")

    log_time(f"[{config_tag}] All 1000 single-record inferences complete.")

    total_records = len(inference_times)
    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if total_records > 0 else 0

    print("\n" + "="*20 + " INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records}")
    print(f"   Total inference time: {total_inference_time:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time:.6f} seconds")
    print("="*61 + "\n")

    return {'config_tag': config_tag, 'pred_dur': pred_dur}


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

    if os.path.exists(combined_csv_path):
        log_time(f"Loading existing combined file '{combined_csv_path}'...")
        df = pd.read_csv(combined_csv_path)
        log_time(f"Loaded {len(df)} rows from '{combined_csv_path}'.")
    else:
        all_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
        if not all_files:
            raise FileNotFoundError(
                f"No CSV files found in '{data_folder}'. "
                f"Download with: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ {data_folder}/"
            )
        log_time(f"Found {len(all_files)} CSV files. Reading one at a time...")
        chunks = []
        for i, f in enumerate(all_files):
            log_time(f"  Reading file {i+1}/{len(all_files)}: {os.path.basename(f)}...")
            chunk = pd.read_csv(f, low_memory=False)
            chunk = clean_col_names(chunk)
            for col in NUMERICAL_FEATURES:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            needed = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ['label']
            chunk.dropna(subset=needed, inplace=True)
            for col in CATEGORICAL_FEATURES:
                chunk[col] = chunk[col].astype(str)
            chunks.append(chunk[needed])
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        log_time(f"Combined {len(all_files)} files → {len(df)} rows.")
        df.to_csv(combined_csv_path, index=False)
        log_time(f"Saved combined data to '{combined_csv_path}'.")

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

    log_time(f"[{config_tag}] [STEP 1/4] Loading FHE circuit from '{model_dir}'...")
    try:
        fhe_model_server = FHEModelServer(model_dir)
        fhe_model_server.load()
        fhe_model_client = FHEModelClient(model_dir)
        log_time(f"[{config_tag}] FHE circuit loaded.")
    except FileNotFoundError as e:
        log_time(f"[{config_tag}] ERROR loading model files: {e}")
        print("Please run model.py first to generate the FHE assets.")
        return

    log_time(f"[{config_tag}] [STEP 2/4] Preparing data using saved artifacts...")
    X_test_final, y_test_full, test_labels = prepare_test_data_with_saved_artifacts(n_components_svd)
    log_time(f"[{config_tag}] {len(test_labels)} test records available, processing {records}.")

    log_time(f"[{config_tag}] [STEP 3/4] Running FHE inference on {records} records...")
    inference_times = []
    preprocessing_times = []
    predictions = []
    true_labels = []

    log_time(f"[{config_tag}] Generating evaluation keys...")
    serialized_evaluation_keys = fhe_model_client.get_serialized_evaluation_keys()
    log_time(f"[{config_tag}] Evaluation keys ready ({len(serialized_evaluation_keys):,} bytes).")

    report_interval = max(1, records // 10)

    for i in range(records):
        single_record = X_test_final[i:i+1]

        start_preprocessing = _time.time()
        encrypted_input = fhe_model_client.quantize_encrypt_serialize(single_record)
        end_preprocessing = _time.time()

        preprocessing_duration = end_preprocessing - start_preprocessing
        preprocessing_times.append(preprocessing_duration)

        start_time = _time.time()
        encrypted_output = fhe_model_server.run(encrypted_input, serialized_evaluation_keys)
        end_time = _time.time()

        result = fhe_model_client.deserialize_decrypt_dequantize(encrypted_output)
        predicted_label = 1 if result[0][1] > 0.5 else 0
        predictions.append(predicted_label)

        true_label_text = test_labels.iloc[i]
        true_label_binary = 0 if true_label_text == 'benign' else 1
        true_labels.append(true_label_binary)

        duration = end_time - start_time
        inference_times.append(duration)

        if (i + 1) % report_interval == 0:
            avg_inf = np.mean(inference_times)
            avg_prep = np.mean(preprocessing_times)
            log_time(f"[{config_tag}] Processed {i+1}/{records} records (avg inference: {avg_inf:.4f}s, avg encrypt: {avg_prep:.6f}s)")

    log_time(f"[{config_tag}] [STEP 4/4] All {records} records processed.")

    log_time(f"[{config_tag}] FHE metrics:")
    log_model_metrics(np.array(true_labels), np.array(predictions))

    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if records > 0 else 0
    mean_preprocessing_time = np.mean(preprocessing_times) if records > 0 else 0

    print("\n" + "="*20 + " INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {records}")
    print(f"   Total inference time: {total_inference_time:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time:.6f} seconds")
    print(f"Mean preprocessing time/record: {mean_preprocessing_time:.6f} seconds")
    print("="*61 + "\n")

    return {'config_tag': config_tag, 'total_inf': total_inference_time, 'mean_inf': mean_inference_time, 'mean_prep': mean_preprocessing_time}


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
        print(f"{'Config':>30s} | {'Predict (s)':>12s}")
        print("-" * 50)
        for r in plaintext_results:
            print(f"{r['config_tag']:>30s} | {r['pred_dur']:>12.1f}")
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
        print("  FHE SUMMARY")
        print("=" * 70)
        print(f"{'Config':>30s} | {'Total Inf (s)':>14s} | {'Mean Inf (s)':>12s} | {'Mean Prep (s)':>13s}")
        print("-" * 80)
        for r in fhe_results:
            print(f"{r['config_tag']:>30s} | {r['total_inf']:>14.1f} | {r['mean_inf']:>12.6f} | {r['mean_prep']:>13.6f}")
        print()

    log_time(f"All {total} configurations complete.")
