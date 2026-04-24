# Dataset: CSE-CIC-IDS-2018
# Replaces: KDD Cup 1999 / NSL-KDD (removed from CIC servers, deprecated)
# Download: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/
# Features: CICFlowMeter-V3; model uses 5 numerical + 2 categorical

from concrete.ml.deployment import FHEModelClient, FHEModelServer
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
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier as RandomForestSklearn
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


def clean_col_names(df):
    """Cleans column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.strip().replace(' ', '_').replace('/', '_').lower() for col in cols]
    df.columns = new_cols
    return df


def load_and_preprocess(n_components_svd=100):
    log_time(f"Loading and preprocessing data (SVD components={n_components_svd})...")
    combined_csv_path = 'CIC-IDS-2018-Combined.csv'
    data_folder = 'CIC-IDS-2018'

    if os.path.exists(combined_csv_path):
        log_time(f"Loading '{combined_csv_path}'...")
        df = pd.read_csv(combined_csv_path)
        log_time(f"Loaded {len(df)} rows.")
    else:
        all_files = glob.glob(os.path.join(data_folder, "*.csv"))
        if not all_files:
            raise FileNotFoundError(
                f"No CSV files found in '{data_folder}'. "
                f"Download with: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ {data_folder}/"
            )
        log_time(f"Reading {len(all_files)} CSV files...")
        df_list = [pd.read_csv(f, low_memory=False) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(combined_csv_path, index=False)
        log_time(f"Combined and saved {len(df)} rows to '{combined_csv_path}'.")

    df = clean_col_names(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numerical_features, inplace=True)
    for col in categorical_features:
        df[col] = df[col].astype(str)
    df['label'] = df['label'].str.strip().str.lower()
    df['binary_label'] = (df['label'] != 'benign').astype(int)

    log_time("Splitting data 80/20...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    numerical_features = ['syn_flag_cnt', 'ack_flag_cnt', 'fin_flag_cnt', 'rst_flag_cnt', 'totlen_fwd_pkts']
    categorical_features = ['protocol', 'dst_port']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    log_time("Fitting preprocessor...")
    X_train_sparse = preprocessor.fit_transform(train_df)
    X_test_sparse = preprocessor.transform(test_df)
    log_time(f"Preprocessing done. X_train: {X_train_sparse.shape}, X_test: {X_test_sparse.shape}")

    with open('preprocessor_cic_ids_2018.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    y_train_full = train_df['binary_label']
    y_test_full = test_df['binary_label']
    test_labels = test_df['label'].reset_index(drop=True)

    del train_df
    del df

    log_time(f"Applying TruncatedSVD ({X_train_sparse.shape[1]} → {n_components_svd})...")
    svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
    X_train_final = svd.fit_transform(X_train_sparse)
    X_test_final = svd.transform(X_test_sparse)
    log_time(f"SVD done. Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    del X_train_sparse
    del X_test_sparse

    log_time(f"Data ready — train: {X_train_final.shape}, test: {X_test_final.shape}")
    return X_train_final, X_test_final, y_train_full, y_test_full, test_labels, preprocessor


def predict_single_record_plaintext(estimators, depth, n_components_svd=100):
    config_tag = f"plaintext|n={estimators},d={depth}"
    log_time(f"[{config_tag}] Starting plaintext inference benchmark")

    X_train_final, X_test_final, y_train_full, y_test_full, test_labels, _ = \
        load_and_preprocess(n_components_svd)

    log_time(f"[{config_tag}] Training sklearn RandomForest ({estimators} estimators, depth={depth})...")
    t_train_start = _time.time()
    classifier = RandomForestSklearn(
        n_estimators=estimators,
        max_depth=depth,
        random_state=42
    )
    classifier.fit(X_train_final, y_train_full)
    log_time(f"[{config_tag}] Training completed in {_time.time() - t_train_start:,.1f}s")

    log_time(f"[{config_tag}] Running inference on 1000 records...")
    inference_times = []
    report_interval = 200

    for i in range(1000):
        single_record = X_test_final[i:i+1]

        start_time = time.time()
        output = classifier.predict(single_record)
        end_time = time.time()

        duration = end_time - start_time
        inference_times.append(duration)

        true_label_text = test_labels.iloc[i]
        true_label_binary = 0 if true_label_text == 'benign' else 1

        if (i + 1) % report_interval == 0:
            avg_so_far = np.mean(inference_times)
            log_time(f"[{config_tag}] Processed {i+1}/1000 records (avg {avg_so_far:.6f}s/record)")

    log_time(f"[{config_tag}] All 1000 records processed.")
    log_time("Calculating final statistics...")

    total_records = len(inference_times)
    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if total_records > 0 else 0

    print("\n" + "="*20 + " INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records}")
    print(f"   Total inference time: {total_inference_time:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time:.6f} seconds")
    print("="*61 + "\n")


def prepare_test_data_with_saved_artifacts(n_components_svd=100):
    log_time(f"Loading test data using saved preprocessor and SVD artifacts...")
    numerical_features = ['syn_flag_cnt', 'ack_flag_cnt', 'fin_flag_cnt', 'rst_flag_cnt', 'totlen_fwd_pkts']
    categorical_features = ['protocol', 'dst_port']

    with open('preprocessor_cic_ids_2018.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    log_time("Loaded saved preprocessor.")

    with open('svd_cic_ids_2018.pkl', 'rb') as f:
        svd = pickle.load(f)
    log_time("Loaded saved SVD.")

    combined_csv_path = 'CIC-IDS-2018-Combined.csv'
    data_folder = 'CIC-IDS-2018'

    if os.path.exists(combined_csv_path):
        log_time(f"Loading '{combined_csv_path}'...")
        df = pd.read_csv(combined_csv_path)
        log_time(f"Loaded {len(df)} rows.")
    else:
        all_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
        if not all_files:
            raise FileNotFoundError(
                f"No CSV files found in '{data_folder}'. "
                f"Download with: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ {data_folder}/"
            )
        log_time(f"Reading {len(all_files)} CSV files...")
        df_list = [pd.read_csv(f, low_memory=False) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(combined_csv_path, index=False)
        log_time(f"Combined and saved {len(df)} rows to '{combined_csv_path}'.")

    df = clean_col_names(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numerical_features, inplace=True)
    for col in categorical_features:
        df[col] = df[col].astype(str)
    df['label'] = df['label'].str.strip().str.lower()
    df['binary_label'] = (df['label'] != 'benign').astype(int)

    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    test_labels = test_df['label'].reset_index(drop=True)

    X_test_sparse = preprocessor.transform(test_df)
    X_test_final = svd.transform(X_test_sparse)
    log_time(f"Test data ready — {X_test_final.shape}, using saved preprocessor and SVD.")

    del df
    return X_test_final, test_labels


def predict_single_record_with_comparison(estimators, depth, records=1000, n_components_svd=100):
    config_tag = f"fhe|n={estimators},d={depth}"
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
    X_test_final, test_labels = prepare_test_data_with_saved_artifacts(n_components_svd)
    log_time(f"[{config_tag}] {len(test_labels)} test records available, processing {records}.")

    log_time(f"[{config_tag}] [STEP 3/4] Running FHE inference on {records} records...")
    inference_times = []
    preprocessing_times = []

    log_time(f"[{config_tag}] Generating evaluation keys...")
    serialized_evaluation_keys = fhe_model_client.get_serialized_evaluation_keys()
    log_time(f"[{config_tag}] Evaluation keys ready ({len(serialized_evaluation_keys):,} bytes).")

    report_interval = max(1, records // 10)

    for i in range(records):
        single_record = X_test_final[i:i+1]

        start_preprocessing = time.time()
        encrypted_input = fhe_model_client.quantize_encrypt_serialize(single_record)
        end_preprocessing = time.time()

        preprocessing_duration = end_preprocessing - start_preprocessing
        preprocessing_times.append(preprocessing_duration)

        start_time = time.time()
        encrypted_output = fhe_model_server.run(encrypted_input, serialized_evaluation_keys)
        end_time = time.time()

        result = fhe_model_client.deserialize_decrypt_dequantize(encrypted_output)
        predicted_label = 1 if result[0][1] > 0.5 else 0

        duration = end_time - start_time
        inference_times.append(duration)

        true_label_text = test_labels.iloc[i]
        true_label_binary = 0 if true_label_text == 'benign' else 1

        if (i + 1) % report_interval == 0:
            avg_inf = np.mean(inference_times)
            avg_prep = np.mean(preprocessing_times)
            log_time(f"[{config_tag}] Processed {i+1}/{records} records (avg inference: {avg_inf:.4f}s, avg encrypt: {avg_prep:.6f}s)")

    log_time(f"[{config_tag}] [STEP 4/4] All {records} records processed. Computing statistics...")

    total_records = len(inference_times)
    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if total_records > 0 else 0
    mean_preprocessing_time = np.mean(preprocessing_times) if total_records > 0 else 0

    print("\n" + "="*20 + " INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records}")
    print(f"   Total inference time: {total_inference_time:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time:.6f} seconds")
    print(f"Mean preprocessing time/record: {mean_preprocessing_time:.6f} seconds")
    print("="*61 + "\n")


if __name__ == "__main__":
    configs = [
        ("plaintext", 2, 4),
        ("plaintext", 4, 2),
        ("plaintext", 4, 4),
        ("fhe", 2, 4, 1000),
        ("fhe", 4, 2, 1000),
        ("fhe", 4, 4, 1000),
    ]
    total = len(configs)
    log_time(f"Starting benchmark suite — {total} configurations to run")

    for idx, cfg in enumerate(configs, 1):
        log_time(f"=== Configuration {idx}/{total} ===")
        if cfg[0] == "plaintext":
            predict_single_record_plaintext(cfg[1], cfg[2])
        else:
            predict_single_record_with_comparison(cfg[1], cfg[2], cfg[3])

    log_time(f"All {total} configurations complete.")
