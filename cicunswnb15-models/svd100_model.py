# Dataset: CIC-UNSW-NB15 (2024)
# Replaces: NF-UNSW-NB15-v3 (UQ hosting unavailable; switched to CICFlowMeter feature schema)
# Download: http://cicresearch.ca/CICDataset/CIC-UNSW/ (see https://www.unb.ca/cic/datasets/cic-unsw-nb15.html)
# Task: binary classification (benign vs attack)
# Features: CICFlowMeter output; model uses 5 numerical + 2 categorical
# This script: SVD 100 components only

from concrete.ml.deployment import FHEModelDev
from concrete.ml.sklearn.rf import RandomForestClassifier
from concrete.ml.common.serialization.dumpers import dump
import datetime
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
    cols = df.columns
    new_cols = [col.strip().replace(' ', '_').replace('/', '_').lower() for col in cols]
    df.columns = new_cols
    return df


log_time(f"Starting svd100_model.py — scikit-learn {sklearn.__version__}")

csv_path = 'CICFlowMeter_out.csv'

cols_needed = [
    'SYN Flag Count', 'ACK Flag Count', 'FIN Flag Count',
    'RST Flag Count', 'Total Length of Fwd Packet',
    'Protocol', 'Dst Port', 'Label'
]

if os.path.exists(csv_path):
    log_time(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path, usecols=cols_needed, low_memory=False)
else:
    raise FileNotFoundError(
        f"Dataset file not found: '{csv_path}'. "
        "Download CIC-UNSW-NB15 from http://cicresearch.ca/CICDataset/CIC-UNSW/"
    )

log_time("Cleaning column names and dropping NaN/Inf rows...")
df = clean_col_names(df)
rows_before = len(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

num_cols_to_downcast = [
    'syn_flag_count', 'ack_flag_count', 'fin_flag_count',
    'rst_flag_count', 'total_length_of_fwd_packet'
]
for col in num_cols_to_downcast:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
df.dropna(subset=num_cols_to_downcast, inplace=True)
log_time(f"Dropped {rows_before - len(df)} rows with NaN/Inf. Remaining: {len(df)} rows.")

df['label'] = df['label'].astype(str).str.strip().str.lower()
df['binary_label'] = (df['label'] != 'benign').astype(int)
log_time(f"Label distribution — benign: {(df['binary_label']==0).sum()}, attack: {(df['binary_label']==1).sum()}")

log_time("Splitting data 80/20 (stratified)...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
log_time(f"Train: {len(train_df)}, Test: {len(test_df)}.")

numerical_features_selected = ['syn_flag_count', 'ack_flag_count', 'fin_flag_count', 'rst_flag_count', 'total_length_of_fwd_packet']
categorical_features_selected = ['protocol', 'dst_port']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features_selected),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_selected)
    ],
    remainder='drop'
)

log_time("Fitting preprocessor on training data...")
X_train_sparse = preprocessor.fit_transform(train_df)
X_test_sparse = preprocessor.transform(test_df)
log_time(f"Preprocessing done. X_train: {X_train_sparse.shape}, X_test: {X_test_sparse.shape} (sparse)")

y_train_full = train_df['binary_label']
y_test_full = test_df['binary_label']

del train_df
del test_df
del df

with open('preprocessor_cicunsw.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
log_time("Preprocessor saved to 'preprocessor_cicunsw.pkl'.")

configs = [
    (2, 2, 100),
    (2, 4, 100),
    (4, 2, 100),
    (4, 4, 100),
]
total_configs = len(configs)

y_train_final = y_train_full
y_test_final = y_test_full

print(f"\n{'Config':>30s} | {'Phase':>20s} | {'Wall Time':>12s}")
print("-" * 70)

trained_configs = []
svd_groups = sorted(set(c[2] for c in configs))

for svd_idx, n_components_svd in enumerate(svd_groups, 1):
    group_configs = [c for c in configs if c[2] == n_components_svd]

    log_time(f"Applying TruncatedSVD: {X_train_sparse.shape[1]} → {n_components_svd} components  [SVD group {svd_idx}/{len(svd_groups)}]...")
    svd = TruncatedSVD(n_components=n_components_svd, random_state=42)

    svd_sample_size = 500_000
    n_rows = X_train_sparse.shape[0]
    if n_rows <= svd_sample_size:
        X_svd_fit = X_train_sparse
    else:
        log_time(f"Subsampling {svd_sample_size:,} rows from {n_rows:,} for SVD fitting...")
        sample_idx = np.random.choice(n_rows, svd_sample_size, replace=False)
        X_svd_fit = X_train_sparse[sample_idx]

    svd.fit(X_svd_fit)
    del X_svd_fit

    chunk_size = 500_000
    log_time(f"Transforming training data in chunks of {chunk_size:,}...")
    chunks = []
    for start in range(0, X_train_sparse.shape[0], chunk_size):
        end = min(start + chunk_size, X_train_sparse.shape[0])
        chunks.append(svd.transform(X_train_sparse[start:end]).astype(np.float32))
    X_train_final = np.vstack(chunks)
    del chunks

    log_time(f"Transforming test data in chunks of {chunk_size:,}...")
    chunks = []
    for start in range(0, X_test_sparse.shape[0], chunk_size):
        end = min(start + chunk_size, X_test_sparse.shape[0])
        chunks.append(svd.transform(X_test_sparse[start:end]).astype(np.float32))
    X_test_final = np.vstack(chunks)
    del chunks

    log_time(f"SVD done. X_train: {X_train_final.shape}, X_test: {X_test_final.shape}. Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}")

    svd_pkl_path = f'svd_cicunsw_{n_components_svd}.pkl'
    with open(svd_pkl_path, 'wb') as f:
        pickle.dump(svd, f)
    log_time(f"SVD saved to '{svd_pkl_path}'.")

    log_time(f"Final training data: {X_train_final.shape}, testing data: {X_test_final.shape}")

    print("\n" + "#" * 70)
    print(f"#  PHASE 1: PLAINTEXT TRAINING & EVALUATION  (SVD {n_components_svd})")
    print("#" * 70)

    for n_estimators, max_depth, _ in group_configs:
        idx = len(trained_configs) + 1
        config_tag = f"n={n_estimators}, d={max_depth}, svd={n_components_svd}"
        print(f"\n{'='*60}")
        print(f"  CONFIG {idx}/{total_configs}: n_estimators={n_estimators}, max_depth={max_depth}, svd={n_components_svd}")
        print(f"{'='*60}")
        sys.stdout.flush()

        t0 = _time.time()
        log_time(f"[{config_tag}] Training RandomForestClassifier...")
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
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
        log_model_metrics(y_test_final, y_pred)

        trained_configs.append({
            'idx': idx,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'n_components_svd': n_components_svd,
            'config_tag': config_tag,
            'classifier': classifier,
            'train_dur': train_dur,
            'pred_dur': pred_dur,
        })

        json_path = f"plaintext_model_{n_estimators}_estimators_{max_depth}_depth_svd_{n_components_svd}.json"
        with open(json_path, "w") as f:
            dump(classifier, f)
        log_time(f"[{config_tag}] Plaintext model saved to '{json_path}'.")

    print("\n" + "=" * 70)
    print(f"  PLAINTEXT SUMMARY  (SVD {n_components_svd})")
    print("=" * 70)
    print(f"{'Config':>30s} | {'Train (s)':>10s} | {'Predict (s)':>12s}")
    print("-" * 60)
    for cfg in trained_configs:
        if cfg['n_components_svd'] == n_components_svd:
            print(f"{cfg['config_tag']:>30s} | {cfg['train_dur']:>10.1f} | {cfg['pred_dur']:>12.1f}")
    print()

    log_time(f"Phase 1 complete for SVD {n_components_svd} — all plaintext variants trained and evaluated.")

    print("\n" + "#" * 70)
    print(f"#  PHASE 2: FHE COMPILATION & SIMULATION  (SVD {n_components_svd})")
    print("#" * 70)

    for cfg in trained_configs:
        if cfg['n_components_svd'] != n_components_svd:
            continue

        config_tag = cfg['config_tag']
        n_estimators = cfg['n_estimators']
        max_depth = cfg['max_depth']
        classifier = cfg['classifier']
        train_dur = cfg['train_dur']
        pred_dur = cfg['pred_dur']

        print(f"\n{'='*60}")
        print(f"  FHE for CONFIG {cfg['idx']}/{total_configs}: n_estimators={n_estimators}, max_depth={max_depth}, svd={n_components_svd}")
        print(f"{'='*60}")
        sys.stdout.flush()

        t0 = _time.time()
        log_time(f"[{config_tag}] Compiling FHE circuit (this may take a long time)...")
        classifier.compile(X_train_final)
        compile_dur = _time.time() - t0
        log_time(f"[{config_tag}] FHE compilation completed in {compile_dur:,.1f}s")

        t0 = _time.time()
        log_time(f"[{config_tag}] Starting FHE simulation prediction on {X_test_final.shape[0]} samples...")
        y_pred_fhe = classifier.predict(X_test_final, fhe="simulate")
        fhe_dur = _time.time() - t0
        log_time(f"[{config_tag}] FHE simulation completed in {fhe_dur:,.1f}s")

        log_time(f"[{config_tag}] FHE metrics:")
        log_model_metrics(y_test_final, y_pred_fhe)

        model_dir = f"./fhe_model_{n_estimators}_estimators_{max_depth}_depth_svd_{n_components_svd}_components/"
        log_time(f"[{config_tag}] Saving compiled FHE circuit to '{model_dir}'...")
        dev = FHEModelDev(model_dir, classifier)
        dev.save()
        log_time(f"[{config_tag}] FHE assets saved. Summary — train: {train_dur:,.1f}s, predict: {pred_dur:,.1f}s, compile: {compile_dur:,.1f}s, fhe_sim: {fhe_dur:,.1f}s")

        cfg['compile_dur'] = compile_dur
        cfg['fhe_dur'] = fhe_dur

    print("\n" + "=" * 70)
    print(f"  FHE SUMMARY  (SVD {n_components_svd})")
    print("=" * 70)
    print(f"{'Config':>30s} | {'Train (s)':>10s} | {'Predict (s)':>12s} | {'Compile (s)':>12s} | {'FHE Sim (s)':>12s}")
    print("-" * 90)
    for cfg in trained_configs:
        if cfg['n_components_svd'] == n_components_svd:
            print(f"{cfg['config_tag']:>30s} | {cfg['train_dur']:>10.1f} | {cfg['pred_dur']:>12.1f} | {cfg['compile_dur']:>12.1f} | {cfg['fhe_dur']:>12.1f}")
    print()

    log_time(f"FHE phase complete for SVD {n_components_svd}.")

    del X_train_final
    del X_test_final
    del svd

print("\n" + "#" * 70)
print("#  GLOBAL SUMMARY")
print("#" * 70)
print(f"{'Config':>30s} | {'Train (s)':>10s} | {'Predict (s)':>12s} | {'Compile (s)':>12s} | {'FHE Sim (s)':>12s}")
print("-" * 90)
for cfg in trained_configs:
    print(f"{cfg['config_tag']:>30s} | {cfg['train_dur']:>10.1f} | {cfg['pred_dur']:>12.1f} | {cfg['compile_dur']:>12.1f} | {cfg['fhe_dur']:>12.1f}")
print()

log_time("All configurations complete.")
