# Dataset: CSE-CIC-IDS-2018
# Replaces: KDD Cup 1999 / NSL-KDD (removed from CIC servers, deprecated)
# Download: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/
# Features: CICFlowMeter-V3 (83 features); model uses 5 numerical + 2 categorical

from concrete.ml.deployment import FHEModelDev
from concrete.ml.sklearn.rf import RandomForestClassifier
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


log_time(f"Starting model.py — scikit-learn {sklearn.__version__}")

numerical_features = ['syn_flag_cnt', 'ack_flag_cnt', 'fin_flag_cnt', 'rst_flag_cnt', 'totlen_fwd_pkts']
categorical_features = ['protocol', 'dst_port']
needed_cols = numerical_features + categorical_features + ['label']

data_folder = 'CIC-IDS-2018'
combined_csv_path = 'CIC-IDS-2018-Combined.csv'

if os.path.exists(combined_csv_path):
    log_time(f"Loading existing combined file '{combined_csv_path}'...")
    df = pd.read_csv(combined_csv_path)
    log_time(f"Loaded {len(df)} rows from '{combined_csv_path}'.")
else:
    log_time(f"No combined file found. Assembling data from '{data_folder}' folder...")
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
        for col in numerical_features:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(subset=needed_cols, inplace=True)
        for col in categorical_features:
            chunk[col] = chunk[col].astype(str)
        chunks.append(chunk[needed_cols])
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

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

print("Applying preprocessing (MinMaxScaler and OneHotEncoder)...")
log_time("Fitting preprocessor on training data...")
X_train_sparse = preprocessor.fit_transform(train_df)
X_test_sparse = preprocessor.transform(test_df)
log_time(f"Preprocessing done. X_train: {X_train_sparse.shape}, X_test: {X_test_sparse.shape} (sparse)")

y_train_full = train_df['binary_label']
y_test_full = test_df['binary_label']

del train_df
del test_df
del df

with open('preprocessor_cic_ids_2018.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
log_time("Preprocessor saved to 'preprocessor_cic_ids_2018.pkl'.")

n_components_svd = 100
log_time(f"Applying TruncatedSVD: {X_train_sparse.shape[1]} → {n_components_svd} components...")

svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
X_train_final = svd.fit_transform(X_train_sparse)
X_test_final = svd.transform(X_test_sparse)
log_time(f"SVD done. X_train: {X_train_final.shape}, X_test: {X_test_final.shape}. Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}")

with open('svd_cic_ids_2018.pkl', 'wb') as f:
    pickle.dump(svd, f)
log_time("SVD saved to 'svd_cic_ids_2018.pkl'.")

del X_train_sparse
del X_test_sparse

y_train_final = y_train_full
y_test_final = y_test_full

log_time(f"Final training data: {X_train_final.shape}, testing data: {X_test_final.shape}")
print(f"\n{'Config':>30s} | {'Phase':>20s} | {'Wall Time':>12s}")
print("-" * 70)

n_estimators_list = [2, 4, 4, 100]
max_depth_list = [2, 2, 4, 2]
total_configs = len(n_estimators_list)

print("\n" + "#" * 70)
print("#  PHASE 1: PLAINTEXT TRAINING & EVALUATION")
print("#" * 70)

trained_configs = []

for idx, (n_estimators, max_depth) in enumerate(zip(n_estimators_list, max_depth_list), 1):
    config_tag = f"n={n_estimators}, d={max_depth}"
    print(f"\n{'='*60}")
    print(f"  CONFIG {idx}/{total_configs}: n_estimators={n_estimators}, max_depth={max_depth}")
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
        'config_tag': config_tag,
        'classifier': classifier,
        'train_dur': train_dur,
        'pred_dur': pred_dur,
    })

print("\n" + "=" * 70)
print("  PLAINTEXT SUMMARY")
print("=" * 70)
print(f"{'Config':>20s} | {'Train (s)':>10s} | {'Predict (s)':>12s}")
print("-" * 50)
for cfg in trained_configs:
    print(f"{cfg['config_tag']:>20s} | {cfg['train_dur']:>10.1f} | {cfg['pred_dur']:>12.1f}")
print()

log_time("Phase 1 complete — all plaintext variants trained and evaluated.")

print("\n" + "#" * 70)
print("#  PHASE 2: FHE COMPILATION & SIMULATION")
print("#" * 70)

for cfg in trained_configs:
    config_tag = cfg['config_tag']
    n_estimators = cfg['n_estimators']
    max_depth = cfg['max_depth']
    classifier = cfg['classifier']
    train_dur = cfg['train_dur']
    pred_dur = cfg['pred_dur']

    print(f"\n{'='*60}")
    print(f"  FHE for CONFIG {cfg['idx']}/{total_configs}: n_estimators={n_estimators}, max_depth={max_depth}")
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
print("  FHE SUMMARY")
print("=" * 70)
print(f"{'Config':>20s} | {'Train (s)':>10s} | {'Predict (s)':>12s} | {'Compile (s)':>12s} | {'FHE Sim (s)':>12s}")
print("-" * 80)
for cfg in trained_configs:
    print(f"{cfg['config_tag']:>20s} | {cfg['train_dur']:>10.1f} | {cfg['pred_dur']:>12.1f} | {cfg['compile_dur']:>12.1f} | {cfg['fhe_dur']:>12.1f}")
print()

log_time("All configurations complete.")
