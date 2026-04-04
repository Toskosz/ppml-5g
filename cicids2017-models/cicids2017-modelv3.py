# Dataset: CSE-CIC-IDS-2018
# Replaces: CIC-IDS-2017 (superseded by IDS-2018 with larger scale and same CICFlowMeter feature schema)
# Download: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/

from concrete.ml.deployment import FHEModelDev
from concrete.ml.sklearn.rf import RandomForestClassifier
import datetime
import glob
import numpy as np
import os
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from zoneinfo import ZoneInfo
from sklearn.decomposition import TruncatedSVD # Import TruncatedSVD

def log_time():
    """Logs the current time in Brasília timezone."""
    brasilia_tz = ZoneInfo("America/Sao_Paulo")
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    brasilia_now = utc_now.astimezone(brasilia_tz)
    formatted_time = brasilia_now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"[LOG] Current time: {formatted_time}")

def log_model_metrics(y_test, y_pred):
    """Prints a standardized set of classification metrics."""
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

log_time()
print(f"Scikit-learn version: {sklearn.__version__}")

data_folder = 'CIC-IDS-2018'
combined_csv_path = 'CIC-IDS-2018-Combined.csv'

if os.path.exists(combined_csv_path):
    print(f"Found existing combined file. Loading '{combined_csv_path}'...")
    df = pd.read_csv(combined_csv_path)
else:
    print(f"No combined file found. Assembling data from '{data_folder}' folder...")
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_folder}'. "
            f"Download with: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ {data_folder}/"
        )
    df_list = [pd.read_csv(file, low_memory=False) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Successfully combined {len(all_files)} files.")
    df.to_csv(combined_csv_path, index=False)
    print(f"Combined data saved to '{combined_csv_path}'.")

def clean_col_names(df):
    """Cleans column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.strip().replace(' ', '_').replace('/', '_').lower() for col in cols]
    df.columns = new_cols
    return df

df = clean_col_names(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df['binary_label'] = (df['label'] != 'benign').astype(int)

# Initial split into full train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Data split into {len(train_df)} training samples and {len(test_df)} testing samples.")

# CICFlowMeter-V3 features available in CSE-CIC-IDS-2018
# Note: total_length_of_bwd_packets from IDS-2017 has no direct equivalent in IDS-2018;
# dropped in favour of a clean 5-feature numerical set.
numerical_features_selected = ['syn_cnt', 'ack_cnt', 'fin_cnt', 'rst_cnt', 'tot_l_fw_pkt']
categorical_features_selected = ['protocol', 'dst_port']

# Preprocessor will output a sparse matrix due to OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features_selected),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_selected)
    ],
    remainder='drop'
)

# Apply the preprocessing pipeline
# X_train_sparse and X_test_sparse will be sparse matrices
print("Applying preprocessing (MinMaxScaler and OneHotEncoder)...")
X_train_sparse = preprocessor.fit_transform(train_df)
X_test_sparse = preprocessor.transform(test_df)
print(f"Shape after initial preprocessing (sparse) - X_train: {X_train_sparse.shape}, X_test: {X_test_sparse.shape}")

y_train_full = train_df['binary_label'] # Renamed to y_train_full for clarity
y_test_full = test_df['binary_label']   # Renamed to y_test_full for clarity

# Delete DataFrames early to free memory as we have X_sparse now
del train_df
del test_df
del df

with open('preprocessor_cic_ids_2018.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)


# --- Dimensionality Reduction using TruncatedSVD ---
n_components_svd = 100 # <-- CRITICAL: Adjust this value
print(f"\nApplying TruncatedSVD to reduce feature count from {X_train_sparse.shape[1]} to {n_components_svd}...")

svd = TruncatedSVD(n_components=n_components_svd, random_state=42)

X_train_dense_reduced = svd.fit_transform(X_train_sparse)
X_test_dense_reduced = svd.transform(X_test_sparse)

print(f"Shape after SVD reduction (dense) - X_train: {X_train_dense_reduced.shape}, X_test: {X_test_dense_reduced.shape}")

with open('svd_cic_ids_2018.pkl', 'wb') as f:
    pickle.dump(svd, f)
print("SVD saved to 'svd_cic_ids_2018.pkl'.")

del X_train_sparse
del X_test_sparse

X_train_final = X_train_dense_reduced
y_train_final = y_train_full

X_test_final = X_test_dense_reduced
y_test_final = y_test_full

print(f"Final training data shape: {X_train_final.shape}")
print(f"Final testing data shape: {X_test_final.shape}")

n_estimators_list = [2]
max_depth_list = [4]

for n_estimators, max_depth in zip(n_estimators_list, max_depth_list):
    print("\n" + "="*60)
    print(f"STARTING TEST FOR n_estimators = {n_estimators}, max_depth = {max_depth}")
    print("="*60)

    log_time()
    print(f"Training RandomForestClassifier...")
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    classifier.fit(X_train_final, y_train_final)

    log_time()
    print("Start prediction in the clear...")
    y_pred = classifier.predict(X_test_final)
    log_time()
    print("Plain text model metrics:")
    log_model_metrics(y_test_final, y_pred)

    log_time()
    print("Compiling model to FHE circuit...")
    classifier.compile(X_train_final)
    log_time()
    print("FHE compilation complete. Saving FHE model assets...")
    fhe_model_dir = f"./cicids2017-models/fhe_model_{n_estimators}_estimators_{max_depth}_depth_svd_{n_components_svd}_components"
    os.makedirs(fhe_model_dir, exist_ok=True)
    dev = FHEModelDev(fhe_model_dir, classifier)
    dev.save()
    log_time()
    print(f"FHE model saved to '{fhe_model_dir}'.")

