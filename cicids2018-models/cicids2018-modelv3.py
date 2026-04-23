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

numerical_features_selected = ['syn_flag_cnt', 'ack_flag_cnt', 'fin_flag_cnt', 'rst_flag_cnt', 'totlen_fwd_pkts']
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


# --- Dimensionality Reduction + Training Loop ---
# Configs: (n_estimators, max_depth, n_components_svd)
configs = [
    (2, 2, 100),
    (2, 2, 200),
    (4, 2, 100),
    (4, 2, 200),
    (4, 4, 100),
    (4, 4, 200),
]

for n_components_svd in sorted(set(c[2] for c in configs)):
    print(f"\n{'#'*60}")
    print(f"# SVD with {n_components_svd} components")
    print(f"{'#'*60}")

    print(f"\nApplying TruncatedSVD ({X_train_sparse.shape[1]} → {n_components_svd})...")
    svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
    X_train_final = svd.fit_transform(X_train_sparse)
    X_test_final = svd.transform(X_test_sparse)
    print(f"SVD done. X_train: {X_train_final.shape}, X_test: {X_test_final.shape}. "
          f"Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    svd_pkl_path = f'svd_cic_ids_2018_{n_components_svd}.pkl'
    with open(svd_pkl_path, 'wb') as f:
        pickle.dump(svd, f)
    print(f"SVD saved to '{svd_pkl_path}'.")

    y_train_final = y_train_full
    y_test_final = y_test_full

    for n_estimators, max_depth, _ in [c for c in configs if c[2] == n_components_svd]:
        print("\n" + "="*60)
        print(f"STARTING TEST FOR n_estimators={n_estimators}, max_depth={max_depth}, svd={n_components_svd}")
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
        fhe_model_dir = f"./fhe_model_{n_estimators}_estimators_{max_depth}_depth_svd_{n_components_svd}_components"
        os.makedirs(fhe_model_dir, exist_ok=True)
        dev = FHEModelDev(fhe_model_dir, classifier)
        dev.save()
        log_time()
        print(f"FHE model saved to '{fhe_model_dir}'.")

        del classifier

    del X_train_final
    del X_test_final
    del svd

