# Preparation to match Scapy.

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
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # Re-added OneHotEncoder
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

# --- Script Start ---
log_time()
print(f"Scikit-learn version: {sklearn.__version__}")

# --- Data Assembly and Loading ---
data_folder = 'CICIDS2017'
combined_csv_path = 'CIC-IDS-2017-Combined.csv'

if os.path.exists(combined_csv_path):
    print(f"Found existing combined file. Loading '{combined_csv_path}'...")
    df = pd.read_csv(combined_csv_path)
else:
    print(f"No combined file found. Assembling data from '{data_folder}' folder...")
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in the '{data_folder}' directory. Please check the path.")
    df_list = [pd.read_csv(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Successfully combined {len(all_files)} files.")
    df.to_csv(combined_csv_path, index=False)
    print(f"Combined data saved to '{combined_csv_path}'.")

def clean_col_names(df):
    """Cleans column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.strip().replace(' ', '_').lower() for col in cols]
    df.columns = new_cols
    return df

df = clean_col_names(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df['binary_label'] = (df['label'] != 'BENIGN').astype(int)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Data split into {len(train_df)} training samples and {len(test_df)} testing samples.")

features_to_use = [
    'flow_iat_mean', 
    'destination_port', 
    'total_length_of_fwd_packets', 
    'total_length_of_fwd_packets'
]

categorical_features = ['destination_port']
numerical_features = ['total_length_of_fwd_packets', 'total_length_of_bwd_packets', 'flow_iat_mean']

# The preprocessor now handles both categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Apply the preprocessing pipeline
X_train = preprocessor.fit_transform(train_df)
X_test = preprocessor.transform(test_df)

y_train = train_df['binary_label']
y_test = test_df['binary_label']

sample_size = 80000
if X_train.shape[0] > sample_size:
    np.random.seed(42) # for reproducibility
    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_train_sampled = X_train[indices]
    y_train_sampled = y_train.iloc[indices]
else:
    X_train_sampled = X_train
    y_train_sampled = y_train

with open('preprocessor_cic_kdd_equivalent.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)


n_estimators_list = [2, 100]
max_depth_list = [2, 4]

for n_estimators, max_depth in zip(n_estimators_list, max_depth_list):
    print("\n" + "="*60)
    print(f"STARTING TEST FOR n_estimators = {n_estimators}, max_depth = {max_depth}")
    print("="*60)

    log_time()
    print(f"Training RandomForestClassifier...")
    # Use .toarray() because OneHotEncoder produces a sparse matrix
    classifier = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_train_sampled.toarray(), y_train_sampled)

    log_time()
    print("Start prediction in the clear...")
    y_pred = classifier.predict(X_test.toarray())
    log_time()
    print("Plain text model metrics:")
    log_model_metrics(y_test, y_pred)

    log_time()
    print("Compiling FHE model...")
    fhe_classifier = classifier.compile(X_train.toarray())
    log_time()
    print("Finished FHE model compilation.")

    log_time()
    print("Making FHE simulation prediction...")
    y_pred_fhe = fhe_classifier.predict(X_test.toarray(), fhe="simulate")
    log_time()
    print("FHE model metrics (simulated):")
    log_model_metrics(y_test, y_pred_fhe)

    print("\nSaving compiled FHE circuit and client assets to disk...")
    dev = FHEModelDev(f"./fhe_model_{n_estimators}_estimators_{max_depth}_depth_kdd_eq/", fhe_classifier)
    dev.save()
    log_time()
    print("FHE assets saved.")
