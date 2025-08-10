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

csv_path = 'NF-UNSW-NB15-v3.csv'

if os.path.exists(csv_path):
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError(f"Dataset file not found: '{csv_path}'. Please check the path.")

def clean_col_names(df):
    """Cleans column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.strip().replace(' ', '_').replace('/', '_').lower() for col in cols]
    df.columns = new_cols
    return df

df = clean_col_names(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Initial split into full train and test sets
# Using 'attack' for stratification as it's the original label
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Data split into {len(train_df)} training samples and {len(test_df)} testing samples.")

numerical_features_selected = [
    'in_bytes',
    'out_bytes'
]
categorical_features_selected = ['protocol', 'l7_proto']

# Preprocessor will output a sparse matrix due to OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features_selected),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_selected)
    ],
    remainder='drop'
)

# Apply the preprocessing pipeline
print("Applying preprocessing (MinMaxScaler and OneHotEncoder)...")
X_train_sparse = preprocessor.fit_transform(train_df)
X_test_sparse = preprocessor.transform(test_df)
print(f"Shape after initial preprocessing (sparse) - X_train: {X_train_sparse.shape}, X_test: {X_test_sparse.shape}")

y_train_full = train_df['label']
y_test_full = test_df['label']

# Delete DataFrames early to free memory
del train_df
del test_df
del df

with open('preprocessor_netflow.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)


X_train_final = X_train_sparse
y_train_final = y_train_full
X_test_final = X_test_sparse
y_test_final = y_test_full

print(f"Final training data shape: {X_train_final.shape}")
print(f"Final testing data shape: {X_test_final.shape}")

n_estimators_list = [2, 4, 2, 4, 2, 4]
max_depth_list = [8, 8, 16, 16, 32, 32]

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
    classifier.fit(X_train_final.toarray(), y_train_final)

    log_time()
    print("Start prediction in the clear...")
    y_pred = classifier.predict(X_test_final.toarray())
    log_time()
    print("Plain text model metrics:")
    log_model_metrics(y_test_final, y_pred)
