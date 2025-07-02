# Preparation to match Scapy.

import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from concrete.ml.sklearn.rf import RandomForestClassifier as ConcreteRandomForestClassifier
from sklearn.metrics import accuracy_score
from zoneinfo import ZoneInfo
import datetime
import numpy as np

def log_time():
    """Prints the current time in Brasília timezone."""
    brasilia_tz = ZoneInfo("America/Sao_Paulo")
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    brasilia_now = utc_now.astimezone(brasilia_tz)
    formatted_time = brasilia_now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"[LOG] Current time: {formatted_time}")

log_time()
print(f"Scikit-learn version: {sklearn.__version__}")

# Load data with correct headers
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

# Load datasets from local files
try:
    train_df = pd.read_csv('Train.txt', delimiter=',', header=None, names=column_names)
    test_df = pd.read_csv('Test.txt', delimiter=',', header=None, names=column_names)
except FileNotFoundError:
    print("Error: Train.txt or Test.txt not found. Please ensure the files are in the correct directory.")
    exit()


# Function to convert dataset flags to Scapy-compatible flags
def convert_flag_to_scapy(flag):
    flag_translation = {
        'REJ': 'R', 'SF': 'PA', 'S0': 'S', 'RSTO': 'R', 'S1': 'S',
        'S2': 'S', 'S3': 'S', 'RSTOS0': 'R', 'OTH': ''
    }
    return ''.join([flag_translation.get(f, '') for f in flag])

# Apply the flag conversion
train_df['flag'] = train_df['flag'].apply(convert_flag_to_scapy)
test_df['flag'] = test_df['flag'].apply(convert_flag_to_scapy)

# Convert multi-class labels to binary: 1 for any attack, 0 for normal
train_df['binary_label'] = (train_df['label'] != 'normal').astype(int)
test_df['binary_label'] = (test_df['label'] != 'normal').astype(int)

# Selecting features and defining categorical/numerical columns
features_to_use = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']
categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = ['src_bytes', 'dst_bytes']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Apply preprocessing
X_train = preprocessor.fit_transform(train_df[features_to_use])
X_test = preprocessor.transform(test_df[features_to_use])
y_train = train_df['binary_label']
y_test = test_df['binary_label']

# --- Select a single record for demonstration ---
single_record_index = 0
X_single_record_processed = X_test[single_record_index].toarray()
original_record_features = test_df[features_to_use].iloc[single_record_index]
actual_label = y_test.iloc[single_record_index]

print("\n" + "="*80)
print("DEMONSTRATION ON A SINGLE RECORD")
print("="*80)
print(f"Selected Record (Index: {single_record_index}) Original Features:\n{original_record_features}")
print(f"\nActual Label: {actual_label} ('1' means attack, '0' means normal)")
print(f"\nPreprocessed (scaled and one-hot encoded) record:\n{X_single_record_processed[0]}")
print("="*80 + "\n")


# --- 1. Non-FHE (Standard Scikit-learn) Prediction ---
print("--- 1. Non-FHE / Cleartext Prediction ---")
log_time()
print("Training standard scikit-learn RandomForestClassifier...")
# Using the standard sklearn classifier for the non-FHE part
sklearn_classifier = SklearnRandomForestClassifier(n_estimators=10, random_state=42)
sklearn_classifier.fit(X_train.toarray(), y_train)

# Make prediction on the single record
prediction_clear = sklearn_classifier.predict(X_single_record_processed)
prediction_proba_clear = sklearn_classifier.predict_proba(X_single_record_processed)

print("\nOriginal Cypher (Processed Data):")
print(X_single_record_processed[0])
print("\nPrediction Result (Cleartext):")
print(f"Predicted Label: {prediction_clear[0]}")
print(f"Prediction Probabilities: {prediction_proba_clear[0]}")
print("-" * 40 + "\n")


# --- 2. FHE (Concrete-ML) Prediction ---
print("--- 2. FHE / Encrypted Prediction ---")
log_time()
print("Training Concrete-ML RandomForestClassifier...")
# Using the Concrete-ML classifier for the FHE part
fhe_classifier = ConcreteRandomForestClassifier(n_estimators=10)
fhe_classifier.fit(X_train.toarray(), y_train)

log_time()
print("Compiling FHE model (this may take a moment)...")
fhe_classifier.compile(X_train.toarray())

# The "encryption" happens implicitly when you pass data to the FHE predict method.
# For demonstration, we can show the quantized version of the data which is what gets encrypted.
quantized_input = fhe_classifier.quantize_input(X_single_record_processed)[0]

log_time()
print("Making FHE prediction on the single record...")
# Run inference on the single encrypted input
prediction_fhe = fhe_classifier.predict(X_single_record_processed, fhe="execute")
prediction_proba_fhe = fhe_classifier.predict_proba(X_single_record_processed, fhe="execute")


print("\nOriginal Cypher (Processed Data):")
print(X_single_record_processed[0])
print("\nQuantized Cypher (Data that gets encrypted):")
print(quantized_input)
print("\nEncrypted Result (this is an encrypted value, not directly viewable):")
print("The FHE execution returns a decrypted result directly for convenience.")
print("Internally, the prediction is computed on encrypted data.")
print("\nDecrypted Result (from FHE execution):")
print(f"Predicted Label: {prediction_fhe[0]}")
print(f"Prediction Probabilities: {prediction_proba_fhe[0]}")
print("-" * 40 + "\n")


# --- 3. Comparison and Summary ---
print("--- 3. Summary of Results ---")
print(f"Original Record Features:\n{original_record_features}\n")
print(f"Actual Label: \t\t{actual_label}")
print(f"Non-FHE Prediction: \t{prediction_clear[0]}")
print(f"FHE Prediction: \t\t{prediction_fhe[0]}")
print(f"\nAre Non-FHE and FHE predictions the same? {'Yes' if prediction_clear[0] == prediction_fhe[0] else 'No'}")
log_time()
print("Done.")
