# train_and_compile.py
#
# This script trains the model, compiles it for FHE, and saves the
# compiled model and the preprocessor to disk.
# Run this script only once to prepare the necessary assets.

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from concrete.ml.sklearn.rf import RandomForestClassifier as ConcreteRandomForestClassifier
from zoneinfo import ZoneInfo
import datetime
import pickle

def log_time():
    """Prints the current time in Brasília timezone."""
    brasilia_tz = ZoneInfo("America/Sao_Paulo")
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    brasilia_now = utc_now.astimezone(brasilia_tz)
    formatted_time = brasilia_now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"[LOG] Current time: {formatted_time}")

def train_and_save_model():
    """
    Trains the classifier, compiles it for FHE, and saves the necessary
    components (FHE circuit, preprocessor) to disk.
    """
    log_time()
    print("--- Training and Compiling FHE Model ---")

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

    try:
        train_df = pd.read_csv('Train.txt', delimiter=',', header=None, names=column_names)
    except FileNotFoundError:
        print("Error: Train.txt not found.")
        return

    # Convert multi-class labels to binary
    train_df['binary_label'] = (train_df['label'] != 'normal').astype(int)

    # Feature selection and preprocessing setup
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

    X_train = preprocessor.fit_transform(train_df[features_to_use])
    y_train = train_df['binary_label']

    # --- Model Training ---
    # Reduced n_estimators for faster compilation and prediction.
    # This is a key parameter to tune for performance vs. accuracy.
    fhe_classifier = ConcreteRandomForestClassifier(n_estimators=3)
    print("Training Concrete-ML RandomForestClassifier with n_estimators=3...")
    fhe_classifier.fit(X_train.toarray(), y_train)

    # --- FHE Compilation ---
    log_time()
    print("Compiling FHE model (this is the time-intensive part)...")
    fhe_classifier.compile(X_train.toarray())
    log_time()
    print("Compilation complete.")

    # --- Save Artifacts ---
    print("Saving compiled FHE circuit and preprocessor to disk...")
    # The compiled model (FHE circuit) is saved.
    fhe_classifier.dump("fhe_model_and_circuit.zip")

    # Save the preprocessor to be used for new data
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print("\nTraining and compilation finished. You can now use 'predict_fhe.py' for fast predictions.")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    train_and_save_model()


# predict_fhe.py
#
# This script loads the pre-compiled FHE model and makes a fast prediction
# on a single record.

import pandas as pd
from concrete.ml.deployment import FHEModelClient, FHEModelDev
import pickle
import numpy as np
from zoneinfo import ZoneInfo
import datetime

def log_time():
    """Prints the current time in Brasília timezone."""
    brasilia_tz = ZoneInfo("America/Sao_Paulo")
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    brasilia_now = utc_now.astimezone(brasilia_tz)
    formatted_time = brasilia_now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"[LOG] Current time: {formatted_time}")

def predict_single_record():
    """
    Loads a pre-compiled FHE model and predicts on a single record from the test set.
    """
    log_time()
    print("--- FHE Fast Prediction ---")

    # --- Load Pre-compiled Model and Preprocessor ---
    print("Loading pre-compiled FHE circuit and preprocessor...")
    # The FHEModelDev class is used to load the full model development environment
    # which includes the compiled circuit.
    fhe_model_dev = FHEModelDev("fhe_model_and_circuit.zip")
    
    # The FHEModelClient contains only what's needed for prediction.
    fhe_model_client = FHEModelClient(fhe_model_dev.client_specs)


    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # --- Prepare Single Record for Prediction ---
    # Load the test data to get a sample record
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
    try:
        test_df = pd.read_csv('Test.txt', delimiter=',', header=None, names=column_names)
    except FileNotFoundError:
        print("Error: Test.txt not found.")
        return

    features_to_use = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']
    single_record_index = 0
    single_record_df = test_df[features_to_use].iloc[[single_record_index]]

    # Preprocess the single record using the loaded preprocessor
    X_single_record_processed = preprocessor.transform(single_record_df).toarray()

    print(f"\nOriginal Features for Record {single_record_index}:\n{single_record_df.iloc[0]}")

    # --- Make FHE Prediction ---
    log_time()
    print("Making FHE prediction (this should be much faster now)...")
    
    # Quantize the input using the client specs
    quantized_input = fhe_model_client.quantize(X_single_record_processed)
    
    # Encrypt, predict, and decrypt
    encrypted_input = fhe_model_client.encrypt(quantized_input)
    encrypted_output = fhe_model_dev.server.run(encrypted_input)
    decrypted_output = fhe_model_client.decrypt(encrypted_output)

    log_time()
    print("Done predicting using FHE.")

    print("\nDecrypted Result (from FHE execution):")
    print(f"Predicted Label: {decrypted_output[0]}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    predict_single_record()
