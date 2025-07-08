# predict_fhe.py
#
# This script loads the pre-compiled FHE model and makes a fast prediction
# on a single record, showing the data's state before and after encryption.

import pandas as pd
from concrete.ml.deployment import FHEModelClient, FHEModelServer
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

def predict_single_record_with_comparison(n_estimators):
    """
    Loads a pre-compiled FHE model, predicts on a single record, and
    compares the data's state before encryption and after decryption.
    """
    log_time()
    print("--- FHE Prediction with Before & After Comparison ---")

    # --- 1. Load Pre-compiled Model and Preprocessor ---
    print("\n[STEP 1] Loading pre-compiled FHE circuit and preprocessor...")
    try:
        fhe_model_server = FHEModelServer(f"./fhe_model_{n_estimators}_estimators/")
        fhe_model_server.load()
        fhe_model_client = FHEModelClient(f"./fhe_model_{n_estimators}_estimators/")
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please run the 'train.py' script first to generate the necessary files.")
        return

    # --- 2. Prepare and Compare Data Before Encryption ---
    print("\n[STEP 2] Preparing and inspecting data on the CLIENT-SIDE before encryption...")
    
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

    # === BEFORE ENCRYPTION: Stage A (Original Data) ===
    print("\n" + "="*20 + " BEFORE ENCRYPTION " + "="*20)
    print("\n(A) Original Human-Readable Features:")
    print(single_record_df.to_string())

    # Preprocess the single record
    X_single_record_processed = preprocessor.transform(single_record_df).toarray()

    # === BEFORE ENCRYPTION: Stage B (Preprocessed Data) - IMPROVED VIEW ===
    print("\n(B) Data After Preprocessing (Showing values for each new feature):")

    processed_series = pd.Series(X_single_record_processed[0], index=preprocessor.get_feature_names_out())
    print(processed_series.to_string())

    # --- 3. Quantize, Encrypt, Predict, and Decrypt ---
    print("\n[STEP 3] Executing the FHE workflow...")
    
    # === BEFORE ENCRYPTION: Stage C (Quantized Data) ===
    encrypted_input = fhe_model_client.quantize_encrypt_serialize(X_single_record_processed)
    print("\n(C) Data After Encrypt: ")
    print(encrypted_input.hex())
    print(f"Length: {len(encrypted_input)}")
    print("="*61 + "\n")

    serialized_evaluation_keys = fhe_model_client.get_serialized_evaluation_keys()


    log_time()
    print("Running inference on SERVER-SIDE (on encrypted data)...")
    encrypted_output = fhe_model_server.run(encrypted_input, serialized_evaluation_keys)
    log_time()
    print("Inference complete.")
    
    print("Decrypting result on CLIENT-SIDE...")
    result = fhe_model_client.deserialize_decrypt_dequantize(encrypted_output)

    # --- 4. Final Result After Decryption ---
    print("\n[STEP 4] Inspecting the final result after decryption...")
    print("\n" + "="*21 + " AFTER DECRYPTION " + "="*21)
    true_label_text = test_df.iloc[single_record_index]['label']
    true_label_binary = 1 if true_label_text != 'normal' else 0

    print("\nFinal Decrypted Prediction:")
    print(f"Predicted Label: {result[0]} (0 = normal, 1 = attack)")
    print(f"    True Label: {true_label_binary} (from '{true_label_text}')")
    print("="*61 + "\n")


if __name__ == "__main__":
    estimators = [2, 5, 10, 25, 50, 100]
    for n in estimators:
        predict_single_record_with_comparison(n)
