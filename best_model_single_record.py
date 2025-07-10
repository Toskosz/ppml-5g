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
import time

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

    print(f"Found {len(test_df)} records to process.")
    features_to_use = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']
    X_test = test_df[features_to_use]

    print("\n[STEP 3] Processing all records...")
    inference_times = []

    serialized_evaluation_keys = fhe_model_client.get_serialized_evaluation_keys()

    for i in range(len(X_test)):
        single_record_df = X_test.iloc[[i]]
        
        X_single_record_processed = preprocessor.transform(single_record_df).toarray()

        encrypted_input = fhe_model_client.quantize_encrypt_serialize(X_single_record_processed)

        start_time = time.time()

        encrypted_output = fhe_model_server.run(encrypted_input, serialized_evaluation_keys)

        end_time = time.time()

        result = fhe_model_client.deserialize_decrypt_dequantize(encrypted_output)

        duration = end_time - start_time
        inference_times.append(duration)

        true_label_text = test_df.iloc[i]['label']
        true_label_binary = 1 if true_label_text != 'normal' else 0

        print(f"Record {i+1}/{len(X_test)} | Predicted: {result[0]} | True: {true_label_binary} | Time: {duration:.4f}s")

    # --- 4. Calculate and Display Results ---
    print("\n[STEP 4] Calculating final statistics...")
    log_time()

    total_records = len(inference_times)
    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if total_records > 0 else 0

    print("\n" + "="*20 + " INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records}")
    print(f"   Total inference time: {total_inference_time:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time:.4f} seconds")
    print("="*61 + "\n")

if __name__ == "__main__":
    predict_single_record_with_comparison(2)
