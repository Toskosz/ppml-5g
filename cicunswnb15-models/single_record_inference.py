# Dataset: CIC-UNSW-NB15 (2024)
# Task: multi-class classification (9 attack categories + benign)
# Prerequisite: run cicunswnb15-models/model.py first to produce:
#   - preprocessor_cicunsw.pkl
#   - svd_cicunsw.pkl
#   - cicunswnb15-models/fhe_model_<n>_estimators_<d>_depth_svd_100_components/

from concrete.ml.deployment import FHEModelClient, FHEModelServer
from concrete.ml.sklearn.rf import RandomForestClassifier
import datetime
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from zoneinfo import ZoneInfo
import time


def log_time():
    """Prints the current time in Brasília timezone."""
    brasilia_tz = ZoneInfo("America/Sao_Paulo")
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    brasilia_now = utc_now.astimezone(brasilia_tz)
    formatted_time = brasilia_now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"[LOG] Current time: {formatted_time}")


def clean_col_names(df):
    """Cleans column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.strip().replace(' ', '_').replace('/', '_').lower() for col in cols]
    df.columns = new_cols
    return df


def load_and_preprocess():
    """
    Loads CIC-UNSW-NB15 data and applies the saved preprocessor + SVD.
    Returns X_train_final, X_test_final, y_train_full, y_test_full, test_labels.
    Requires preprocessor_cicunsw.pkl and svd_cicunsw.pkl to exist
     (produced by cicunswnb15-models/model.py).
    """
    csv_path = 'CICFlowMeter_out.csv'

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found: '{csv_path}'. "
            "Download CIC-UNSW-NB15 from http://cicresearch.ca/CICDataset/CIC-UNSW/"
        )

    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path, low_memory=False)

    df = clean_col_names(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Data split: {len(train_df)} train / {len(test_df)} test samples.")

    y_train_full = train_df['label']
    y_test_full = test_df['label']
    test_labels = test_df['label'].reset_index(drop=True)

    del df

    print("Loading saved preprocessor...")
    with open('preprocessor_cicunsw.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    X_train_sparse = preprocessor.transform(train_df)
    X_test_sparse = preprocessor.transform(test_df)

    del train_df
    del test_df

    print("Loading saved SVD...")
    with open('svd_cicunsw.pkl', 'rb') as f:
        svd = pickle.load(f)

    X_train_final = svd.transform(X_train_sparse)
    X_test_final = svd.transform(X_test_sparse)

    print(f"Final shapes — X_train: {X_train_final.shape}, X_test: {X_test_final.shape}")

    return X_train_final, X_test_final, y_train_full, y_test_full, test_labels


def predict_single_record_plain_text(estimators, depth, n_components_svd=100):
    log_time()
    print(f"\n--- Plaintext Inference | estimators={estimators}, depth={depth} ---")

    X_train_final, X_test_final, y_train_full, _, test_labels = load_and_preprocess()

    classifier = RandomForestClassifier(
        n_estimators=estimators,
        max_depth=depth,
        random_state=42
    )
    classifier.fit(X_train_final, y_train_full)

    inference_times = []

    for i in range(1000):
        single_record = X_test_final[i:i+1]

        start_time = time.time()
        result = classifier.predict(single_record)
        end_time = time.time()

        duration = end_time - start_time
        inference_times.append(duration)

        true_label = test_labels.iloc[i]

#        print(f"Record {i+1}/1000 | Predicted: {result[0]} | True: {true_label} | Time: {duration:.4f}s")

    print("\n[STEP 4] Calculating final statistics...")
    log_time()

    total_records = len(inference_times)
    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if total_records > 0 else 0

    print("\n" + "="*20 + " INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records}")
    print(f"   Total inference time: {total_inference_time:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time:.6f} seconds")
    print("="*61 + "\n")


def predict_single_record_with_comparison(estimators, depth, records=1000, n_components_svd=100):
    """
    Loads a pre-compiled FHE model and runs encrypted inference on individual records.
    Uses argmax over the 9-class output vector to determine the predicted class.
    """
    log_time()
    print(f"\n--- FHE Inference | estimators={estimators}, depth={depth} ---")

     model_dir = f"./cicunswnb15-models/fhe_model_{estimators}_estimators_{depth}_depth_svd_{n_components_svd}_components/"

    print("\n[STEP 1] Loading pre-compiled FHE circuit...")
    try:
        fhe_model_server = FHEModelServer(model_dir)
        fhe_model_server.load()
        fhe_model_client = FHEModelClient(model_dir)
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
         print("Please run cicunswnb15-models/model.py first to generate the FHE assets.")
        return

    print("\n[STEP 2] Preparing data on the CLIENT-SIDE before encryption...")
    _, X_test_final, _, _, test_labels = load_and_preprocess()
    print(f"Found {len(test_labels)} records to process.")

    print("\n[STEP 3] Processing records...")
    inference_times = []
    preprocessing_times = []

    serialized_evaluation_keys = fhe_model_client.get_serialized_evaluation_keys()

    for i in range(records):
        single_record = X_test_final[i:i+1]

        start_preprocessing = time.time()
        encrypted_input = fhe_model_client.quantize_encrypt_serialize(single_record)
        end_preprocessing = time.time()
        preprocessing_times.append(end_preprocessing - start_preprocessing)

        start_time = time.time()
        encrypted_output = fhe_model_server.run(encrypted_input, serialized_evaluation_keys)
        end_time = time.time()

        result = fhe_model_client.deserialize_decrypt_dequantize(encrypted_output)
        # Multi-class: pick the class with the highest probability
        predicted_label = int(np.argmax(result[0]))

        duration = end_time - start_time
        inference_times.append(duration)

        true_label = test_labels.iloc[i]

#        print(f"Record {i+1}/{records} | Predicted: {predicted_label} | True: {true_label} | Time: {duration:.4f}s")

    print("\n[STEP 4] Calculating final statistics...")
    log_time()

    total_records = len(inference_times)
    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if total_records > 0 else 0
    mean_preprocessing_time = np.mean(preprocessing_times) if total_records > 0 else 0

    print("\n" + "="*20 + " INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records}")
    print(f"   Total inference time: {total_inference_time:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time:.6f} seconds")
    print(f"Mean preprocessing time/record: {mean_preprocessing_time:.6f} seconds")
    print("="*61 + "\n")


if __name__ == "__main__":
    models_to_test = [
        {'estimators': 2, 'depth': 2},
        {'estimators': 4, 'depth': 2},
        {'estimators': 2, 'depth': 4},
        {'estimators': 4, 'depth': 4},
    ]

    for model_params in models_to_test:
        estimators = model_params['estimators']
        depth = model_params['depth']
        print(f"\n{'='*20} TESTING MODEL: estimators={estimators}, depth={depth} {'='*20}")
        predict_single_record_plain_text(estimators, depth)
        predict_single_record_with_comparison(estimators, depth, records=1000)
