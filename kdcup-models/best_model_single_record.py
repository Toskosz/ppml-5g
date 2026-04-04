# Dataset: CSE-CIC-IDS-2018
# Replaces: KDD Cup 1999 / NSL-KDD (removed from CIC servers, deprecated)
# Download: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ CIC-IDS-2018/
# Features: CICFlowMeter-V3; model uses 5 numerical + 2 categorical

from concrete.ml.deployment import FHEModelClient, FHEModelServer
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
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier as RandomForestSklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
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


def load_and_preprocess(n_components_svd=100):
    """
    Loads CSE-CIC-IDS-2018 data, applies preprocessing and TruncatedSVD.
    Returns X_train_final, X_test_final, y_train_full, y_test_full, test_labels, preprocessor.
    """
    combined_csv_path = 'CIC-IDS-2018-Combined.csv'
    data_folder = 'CIC-IDS-2018'

    if os.path.exists(combined_csv_path):
        df = pd.read_csv(combined_csv_path)
    else:
        all_files = glob.glob(os.path.join(data_folder, "*.csv"))
        if not all_files:
            raise FileNotFoundError(
                f"No CSV files found in '{data_folder}'. "
                f"Download with: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ {data_folder}/"
            )
        df_list = [pd.read_csv(f, low_memory=False) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(combined_csv_path, index=False)

    df = clean_col_names(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df['binary_label'] = (df['label'] != 'benign').astype(int)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    numerical_features = ['syn_cnt', 'ack_cnt', 'fin_cnt', 'rst_cnt', 'tot_l_fw_pkt']
    categorical_features = ['protocol', 'dst_port']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    X_train_sparse = preprocessor.fit_transform(train_df)
    X_test_sparse = preprocessor.transform(test_df)

    with open('preprocessor_cic_ids_2018.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    y_train_full = train_df['binary_label']
    y_test_full = test_df['binary_label']
    # Keep label strings for result printing
    test_labels = test_df['label'].reset_index(drop=True)

    del train_df
    del df

    svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
    X_train_final = svd.fit_transform(X_train_sparse)
    X_test_final = svd.transform(X_test_sparse)

    del X_train_sparse
    del X_test_sparse

    return X_train_final, X_test_final, y_train_full, y_test_full, test_labels, preprocessor


def predict_single_record_plaintext(estimators, depth, n_components_svd=100):
    log_time()
    print(f"\n--- Plaintext Inference | estimators={estimators}, depth={depth} ---")

    X_train_final, X_test_final, y_train_full, y_test_full, test_labels, _ = \
        load_and_preprocess(n_components_svd)

    classifier = RandomForestSklearn(
        n_estimators=estimators,
        max_depth=depth,
        random_state=42
    )
    classifier.fit(X_train_final, y_train_full)

    inference_times = []

    for i in range(1000):
        single_record = X_test_final[i:i+1]

        start_time = time.time()
        output = classifier.predict(single_record)
        end_time = time.time()

        duration = end_time - start_time
        inference_times.append(duration)

        true_label_text = test_labels.iloc[i]
        true_label_binary = 0 if true_label_text == 'benign' else 1

#        print(f"Record {i+1}/1000 | Predicted: {output[0]} | True: {true_label_binary} | Time: {duration:.4f}s")

    log_time()
    print("\nCalculating final statistics...")

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
    Loads a pre-compiled FHE model, runs inference on individual records,
    and reports timing for both FHE execution and preprocessing.
    """
    log_time()
    print(f"\n--- FHE Inference | estimators={estimators}, depth={depth} ---")

    model_dir = f"./kdcup-models/fhe_model_{estimators}_estimators_{depth}_depth_svd_{n_components_svd}_components/"

    print("\n[STEP 1] Loading pre-compiled FHE circuit and preprocessor...")
    try:
        fhe_model_server = FHEModelServer(model_dir)
        fhe_model_server.load()
        fhe_model_client = FHEModelClient(model_dir)
        with open('preprocessor_cic_ids_2018.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please run model.py first to generate the FHE assets.")
        return

    print("\n[STEP 2] Preparing data on the CLIENT-SIDE before encryption...")

    _, X_test_final, _, _, test_labels, _ = load_and_preprocess(n_components_svd)

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

        preprocessing_duration = end_preprocessing - start_preprocessing
        preprocessing_times.append(preprocessing_duration)

        start_time = time.time()
        encrypted_output = fhe_model_server.run(encrypted_input, serialized_evaluation_keys)
        end_time = time.time()

        result = fhe_model_client.deserialize_decrypt_dequantize(encrypted_output)
        predicted_label = 1 if result[0][1] > 0.5 else 0

        duration = end_time - start_time
        inference_times.append(duration)

        true_label_text = test_labels.iloc[i]
        true_label_binary = 0 if true_label_text == 'benign' else 1

#        print(f"Record {i+1}/{records} | Predicted: {predicted_label} | True: {true_label_binary} | Time: {duration:.4f}s")

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
    predict_single_record_plaintext(2, 4)
    predict_single_record_plaintext(4, 2)
    predict_single_record_plaintext(4, 4)
    predict_single_record_plaintext(100, 2)

    predict_single_record_with_comparison(2, 4, 1000)
    predict_single_record_with_comparison(4, 2, 1000)
    predict_single_record_with_comparison(4, 4, 1000)
    predict_single_record_with_comparison(100, 2, 1000)
