from concrete.ml.deployment import FHEModelClient, FHEModelServer
from sklearn.decomposition import TruncatedSVD # Import TruncatedSVD
from sklearn.model_selection import train_test_split
import datetime
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from zoneinfo import ZoneInfo
import numpy as np
import time

def log_time():
    """Prints the current time in Brasília timezone."""
    brasilia_tz = ZoneInfo("America/Sao_Paulo")
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    brasilia_now = utc_now.astimezone(brasilia_tz)
    formatted_time = brasilia_now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"[LOG] Current time: {formatted_time}")

def predict_single_record_plaintext(estimators, depth):

    classifier = RandomForestClassifier(
        n_estimators=estimators,
        max_depth=depth,
        random_state=42
    )
    classifier.fit(X_train_final, y_train_final)

    inference_times = []

    for i in range(1000):
        single_record_df = X_test_final[i:i+1].toarray()

        start_time = time.time()

        output = classifier.predict(single_record_df)

        end_time = time.time()

        duration = end_time - start_time
        inference_times.append(duration)

        true_label_text = test_df[i]['label']
        true_label_binary = 1 if true_label_text != 'normal' else 0

        print(f"Record {i+1}/{X_test_final.shape[0]} | Predicted: {output[0]} | True: {true_label_binary} | Time: {duration:.4f}s")

    log_time()
    print("\nCalculating final statistics...")

    total_records = len(inference_times)
    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if total_records > 0 else 0

    print("\n" + "="*20 + " INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records}")
    print(f"   Total inference time: {total_inference_time:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time:.4f} seconds")
    print("="*61 + "\n")

def predict_single_record_with_comparison(estimators, depth, svd):
    log_time()
    print("--- FHE Prediction with Before & After Comparison ---")

    print("\n[STEP 1] Loading pre-compiled FHE circuit and preprocessor...")
    try:
        fhe_model_server = FHEModelServer(f"./cicids2017-models/fhe_model_{estimators}_estimators_{depth}_depth_svd_{svd}_components/")
        fhe_model_server.load()
        fhe_model_client = FHEModelClient(f"./cicids2017-models/fhe_model_{estimators}_estimators_{depth}_depth_svd_{svd}_components/")
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please run the 'train.py' script first to generate the necessary files.")
        return

    print("\n[STEP 2] Preparing and inspecting data on the CLIENT-SIDE before encryption...")
    
    print("\n[STEP 3] Processing all records...")
    inference_times = []

    serialized_evaluation_keys = fhe_model_client.get_serialized_evaluation_keys()

    for i in range(1000):
        single_record_df = X_test_final.iloc[[i]]
        
        X_single_record_processed = preprocessor.transform(single_record_df).toarray()

        encrypted_input = fhe_model_client.quantize_encrypt_serialize(X_single_record_processed)

        start_time = time.time()

        encrypted_output = fhe_model_server.run(encrypted_input, serialized_evaluation_keys)

        end_time = time.time()

        result = fhe_model_client.deserialize_decrypt_dequantize(encrypted_output)

        predicted_label = 1 if result[0][1] > 0.5 else 0

        duration = end_time - start_time
        inference_times.append(duration)

        true_label_text = test_df[i]['label']
        true_label_binary = 1 if true_label_text != 'normal' else 0

        print(f"Record {i+1}/{len(X_test_final) * 0.1} | Predicted: {predicted_label} | True: {true_label_binary} | Time: {duration:.4f}s")

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

def clean_col_names(df):
    cols = df.columns
    new_cols = [col.strip().replace(' ', '_').lower() for col in cols]
    df.columns = new_cols
    return df

if __name__ == "__main__":

    combined_csv_path = 'CIC-IDS-2017-Combined.csv'
    df = pd.read_csv(combined_csv_path)

    df = clean_col_names(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df['binary_label'] = (df['label'] != 'BENIGN').astype(int)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Data split into {len(train_df)} training samples and {len(test_df)} testing samples.")

    numerical_features_selected = ['syn_flag_count','ack_flag_count', 'fin_flag_count', 'rst_flag_count', 'total_length_of_fwd_packets', 'total_length_of_bwd_packets']
    categorical_features_selected = ['destination_port']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features_selected),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_selected)
        ],
        remainder='drop'
    )

    print("Applying preprocessing (MinMaxScaler and OneHotEncoder)...")
    X_train_sparse = preprocessor.fit_transform(train_df)
    X_test_sparse = preprocessor.transform(test_df)
    print(f"Shape after initial preprocessing (sparse) - X_train: {X_train_sparse.shape}, X_test: {X_test_sparse.shape}")

    y_train_full = train_df['binary_label']
    y_test_full = test_df['binary_label']

    del train_df
    del df

    n_components_svd = 200
    print(f"\nApplying TruncatedSVD to reduce feature count from {X_train_sparse.shape[1]} to {n_components_svd}...")

    svd = TruncatedSVD(n_components=n_components_svd, random_state=42)

    X_train_dense_reduced = svd.fit_transform(X_train_sparse)
    X_test_dense_reduced = svd.transform(X_test_sparse)

    print(f"Shape after SVD reduction (dense) - X_train: {X_train_dense_reduced.shape}, X_test: {X_test_dense_reduced.shape}")

    del X_train_sparse
    del X_test_sparse

    X_train_final = X_train_dense_reduced
    y_train_final = y_train_full

    X_test_final = X_test_dense_reduced

    print(f"Final training data shape: {X_train_final.shape}")
    print(f"Final testing data shape: {X_test_final.shape}")

    if X_train_final is not X_train_dense_reduced:
        del X_train_dense_reduced
    if X_test_final is not X_test_dense_reduced:
        del X_test_dense_reduced

    predict_single_record_plaintext(2, 2)
    predict_single_record_plaintext(100, None)
    predict_single_record_plaintext(100, 4)

    predict_single_record_with_comparison(2, 2, 200)
    predict_single_record_with_comparison(2, None, 200)
    predict_single_record_with_comparison(100, 4, 200)
