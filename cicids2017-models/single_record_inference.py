from concrete.ml.deployment import FHEModelClient, FHEModelServer
from concrete.ml.sklearn.rf import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD # Import TruncatedSVD
from sklearn.model_selection import train_test_split
import datetime
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
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

def predict_single_record_plaintext(estimators, depth, svd):
    combined_csv_path = 'CIC-IDS-2017-Combined.csv'
    df = pd.read_csv(combined_csv_path)

    df = clean_col_names(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df['binary_label'] = (df['label'] != 'BENIGN').astype(int)

    classifier = RandomForestClassifier(
        n_estimators=estimators,
        max_depth=depth,
        random_state=42
    )

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    y_train_full = train_df['binary_label'] # Renamed to y_train_full for clarity

    classifier.fit(X_train_final, y_train_full)

    inference_times = []

    for i in range(1000):

        X_single_record_processed = X_test_final[i:i+1]

        start_time = time.time()

        result = classifier.predict(X_single_record_processed)

        end_time = time.time()

        predicted_label = result

        duration = end_time - start_time
        inference_times.append(duration)

        true_label_text = test_df.iloc[i]['label']
        true_label_binary = 1 if true_label_text != 'BENIGN' else 0

#        print(f"Record {i+1}/1000 | Predicted: {predicted_label} | True: {true_label_binary} | Time: {duration:.4f}s")

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



def predict_single_record_with_comparison(estimators, depth, svd):
    log_time()
    print("--- FHE Prediction with Before & After Comparison ---")

    print("\n[STEP 1] Loading pre-compiled FHE circuit and preprocessor...")
    try:
        fhe_model_server = FHEModelServer(f"./cicids2017-models/fhe_model_{estimators}_estimators_{depth}_depth_svd_{svd}_components/")
        fhe_model_server.load()
        fhe_model_client = FHEModelClient(f"./cicids2017-models/fhe_model_{estimators}_estimators_{depth}_depth_svd_{svd}_components/")
        
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please run the 'train.py' script first to generate the necessary files.")
        return

    print("\n[STEP 2] Preparing and inspecting data on the CLIENT-SIDE before encryption...")
    
    print("\n[STEP 3] Processing all records...")
    inference_times = []
    preprocessing_times = []

    serialized_evaluation_keys = fhe_model_client.get_serialized_evaluation_keys()

    for i in range(1000):

        X_single_record_processed = X_test_final[i:i+1]

        start_preprocessing_time = time.time()
        encrypted_input = fhe_model_client.quantize_encrypt_serialize(X_single_record_processed)
        end_preprocessing_time = time.time()
        preprocessing_duration = end_preprocessing_time - start_preprocessing_time
        preprocessing_times.append(preprocessing_duration)

        start_time = time.time()

        encrypted_output = fhe_model_server.run(encrypted_input, serialized_evaluation_keys)

        end_time = time.time()

        result = fhe_model_client.deserialize_decrypt_dequantize(encrypted_output)

        predicted_label = 1 if result[0][1] > 0.5 else 0

        duration = end_time - start_time
        inference_times.append(duration)

        true_label_text = test_df.iloc[i]['label']
        true_label_binary = 1 if true_label_text != 'BENIGN' else 0

#        print(f"Record {i+1}/1000 | Predicted: {predicted_label} | True: {true_label_binary} | Time: {duration:.4f}s")

    print("\n[STEP 4] Calculating final statistics...")
    log_time()

    total_records = len(inference_times)
    total_inference_time = sum(inference_times)
    mean_inference_time = np.mean(inference_times) if total_records > 0 else 0
    mean_preprocessing_time = np.mean(preprocessing_times) if total_records > 0 else 0

    print("\n" + "="*20 + " INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records}")
    print(f"   Total inference time: {total_inference_time:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time:.4f} seconds")
    print(f"Mean preprocessing time/record: {mean_preprocessing_time:.4f} seconds")

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

    del train_df
    del df

    models_to_test = [
        {'estimators': 2, 'depth': 2, 'svd': 200},
        {'estimators': 4, 'depth': 2, 'svd': 200},
        {'estimators': 4, 'depth': 4, 'svd': 200},
    ]

    svd_cache = {}

    for model_params in models_to_test:
        estimators = model_params['estimators']
        depth = model_params['depth']
        n_components_svd = model_params['svd']

        print(f"\n{'='*20} TESTING MODEL: estimators={estimators}, depth={depth}, svd={n_components_svd} {'='*20}")

        if n_components_svd not in svd_cache:
            print(f"\nApplying TruncatedSVD to reduce feature count from {X_train_sparse.shape[1]} to {n_components_svd}...")
            svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
            X_train_dense_reduced = svd.fit_transform(X_train_sparse)
            X_test_dense_reduced = svd.transform(X_test_sparse)
            svd_cache[n_components_svd] = (X_train_dense_reduced, X_test_dense_reduced)
            print(f"Shape after SVD reduction (dense) - X_train: {X_train_dense_reduced.shape}, X_test: {X_test_dense_reduced.shape}")
        else:
            print(f"\nUsing cached SVD transformation for {n_components_svd} components...")
            X_train_dense_reduced, X_test_dense_reduced = svd_cache[n_components_svd]

        X_train_final = X_train_dense_reduced
        
        X_test_final = X_test_dense_reduced

        print(f"Final training data shape: {X_train_final.shape}")
        print(f"Final testing data shape: {X_test_final.shape}")

        predict_single_record_plaintext(estimators, depth, n_components_svd)
        predict_single_record_with_comparison(estimators, depth, n_components_svd)

    del X_train_sparse
    del X_test_sparse
    del svd_cache

