from concrete.ml.deployment import FHEModelClient, FHEModelServer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD # Import TruncatedSVD
from sklearn.model_selection import train_test_split
import datetime
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from zoneinfo import ZoneInfo
import numpy as np
import time
import joblib

def log_time():
    """Prints the current time in Brasília timezone."""
    brasilia_tz = ZoneInfo("America/Sao_Paulo")
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    brasilia_now = utc_now.astimezone(brasilia_tz)
    formatted_time = brasilia_now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"[LOG] Current time: {formatted_time}")


def measure_inference_time(estimators, depth, svd):
    log_time()
    print("--- FHE vs Plaintext Inference Time Measurement ---")

    print("\n[STEP 1] Loading FHE model...")
    try:
        fhe_model_server = FHEModelServer(f"./netflow-models/fhe_model_{estimators}_estimators_{depth}_depth_svd_{svd}_components/")
        fhe_model_server.load()
        fhe_model_client = FHEModelClient(f"./netflow-models/fhe_model_{estimators}_estimators_{depth}_depth_svd_{svd}_components/")
        
    except FileNotFoundError as e:
        print(f"Error loading FHE model files: {e}")
        print("Please run the training script first to generate the necessary files.")
        return

    print("\n[STEP 2] Loading plaintext model...")
    try:
        # NOTE: This script assumes the plaintext model is saved in a specific path.
        # You may need to adjust this path depending on where your training script saves the models.
        plaintext_model_path = f"./netflow-models/plaintext_model_{estimators}_estimators_{depth}_depth_svd_{svd}_components.joblib"
        plaintext_model = joblib.load(plaintext_model_path)
        print(f"Successfully loaded plaintext model from {plaintext_model_path}")
    except FileNotFoundError as e:
        print(f"Error loading plaintext model file: {e}")
        print(f"Attempted to load from: {plaintext_model_path}")
        print("Please ensure the plaintext model is saved in the expected location and format.")
        return

    print("\n[STEP 3] Processing all records...")
    fhe_inference_times = []
    plaintext_inference_times = []

    serialized_evaluation_keys = fhe_model_client.get_serialized_evaluation_keys()

    for i in range(1000):

        X_single_record_processed = X_test_final[i:i+1]
        true_label_binary = test_df.iloc[i]['label']

        # --- FHE Prediction ---
        encrypted_input = fhe_model_client.quantize_encrypt_serialize(X_single_record_processed)

        start_time_fhe = time.time()

        encrypted_output = fhe_model_server.run(encrypted_input, serialized_evaluation_keys)

        end_time_fhe = time.time()

        result_fhe = fhe_model_client.deserialize_decrypt_dequantize(encrypted_output)

        predicted_label_fhe = 1 if result_fhe[0][1] > 0.5 else 0

        duration_fhe = end_time_fhe - start_time_fhe
        fhe_inference_times.append(duration_fhe)

        # --- Plaintext Prediction ---
        start_time_plain = time.time()
        predicted_label_plain = plaintext_model.predict(X_single_record_processed)[0]
        end_time_plain = time.time()
        duration_plain = end_time_plain - start_time_plain
        plaintext_inference_times.append(duration_plain)

        print(f"Record {i+1}/1000 | True: {true_label_binary} | FHE Pred: {predicted_label_fhe} ({duration_fhe:.4f}s) | Plain Pred: {predicted_label_plain} ({duration_plain:.6f}s)")

    print("\n[STEP 4] Calculating final statistics...")
    log_time()

    # FHE stats
    total_records = len(fhe_inference_times)
    total_inference_time_fhe = sum(fhe_inference_times)
    mean_inference_time_fhe = np.mean(fhe_inference_times) if total_records > 0 else 0

    print("\n" + "="*20 + " FHE INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records}")
    print(f"   Total inference time: {total_inference_time_fhe:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time_fhe:.4f} seconds")
    print("="*61 + "\n")

    # Plaintext stats
    total_records_plain = len(plaintext_inference_times)
    total_inference_time_plain = sum(plaintext_inference_times)
    mean_inference_time_plain = np.mean(plaintext_inference_times) if total_records_plain > 0 else 0

    print("\n" + "="*20 + " PLAINTEXT INFERENCE SUMMARY " + "="*20)
    print(f"Total records processed: {total_records_plain}")
    print(f"   Total inference time: {total_inference_time_plain:.4f} seconds")
    print(f"Mean inference time/record: {mean_inference_time_plain:.6f} seconds")
    print("="*65 + "\n")

def clean_col_names(df):
    """Cleans column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.strip().replace(' ', '_').replace('/', '_').lower() for col in cols]
    df.columns = new_cols
    return df

if __name__ == "__main__":

    combined_csv_path = 'NF-UNSW-NB15-v3.csv'
    df = pd.read_csv(combined_csv_path)

    df = clean_col_names(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df['binary_label'] = df['label']

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Data split into {len(train_df)} training samples and {len(test_df)} testing samples.")

    numerical_features_selected = [
        'in_bytes',
        'out_bytes',
    ]
    categorical_features_selected = ['protocol', 'l7_proto']

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
        {'estimators': 2, 'depth': 2, 'svd': 100},
        {'estimators': 4, 'depth': 2, 'svd': 100},
        {'estimators': 2, 'depth': 4, 'svd': 100},
        {'estimators': 4, 'depth': 4, 'svd': 100},
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

        measure_inference_time(estimators, depth, n_components_svd)

    del X_train_sparse
    del X_test_sparse
    del svd_cache
