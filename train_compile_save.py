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
    fhe_classifier = ConcreteRandomForestClassifier(n_estimators=100)
        print("Training Concrete-ML RandomForestClassifier with n_estimators=100...")
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
    dev = FHEModelDev("./fhe_model/", fhe_classifier)
    dev.save()

    # Save the preprocessor to be used for new data
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print("\nTraining and compilation finished. You can now use 'predict_fhe.py' for fast predictions.")
    print("-" * 50 + "\n")

train_and_save_model()
