# Preparation to match Scapy.

import pandas as pd
import pickle
import sklearn
from concrete.ml.deployment import FHEModelDev
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
from concrete.ml.sklearn.rf import RandomForestClassifier
from sklearn.metrics import accuracy_score
from zoneinfo import ZoneInfo
import datetime

def log_time(): 

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

train_df = pd.read_csv('Train.txt', delimiter=',', header=None, names=column_names)
test_df = pd.read_csv('Test.txt', delimiter=',', header=None, names=column_names)

# Function to convert dataset flags to Scapy-compatible flags
def convert_flag_to_scapy(flag):
    flag_translation = {
        'REJ': 'R',
        'SF': 'PA',
        'S0': 'S',
        'RSTO': 'R',
        'S1': 'S',
        'S2': 'S',
        'S3': 'S',
        'RSTOS0': 'R',
        'OTH': ''
    }
    return ''.join([flag_translation.get(f, '') for f in flag])

# Apply the flag conversion to the 'flag' column
train_df['flag'] = train_df['flag'].apply(convert_flag_to_scapy)
test_df['flag'] = test_df['flag'].apply(convert_flag_to_scapy)

# Convert the multi-class labels into binary labels: 1 for any attack, 0 for normal
train_df['binary_label'] = (train_df['label'] != 'normal').astype(int)
test_df['binary_label'] = (test_df['label'] != 'normal').astype(int)

# Selecting only the necessary features
features_to_use = ['protocol_type', 'service', 'src_bytes', 'dst_bytes']
categorical_features = ['protocol_type', 'service']
numerical_features = ['src_bytes', 'dst_bytes']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'  # This drops the columns that are not explicitly transformed
)

# Apply preprocessing to both training and testing data
X_train = preprocessor.fit_transform(train_df[features_to_use])
X_test = preprocessor.transform(test_df[features_to_use])

# Save the preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Labels for training and testing
y_train = train_df['binary_label']
y_test = test_df['binary_label']

n_estimators_list = [2, 5, 10, 25, 50, 100] # List of n_estimators values to test

for n_estimators in n_estimators_list:
    print("\n" + "="*60)
    print(f"STARTING TEST FOR n_estimators = {n_estimators}")
    print("="*60)
    
    log_time()
    print(f"Training RandomForestClassifier with {n_estimators} estimators...")

    # Create and train a RandomForest classifier
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=2)
    classifier.fit(X_train.toarray(), y_train)

    # Evaluate clear text
    log_time()
    print(f"Start clear with {n_estimators} estimators...")
    y_pred = classifier.predict(X_test.toarray())
    log_time()
    print(f"Clear Accuracy with {n_estimators} estimators: {accuracy_score(y_test, y_pred)}")

    # Compile FHE
    log_time()
    print(f"Compiling FHE model with {n_estimators} estimators...")
    classifier.compile(X_train.toarray())
    log_time()
    print("Finished compiling.")

    log_time()
    print(f"Making FHE prediction with {n_estimators} estimators")
    y_pred_fhe = classifier.predict(X_test.toarray(), fhe="simulate")
    log_time()
    print(f"Finished prediction with {n_estimators}")

    log_time()
    # Evaluate the FHE classifier
    print(f"FHE Accuracy with {n_estimators} estimators: {accuracy_score(y_test, y_pred_fhe)}")

    # --- Save Artifacts ---
    print("Saving compiled FHE circuit and preprocessor to disk...")
    # The compiled model (FHE circuit) is saved.
    dev = FHEModelDev(f"./fhe_model_{n_estimators}_estimators/", classifier)
    dev.save()
