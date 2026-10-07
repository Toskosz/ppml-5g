import pandas as pd

def analyze_dataset(file_path):
    try:
        # Define column names for the dataset
        column_names = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
            "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "level"
        ]
        
        # Read the dataset from the specified file path
        df = pd.read_csv(file_path, names=column_names)
        
        # Extract the unique values for the specified columns
        protocol_type_range = df["protocol_type"].unique()
        service_range = df["service"].unique()
        src_bytes_min = df["src_bytes"].min()
        src_bytes_max = df["src_bytes"].max()
        dst_bytes_min = df["dst_bytes"].min()
        dst_bytes_max = df["dst_bytes"].max()

        # Print the results
        print(f"Protocol Type Range: {protocol_type_range}")
        print(f"Service Range: {service_range}")
        print(f"Src Bytes Range: {src_bytes_min} - {src_bytes_max}")
        print(f"Dst Bytes Range: {dst_bytes_min} - {dst_bytes_max}")

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Specify the path to the dataset file
    dataset_path = "/Users/thiagotokarski/ppml-5g/Test.txt"
    analyze_dataset(dataset_path)