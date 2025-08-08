import pandas as pd

# Define file paths
input_file = 'NF-UQ-NIDS-v2.csv'
output_file = 'sample-NF-UQ-NIDS-v2.csv'

# Number of rows to read
n_rows = 1000

print(f'Reading the first {n_rows} rows from {input_file}...')

# Read the first n_rows from the large CSV
df_sample = pd.read_csv(input_file, nrows=n_rows)

# Save the sample to a new CSV file
df_sample.to_csv(output_file, index=False)

print(f'Successfully created sample file: {output_file}')
