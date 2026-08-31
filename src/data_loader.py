import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import requests
import zipfile
import io

warnings.filterwarnings('ignore')

def download_bank_marketing_data(download_dir='data/raw'):
    """Download the Bank Marketing dataset from UCI if it doesn't exist."""
    os.makedirs(download_dir, exist_ok=True)
    target_file = os.path.join(download_dir, 'bank-additional-full.csv')
    
    if os.path.exists(target_file):
        print(f"Dataset already exists at: {target_file}")
        return target_file
    
    print("Dataset not found. Downloading from UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        print("Download successful. Extracting...")
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = 'bank-additional/bank-additional-full.csv'
            with z.open(csv_name) as source, open(target_file, 'wb') as target:
                target.write(source.read())
        print(f"Extracted to: {target_file}")
        return target_file
    except Exception as e:
        print(f"Error downloading: {e}")
        raise

def load_and_process_data(
    raw_data_path=None,
    processed_data_dir='data/processed',
    test_size=0.2,
    random_state=42,
    max_rows=9000
):
    if raw_data_path is None:
        raw_data_path = download_bank_marketing_data()
    
    print("=" * 60)
    print("STEP 1: Loading and Processing Bank Marketing Dataset")
    print("=" * 60)
    
    print(f"Loading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path, sep=';')
    print(f"Original dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    target_counts = df['y'].value_counts()
    print(f"Target distribution:\n  No (0): {target_counts.get('no', 0)}\n  Yes (1): {target_counts.get('yes', 0)}")
    
    # Drop 'duration' to prevent data leakage
    if 'duration' in df.columns:
        print("Dropping 'duration' column to prevent data leakage...")
        df = df.drop(columns=['duration'])
    
    X = df.drop(columns=['y'])
    y = df['y'].map({'yes': 1, 'no': 0})
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {categorical_cols}")
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # --- FIXED SUBSAMPLING ---
    if len(X) > max_rows:
        print(f"Subsampling from {len(X)} rows to {max_rows} rows (stratified)...")
        # Use train_test_split with train_size to get a stratified subsample
        X, _, y, _ = train_test_split(
            X, y,
            train_size=max_rows,
            random_state=random_state,
            stratify=y
        )
        print(f"Subsampled dataset shape: {len(X)} rows")
    
    print(f"Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {len(X_train)} rows")
    print(f"Test set: {len(X_test)} rows")
    
    os.makedirs(processed_data_dir, exist_ok=True)
    np.save(f'{processed_data_dir}/X_train.npy', X_train.values)
    np.save(f'{processed_data_dir}/X_test.npy', X_test.values)
    np.save(f'{processed_data_dir}/y_train.npy', y_train.values)
    np.save(f'{processed_data_dir}/y_test.npy', y_test.values)
    
    with open(f'{processed_data_dir}/feature_names.txt', 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    
    print(f"Saved processed data to: {processed_data_dir}")
    print("=" * 60)
    return X_train.values, X_test.values, y_train.values, y_test.values, X.columns.tolist()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_and_process_data()
    print("\n✅ Data loading complete!")
    print(f"Features: {feature_names}")
