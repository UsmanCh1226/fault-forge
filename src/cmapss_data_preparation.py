import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- Global Configuration Constants ---
SEQUENCE_LENGTH = 30  # Look-back window size
NUM_FEATURES = 14     # Number of selected sensor features
INPUT_DIM = SEQUENCE_LENGTH * NUM_FEATURES # Flattened sequence length for VAE
BATCH_SIZE = 128
LATENT_DIM = 20       # Used only for reference, but defined in vae_time_series.py

def load_and_preprocess_data():
    """
    Loads CMAPSS data, selects features, normalizes, and calculates RUL (Remaining Useful Life).
    NOTE: Uses mock data if 'train_FD001.txt' is not found.
    """
    data_dir = 'data/01_raw'
    train_data_path = os.path.join(data_dir, 'train_FD001.txt')

    if not os.path.exists(train_data_path):
        print("MOCK DATA: Generating 5000 samples with balanced label distribution.")

        # --- MOCK DATA BLOCK (Ensures both label=0 and label=1 samples) ---
        num_rows = 5000
        mock_data = np.random.rand(num_rows, 26)
        mock_data[:, 0] = np.tile(np.arange(20), 250)[:num_rows]  # unit_id
        cols = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
        df = pd.DataFrame(mock_data, columns=cols)

        # Half normal (RUL=30), half anomaly (RUL=10)
        df['RUL'] = np.where(np.arange(num_rows) < num_rows // 2, 30, 10)
        df['label'] = (df['RUL'] < 15).astype(int)

        print("Label distribution in mock data:")
        print(df['label'].value_counts())
    else:
        # Actual CMAPSS loading (adjust column names based on your file)
        col_names = ['unit_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
        df = pd.read_csv(train_data_path, sep=r'\s+', header=None, names=col_names, index_col=False)

        # --- Feature Selection ---
        # Select sensor features commonly used for RUL prediction
        sensor_cols = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]]
        selected_features = ['cycle'] + sensor_cols

        df_features = df[selected_features].copy()

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

        max_cycles = df.groupby('unit_id')['cycle'].max()
        RUL_limit = 125

        def calculate_rul(df_unit):
            max_c = max_cycles[df_unit.iloc[0]['unit_id']]
            rul = max_c - df_unit['cycle']
            return np.minimum(rul, RUL_limit)

        df['RUL'] = df.groupby('unit_id', group_keys=False).apply(calculate_rul)
        df['label'] = (df['RUL'] < 15).astype(int)

        df_combined = pd.concat([df[['unit_id', 'label']].reset_index(drop=True), df_normalized], axis=1)
        return df_combined

    # --- Feature Selection for mock data (use 14 total) ---
    sensor_cols = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]]
    selected_features = ['cycle'] + sensor_cols
    df_features = df[selected_features].copy()

    # --- Normalization ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

    # Combine normalized features with unit_id and label
    df_combined = pd.concat([df[['unit_id', 'label']].reset_index(drop=True), df_normalized], axis=1)
    return df_combined

def create_sequences(df):
    """
    Creates sequences of length SEQUENCE_LENGTH for each engine unit.
    """
    sequences = []
    labels = []

    # Get the normalized features (excluding unit_id and label)
    feature_cols = [col for col in df.columns if col not in ['unit_id', 'label']]
    
    for unit_id in df['unit_id'].unique():
        unit_df = df[df['unit_id'] == unit_id].copy().reset_index(drop=True)
        data = unit_df[feature_cols].values
        label_vector = unit_df['label'].values
        
        # Create sliding window sequences
        for i in range(len(data) - SEQUENCE_LENGTH):
            seq = data[i : i + SEQUENCE_LENGTH]
            label = label_vector[i + SEQUENCE_LENGTH - 1]
            sequences.append(seq.flatten())
            labels.append(label)
            
    return np.array(sequences), np.array(labels)

def get_data_loaders():
    """
    Loads preprocessed data, generates sequences, splits, and returns everything for downstream steps.
    """
    df_combined = load_and_preprocess_data()
    X_flat, y = create_sequences(df_combined)
    X_tensor = torch.from_numpy(X_flat).float()
    y_tensor = torch.from_numpy(y).long()

    # Phase 1 VAE Training: Only uses NORMAL data (label 0)
    normal_indices = (y == 0)
    X_normal = X_tensor[normal_indices]
    y_normal = y_tensor[normal_indices]
    normal_dataset = TensorDataset(X_normal, y_normal)
    normal_data_loader = DataLoader(normal_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Phase 2 Synthesis: Needs the ANOMALY data (label 1)
    anomaly_indices = (y == 1)
    anomaly_data_tensor = X_tensor[anomaly_indices]

    # Phase 3 Classifier Training: Needs ALL data (Normal + Anomaly)
    full_dataset = TensorDataset(X_tensor, y_tensor)

    sequence_3d_shape = (SEQUENCE_LENGTH, X_flat.shape[1] // SEQUENCE_LENGTH)

    print(f"Data ready. Total sequences: {X_flat.shape[0]}")
    print(f"Normal sequences for VAE training: {X_normal.size(0)}")
    print(f"Anomaly sequences for synthesis: {anomaly_data_tensor.size(0)}")
    print(f"Flattened Input Dimension (INPUT_DIM): {INPUT_DIM}")

    return normal_data_loader, anomaly_data_tensor, sequence_3d_shape, full_dataset

if __name__ == "__main__":
    normal_data_loader, anomaly_data_tensor, sequence_3d_shape, full_dataset = get_data_loaders()
    X_all = full_dataset.tensors[0].numpy()
    y_all = full_dataset.tensors[1].numpy()

    # Save to proper location
    os.makedirs('data/03_processed', exist_ok=True)
    np.save('data/03_processed/X_train_combined.npy', X_all)
    np.save('data/03_processed/y_train_combined.npy', y_all)
    print("Saved combined training data to 'data/03_processed/X_train_combined.npy' and 'y_train_combined.npy'.")