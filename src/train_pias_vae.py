import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

# Import the necessary training and synthesis functions from your advanced VAE module
from vae_time_series import train_vae_from_dataloader, synthesize_and_save, SEQUENCE_LENGTH, NUM_FEATURES, DEVICE

# --- Global Configuration Constants ---
SEQUENCE_LENGTH = 30  
NUM_FEATURES = 14     
BATCH_SIZE = 128
VAE_EPOCHS = 50 
KLD_WEIGHT = 0.001
NUM_SYNTHETIC_SAMPLES_PER_ANOMALY = 50 # Multiplier for synthesizing data

# --- Data Loading and Preprocessing Functions (Uses FD001 data) ---

def load_and_preprocess_data():
    """
    Loads CMAPSS data, selects features, normalizes, and calculates RUL/labels.
    """
    print("--- Phase 0: Data Loading and Preprocessing ---")
    
    # CRITICAL FIX: Corrected file name to 'train_FD001.txt' based on user clarification.
    train_data_path = 'CMAPSSData/train_FD001.txt' 
    
    # Defines all 26 columns present in the typical CMAPSS data files
    col_names = ['unit_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    
    try:
        # The FD001 data uses variable width/multiple spaces as separators
        df = pd.read_csv(train_data_path, sep=r'\s+', header=None, names=col_names, index_col=False)
        
    except FileNotFoundError:
        print(f"\nERROR: Data file '{train_data_path}' not found.")
        print("Please ensure the file is named 'train_FD001.txt' and located in the CMAPSSData subdirectory.")
        return None, None
    except KeyError as e:
        print(f"\nFATAL ERROR: A column access error occurred: {e}")
        print("This often happens if the raw file contains extra unnamed columns.")
        print("We are continuing by only using the clean 26 named columns.")


    # Based on CMAPSS analysis, these 13 sensors are most indicative of degradation
    sensor_cols_13 = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]]
    # We include 'cycle' (1) + 13 sensors = 14 features (matches NUM_FEATURES=14)
    selected_features = ['cycle'] + sensor_cols_13
    df_features = df[selected_features].copy()

    # Normalize features between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

    # Calculate RUL (Remaining Useful Life)
    max_cycles = df.groupby('unit_id')['cycle'].max()
    RUL_limit = 125 # Truncate RUL at 125 cycles
    
    def calculate_rul(df_unit):
        # Get the max cycle for this specific unit
        max_c = max_cycles[df_unit.iloc[0]['unit_id']]
        rul = max_c - df_unit['cycle']
        return np.minimum(rul, RUL_limit) # Apply the RUL limit

    df['RUL'] = df.groupby('unit_id', group_keys=False).apply(calculate_rul)
    # Define Anomaly Label: 1 if RUL < 15 cycles, 0 otherwise
    df['label'] = (df['RUL'] < 15).astype(int) 

    df_combined = pd.concat([df[['unit_id', 'label']].reset_index(drop=True), df_normalized], axis=1)

    return df_combined, scaler

def create_sequences(df_combined):
    """
    Creates sequences of length SEQUENCE_LENGTH for each engine unit.
    Returns 3D sequences (N, seq_len, num_features) and labels (N,).
    """
    sequences = []
    labels = []
    
    feature_cols = [col for col in df_combined.columns if col not in ['unit_id', 'label']]
    
    for unit_id in df_combined['unit_id'].unique():
        unit_df = df_combined[df_combined['unit_id'] == unit_id].copy().reset_index(drop=True)
        data = unit_df[feature_cols].values
        label_vector = unit_df['label'].values
        
        for i in range(len(data) - SEQUENCE_LENGTH + 1):
            seq = data[i : i + SEQUENCE_LENGTH]
            # The label of the sequence is determined by the label of the final cycle in the sequence
            label = label_vector[i + SEQUENCE_LENGTH - 1] 
            
            sequences.append(seq) # Keep as 3D (30, 14) 
            labels.append(label)
            
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

def get_data_for_vae_training():
    """Orchestrates loading, preprocessing, and splitting for VAE."""
    df_combined, scaler = load_and_preprocess_data()
    if df_combined is None:
        return None
        
    X_seq_3d, y = create_sequences(df_combined)
    
    # --- Data Splitting for PIAS-VAE ---
    # Phase 1 VAE Training: Only uses NORMAL data (label 0)
    normal_indices = (y == 0)
    X_normal_tensor = torch.from_numpy(X_seq_3d[normal_indices]).float()
    
    # Create DataLoader for VAE Training
    normal_dataset = TensorDataset(X_normal_tensor)
    normal_data_loader = DataLoader(normal_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"\nData ready. Total sequences: {X_seq_3d.shape[0]}")
    print(f"Normal sequences for VAE training: {X_normal_tensor.size(0)}")
    print(f"Anomaly sequences (real): {np.sum(y == 1)}")
    
    return normal_data_loader, X_all_seq_3d, y


# --- Main Execution for PIAS Pipeline ---

if __name__ == "__main__":
    
    # 0. Load and Prepare Data
    data_results = get_data_for_vae_training()
    
    if data_results is None:
        exit()
    
    normal_data_loader, X_all_seq_3d, y_all = data_results
    
    # --- 1. Train VAE (Phase 1) ---
    print("\n--- Phase 1: Training LSTM-VAE on Normal Sequences ---")
    # train_vae_from_dataloader is imported from vae_time_series.py
    vae_model, _ = train_vae_from_dataloader(
        dataloader=normal_data_loader,
        epochs=VAE_EPOCHS,
        lr=1e-3,
        kld_weight=KLD_WEIGHT,
        kl_anneal={'start': 0.0, 'end': KLD_WEIGHT, 'n_epochs': int(VAE_EPOCHS * 0.5)}, 
        model_savepath='models/vae_normal_manifold_weights.pth', # Path is relative to project root
        device=DEVICE
    )
    
    # --- 2. Augment Data (Phase 2) ---
    print("\n--- Phase 2: Generating Synthetic Anomalies via PIAS Interpolation ---")
    
    synthesize_and_save(
        model=vae_model,
        X_all=X_all_seq_3d,
        y_all=y_all,
        save_dir='data/03_processed', # Path is relative to project root
        n_per_anomaly=NUM_SYNTHETIC_SAMPLES_PER_ANOMALY,
        device=DEVICE
    )
    
    print("\nPIAS VAE training and data augmentation pipeline complete.")
    print("Next step: Run 'python3 src/train_classifier.py' to train the anomaly detector (Phase 3 & 4).")