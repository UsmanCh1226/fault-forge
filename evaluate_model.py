import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings

# Ignore pandas future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration Constants (Must match training constants) ---
SEQUENCE_LENGTH = 30
NUM_FEATURES = 13 # Match VAE script (Cycle + 12 sensors)
INPUT_DIM = SEQUENCE_LENGTH * NUM_FEATURES
BATCH_SIZE = 128
MODEL_PATH = 'models/classifier_weights.pth'

# --- 1. Classifier Model Definition (Must match train_classifier.py) ---
class HealthClassifier(nn.Module):
    def __init__(self, input_dim):
        super(HealthClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

# --- 2. Data Preparation Functions (Copied from previous scripts) ---

def load_and_preprocess_data():
    """ Loads CMAPSS data and prepares features and RUL labels. """
    data_dir = 'data/01_raw'
    train_data_path = os.path.join(data_dir, 'train_FD001.txt')
    
    # Selected features based on analysis (cycle + 12 sensors)
    sensor_cols = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]]
    
    if not os.path.exists(train_data_path):
        print("MOCK DATA: Using placeholder data.")
        # Create mock data structure resembling the real data
        mock_data = np.random.rand(5000, 26) 
        mock_data[:, 0] = np.tile(np.arange(20), 250)[:5000] # Mock Unit IDs
        cols = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
        df = pd.DataFrame(mock_data, columns=cols)
        df['unit_id'] = (df.index // 20) + 1 # Assign unit IDs 1 to 250
    else:
        col_names = ['unit_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
        df = pd.read_csv(train_data_path, sep=r'\s+', header=None, names=col_names, index_col=False)

    selected_features = ['cycle'] + sensor_cols 
    df_features = df[selected_features].copy()

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

    # Calculate RUL and Classification Label
    max_cycles = df.groupby('unit_id')['cycle'].max()
    RUL_limit = 125 # Max RUL is capped at 125 (as is standard for this dataset)
    
    def calculate_rul(df_unit):
        max_c = max_cycles[df_unit.iloc[0]['unit_id']]
        rul = max_c - df_unit['cycle']
        return np.minimum(rul, RUL_limit)

    df['RUL'] = df.groupby('unit_id', group_keys=False).apply(calculate_rul)
    # Binary classification label: 1 if RUL < 15, 0 otherwise (approaching failure)
    df['label'] = (df['RUL'] < 15).astype(int)

    df_combined = pd.concat([df[['unit_id', 'label']].reset_index(drop=True), df_normalized], axis=1)
    return df_combined

def create_sequences(df):
    """ Creates sequences of length SEQUENCE_LENGTH for each engine unit. """
    sequences = []
    labels = []
    feature_cols = [col for col in df.columns if col not in ['unit_id', 'label']]
    
    for unit_id in df['unit_id'].unique():
        unit_df = df[df['unit_id'] == unit_id].copy().reset_index(drop=True)
        data = unit_df[feature_cols].values
        label_vector = unit_df['label'].values
        
        # Create sliding windows
        for i in range(len(data) - SEQUENCE_LENGTH):
            # Sequence: flattened 1D array of features for the window
            sequences.append(data[i : i + SEQUENCE_LENGTH].flatten())
            # Label: the label (0 or 1) at the *end* of the sequence window
            labels.append(label_vector[i + SEQUENCE_LENGTH - 1])
            
    X_tensor = torch.from_numpy(np.array(sequences)).float()
    y_tensor = torch.from_numpy(np.array(labels)).long()
    
    return X_tensor, y_tensor


def evaluate(model, data_loader, device):
    """ Runs the model against the data loader and collects predictions. """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Evaluating Model"):
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    print("Loading and preparing full dataset for evaluation...")
    X_full, y_full = create_sequences(load_and_preprocess_data())
    
    # Use the full dataset for evaluation (as a proxy for a test set)
    full_dataset = TensorDataset(X_full, y_full)
    full_data_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model weights not found at {MODEL_PATH}.")
        print("Please ensure train_classifier.py ran successfully and created 'models/classifier_weights.pth'.")
        return

    print(f"Loading trained classifier from {MODEL_PATH}...")
    model = HealthClassifier(INPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # 3. Evaluate Performance
    y_true, y_pred = evaluate(model, full_data_loader, device)

    # 4. Report Results
    total_samples = len(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("\n--- Model Evaluation Results (Full Dataset) ---")
    print(f"Total Samples Evaluated: {total_samples}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Report Confusion Matrix
    # [[TN, FP],
    #  [FN, TP]]
    print("\nConfusion Matrix:")
    # Use the tabulate package format for a cleaner output if available, otherwise print raw matrix
    try:
        from tabulate import tabulate
        table = [
            ["Predicted Normal (0)", conf_matrix[0, 0], conf_matrix[0, 1]],
            ["Predicted Anomaly (1)", conf_matrix[1, 0], conf_matrix[1, 1]]
        ]
        print(tabulate(table, headers=["Actual", "Normal (0)", "Anomaly (1)"]))
    except ImportError:
        print(conf_matrix)

    tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0) # Handle potential size issues with mock data

    # Calculate Recall and Precision for Anomaly class (1)
    # Recall (Sensitivity): How many actual anomalies were correctly identified? TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Precision: When the model predicts an anomaly, how often is it correct? TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print("\nDetailed Metrics:")
    print(f"True Negatives (TN - Correctly Normal): {tn}")
    print(f"False Positives (FP - Misclassified as Anomaly): {fp} (Type I Error)")
    print(f"False Negatives (FN - Misclassified as Normal): {fn} (Type II Error - **Worst for RUL!**)")
    print(f"True Positives (TP - Correctly Anomaly): {tp}")

    print(f"\nAnomaly Recall (Ability to find true anomalies): {recall:.4f}")
    print(f"Anomaly Precision (Confidence in anomaly prediction): {precision:.4f}")

if __name__ == '__main__':
    main()
    