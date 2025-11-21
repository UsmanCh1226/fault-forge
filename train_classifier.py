# train_classifier.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import sequence constants from your VAE module for compatibility
from src.vae_time_series import SEQUENCE_LENGTH, NUM_FEATURES

# --- Config ---
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
MODEL_PATH = 'models/classifier_weights.pth'
CONF_MATRIX_PATH = 'reports/confusion_matrix_best.png'
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LSTM Classifier ---
class LSTMClassifier(nn.Module):
    def __init__(self, num_features: int = NUM_FEATURES, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # binary classification -> 2 logits
        )

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]             # (batch, hidden_size)
        logits = self.fc(last_hidden)     # (batch, 2)
        return logits

# --- Utilities ---
def load_combined_data(x_path='data/03_processed/X_train_combined.npy', y_path='data/03_processed/y_train_combined.npy'):
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Combined dataset not found. Run the VAE synthesis step first.")
    X = np.load(x_path)
    y = np.load(y_path)
    print(f'Loaded combined data X:{X.shape} y:{y.shape}')

    # If data is flattened (N, seq_len * num_features) -> reshape to (N, seq_len, num_features)
    if X.ndim == 2 and X.shape[1] == SEQUENCE_LENGTH * NUM_FEATURES:
        X = X.reshape(-1, SEQUENCE_LENGTH, NUM_FEATURES)
        print(f'Reshaped flattened X to {X.shape}')
    elif X.ndim == 3:
        # ensure shape matches expected
        if X.shape[1] != SEQUENCE_LENGTH or X.shape[2] != NUM_FEATURES:
            raise ValueError(f"Unexpected sequence shape {X.shape}. Expect (N, {SEQUENCE_LENGTH}, {NUM_FEATURES})")
    else:
        raise ValueError("Unsupported X shape. Expected 2D flattened or 3D sequences.")
    return X, y

def create_dataloaders(X: np.ndarray, y: np.ndarray, test_size=0.2):
    # Stratified split to keep class balance across train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train/Val split: {X_train.shape[0]} / {X_val.shape[0]} sequences")
    return train_loader, val_loader

def compute_class_weights(y):
    # Return weight tensor for CrossEntropyLoss (higher weight for minority)
    classes, counts = np.unique(y, return_counts=True)
    # inverse frequency
    inv_freq = 1.0 / counts
    norm = inv_freq / np.sum(inv_freq) * len(classes)
    weights = np.zeros(len(classes), dtype=np.float32)
    for cls, w in zip(classes, norm):
        weights[cls] = w
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    # Focus on anomaly class = 1
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return p, r, f1, cm, y_true, y_pred

def plot_and_save_confusion_matrix(cm, classes=['Normal','Anomaly'], outpath=CONF_MATRIX_PATH):
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Best Model)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved confusion matrix to {outpath}")

# --- Training loop ---
def train():
    X, y = load_combined_data()
    train_loader, val_loader = create_dataloaders(X, y, test_size=0.2)

    # model init
    model = LSTMClassifier(num_features=NUM_FEATURES).to(DEVICE)

    # class weights
    class_weights = compute_class_weights(y)
    print(f"Class weights: {class_weights.cpu().numpy()} (used for CrossEntropyLoss)")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)

    best_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluate on validation set
        p, r, f1, cm, y_true, y_pred = evaluate_model(model, val_loader)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {epoch_loss:.6f} | Anomaly P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")

        # Scheduler step based on validation F1 (we want to maximize it)
        scheduler.step(f1)

        # Save best model by anomaly F1
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"*** New best model saved (Epoch {epoch}, Anomaly F1={f1:.4f}) ***")
            # save confusion matrix for inspection
            plot_and_save_confusion_matrix(cm, outpath=CONF_MATRIX_PATH)

    print(f"\nTraining complete. Best Epoch: {best_epoch} with Anomaly F1: {best_f1:.4f}")
    print(f"Final model saved to: {MODEL_PATH}")
    print(f"Confusion matrix for best model saved to: {CONF_MATRIX_PATH}")

if __name__ == '__main__':
    train()
