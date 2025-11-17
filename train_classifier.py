import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple

# --- Configuration Constants ---
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
MODEL_PATH = 'models/classifier_weights.pth'

class HealthClassifier(nn.Module):
    """
    A Deep Neural Network (DNN) for binary classification (Normal vs. Anomaly).
    """
    def __init__(self, input_dim):
        super(HealthClassifier, self).__init__()
        # Defining a simple DNN with ReLU activation and Dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output 2 classes: 0 (Normal) or 1 (Anomaly)
        )

    def forward(self, x):
        return self.network(x)

def load_combined_data() -> Tuple[DataLoader, int]:
    """
    Loads the combined real and synthetic training data created in Phase 2
    by 'train_pias_vae.py' (saved as .npy files).
    """
    data_dir = 'data/03_processed'
    X_path = os.path.join(data_dir, 'X_train_combined.npy')
    y_path = os.path.join(data_dir, 'y_train_combined.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("---------------------------------------------------------")
        print(f"FATAL ERROR: Combined data not found in '{data_dir}/'.")
        print("Please ensure 'train_pias_vae.py' was run successfully first.")
        print("Expected files: X_train_combined.npy and y_train_combined.npy")
        print("---------------------------------------------------------")
        raise FileNotFoundError("Combined training data is missing.") 

    X_combined = np.load(X_path)
    y_combined = np.load(y_path)
    
    print(f"Loaded combined data successfully. X shape: {X_combined.shape}, Y shape: {y_combined.shape}")
    
    # Convert to PyTorch Tensors
    X_tensor = torch.from_numpy(X_combined).float()
    y_tensor = torch.from_numpy(y_combined).long()
    
    # Create DataLoader for training
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # The input dimension is derived from the loaded data
    input_dim = X_combined.shape[1]
    return data_loader, input_dim

def train_classifier(model, data_loader, criterion, optimizer, device):
    """
    Trains the classifier model for one epoch.
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    # Iterate over the data batches with a progress bar
    for data, labels in tqdm(data_loader, desc="Training Classifier"):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        
        # Calculate accuracy for monitoring
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct_predictions / len(data_loader.dataset)
    return avg_loss, accuracy

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Combined Training Data
    try:
        data_loader, input_dim = load_combined_data()
    except FileNotFoundError:
        return # Exit if data is not available

    # 2. Initialize Classifier Model and Components
    model = HealthClassifier(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss() # Standard loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n--- Starting PIAS-VAE Phase 3: Training Classifier ({NUM_EPOCHS} epochs) ---")
    
    # 3. Training Loop
    best_accuracy = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        loss, accuracy = train_classifier(model, data_loader, criterion, optimizer, device)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS}: Loss: {loss:.6f}, Accuracy: {accuracy:.4f}")
        
        # Save the model only if the current accuracy is the best seen so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs('models', exist_ok=True)
            # --- CONFIRMATION MESSAGE ADDED HERE ---
            print(f"*** Saving best model weights with accuracy: {best_accuracy:.4f} ***")
            torch.save(model.state_dict(), MODEL_PATH)
            
    print(f"\nClassifier training complete. Final best accuracy: {best_accuracy:.4f}")
    print(f"Final weights saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()