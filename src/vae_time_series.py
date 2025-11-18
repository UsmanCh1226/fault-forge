import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
# Add src directory to the path to allow importing vae_time_series
sys.path.append(os.path.join(os.getcwd(), 'src')) 

# --- Import VAE Model and Constants from  file ---
try:
    from vae_time_series import PIASVAE, vae_loss_function, INPUT_DIM, LATENT_DIM
except ImportError:
    print("Error: Could not import PIASVAE from src/vae_time_series.py.")
    print("Please ensure your vae_time_series.py is in a 'src' subdirectory.")
    sys.exit(1)


# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MODEL_PATH = 'models/vae_normal_manifold_weights.pth'
# KLD_WEIGHT is the beta parameter in the VAE formulation. 
# We'll start with 1.0 (standard VAE)
KLD_WEIGHT = 1.0 

def load_data():
    """Load combined data and filter for Normal (label 0) sequences."""
    try:
        X_combined = np.load('data/03_processed/X_train_combined.npy')
        y_combined = np.load('data/03_processed/y_train_combined.npy')
    except FileNotFoundError:
        print("Error: Required data files not found. Ensure 'data/03_processed' exists and contains X_train_combined.npy and y_train_combined.npy.")
        return None

    # Filter for Normal data (Label 0) for Phase 1 VAE training
    X_normal = X_combined[y_combined == 0]
    # Ensure data is flattened as required by the VAE architecture
    X_normal = X_normal.reshape(X_normal.shape[0], -1) 
    
    # Check dimensions match the expected INPUT_DIM
    if X_normal.shape[1] != INPUT_DIM:
        print(f"Dimension Mismatch: Loaded data has {X_normal.shape[1]} features, but VAE expects {INPUT_DIM}.")
        print("Please check your data preparation script and the INPUT_DIM constant in vae_time_series.py.")
        return None
        
    print(f"Loaded {X_combined.shape[0]} total samples. Training VAE on {X_normal.shape[0]} Normal samples.")

    # Convert to PyTorch tensors
    X_normal_tensor = torch.from_numpy(X_normal).float()
    
    # Create DataLoader
    dataset = TensorDataset(X_normal_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return dataloader

def train_vae(dataloader):
    """Main training loop for the VAE."""
    # Initialize VAE using constants from the imported model file
    vae = PIASVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    
    # Storage for plotting the loss history
    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

    print(f"\n--- Starting PIAS-VAE Phase 1: VAE Training on Normal Data ({EPOCHS} epochs) ---")
    print(f"Model Specs: Input={INPUT_DIM}, Latent={LATENT_DIM}, KLD Weight={KLD_WEIGHT}")
    
    for epoch in range(1, EPOCHS + 1):
        vae.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        for batch_idx, data in enumerate(dataloader):
            x = data[0].to(DEVICE)
            optimizer.zero_grad()
            
            # The PIASVAE forward method returns 4 values, but we only need the first 3 for loss
            recon_x, mu, log_var, _ = vae(x)
            
            # Use the loss function from your model file
            loss, KLD = vae_loss_function(recon_x, x, mu, log_var, kld_weight=KLD_WEIGHT)
            
            # Since vae_loss_function returns the total loss and KLD, we calculate BCE/Recon loss here
            recon_loss = loss - KLD 
            
            loss.backward()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += KLD.item()
            optimizer.step()

        # Average losses over the dataset (normalized by number of samples)
        num_samples = len(dataloader.dataset)
        avg_loss = train_loss / num_samples
        avg_recon_loss = train_recon_loss / num_samples
        avg_kl_loss = train_kl_loss / num_samples
        
        history['total_loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{EPOCHS}: Total VAE Loss: {avg_loss:.4f} | Recon Loss: {avg_recon_loss:.4f} | KL Loss: {avg_kl_loss:.4f}')

    # Save final model weights
    os.makedirs('models', exist_ok=True)
    torch.save(vae.state_dict(), MODEL_PATH)
    print(f"\nVAE training complete. Weights saved to '{MODEL_PATH}'")
    
    return vae, history

def plot_loss(history):
    """Plots the VAE training loss components."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['total_loss'], label='Total VAE Loss')
    plt.plot(history['recon_loss'], label='Reconstruction Loss (BCE)')
    plt.plot(history['kl_loss'], label=f'KL Divergence Loss (Weighted by beta={KLD_WEIGHT})')
    plt.title('Phase 1: PIAS-VAE Training Loss History (Normal Data)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (per sample)')
    plt.legend()
    plt.grid(True)
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/vae_loss_history.png')
    plt.show()

def main():
    dataloader = load_data()
    if dataloader is None:
        return
        
    vae, history = train_vae(dataloader)
    
    plot_loss(history)
    print("Loss history saved to 'reports/vae_loss_history.png'.")

if __name__ == '__main__':
    main()