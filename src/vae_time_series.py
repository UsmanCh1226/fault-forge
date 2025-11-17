import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Constants (Must match data preparation settings) ---
# Assuming a flattened sequence length based on typical CMAPSS setup
# SEQUENCE_LENGTH * NUM_FEATURES (e.g., 30 * 14 = 420)
INPUT_DIM = 420 
LATENT_DIM = 20 # Dimension of the latent space (Z)

class Encoder(nn.Module):
    """
    Encodes the flattened time series sequence into the latent space (mu and log_var).
    """
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

class Decoder(nn.Module):
    """
    Decodes a sample from the latent space back into the original time series space.
    """
    def __init__(self, latent_dim=LATENT_DIM, output_dim=INPUT_DIM):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        # Final layer outputs the reconstructed sequence
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        # Use Sigmoid to scale output between 0 and 1, matching the normalized input data
        return torch.sigmoid(self.fc3(h))

class PIASVAE(nn.Module):
    """
    The complete PIAS-VAE model combining the Encoder and Decoder.
    """
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super(PIASVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, log_var):
        """
        The reparameterization trick to sample Z from the latent distribution.
        """
        std = torch.exp(0.5 * log_var)
        # Generate random noise (epsilon)
        eps = torch.randn_like(std)
        # Z = mu + std * epsilon
        return mu + eps * std

    def forward(self, x):
        # 1. Encode
        mu, log_var = self.encoder(x)
        
        # 2. Reparameterize (Sample Z)
        z = self.reparameterize(mu, log_var)
        
        # 3. Decode
        recon_x = self.decoder(z)
        
        return recon_x, mu, log_var, z

def vae_loss_function(recon_x, x, mu, log_var, kld_weight):
    """
    The VAE loss function: Reconstruction Loss + KLD Loss.
    
    1. Reconstruction Loss (BCE): Measures how well the VAE reconstructs the input.
    2. KLD Loss (KL Divergence): Measures how close the latent distribution is to a standard Gaussian.
    """
    # 1. Reconstruction Loss (Binary Cross Entropy)
    # The input 'x' and output 'recon_x' are normalized between 0 and 1
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_DIM), reduction='sum')

    # 2. KL Divergence Loss
    # KLD = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Final VAE Loss: Weighted sum
    return BCE + kld_weight * KLD

if __name__ == '__main__':
    # Example usage (will only run if this file is executed directly)
    model = PIASVAE()
    print(model)