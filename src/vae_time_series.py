# src/vae_time_series.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional, List

# -------------------------
# Configuration (tweak as needed)
# -------------------------
SEQUENCE_LENGTH = 30
NUM_FEATURES = 14
LATENT_DIM = 20
HIDDEN_DIM = 128
NUM_LAYERS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# PIAS LSTM-VAE
# -------------------------
class PIASVAE(nn.Module):
    """
    LSTM-based VAE for multivariate time-series sequences.
    Input: (batch, seq_len, num_features)
    Output: reconstructions with same shape, plus mu/logvar
    """
    def __init__(self, seq_len: int = SEQUENCE_LENGTH, num_features: int = NUM_FEATURES,
                 latent_dim: int = LATENT_DIM, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: LSTM -> take last hidden state -> project to mu/logvar
        self.encoder_lstm = nn.LSTM(input_size=num_features,
                                    hidden_size=hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    bidirectional=False)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: map latent z -> repeated inputs to LSTM -> produce sequence
        # We'll feed the latent vector (or transformed) as input at every time step.
        self.latent_to_dec_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_size=hidden_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, num_features)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, num_features)
        _, (h_n, _) = self.encoder_lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        h_last = h_n[-1]                    # (batch, hidden_dim)
        mu = self.fc_mu(h_last)             # (batch, latent_dim)
        logvar = self.fc_logvar(h_last)     # (batch, latent_dim)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        # transform z -> per-timestep input
        dec_input = self.latent_to_dec_input(z)          # (batch, hidden_dim)
        # repeat across sequence length
        dec_input_seq = dec_input.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        dec_out, _ = self.decoder_lstm(dec_input_seq)   # (batch, seq_len, hidden_dim)
        out_seq = self.output_fc(dec_out)               # (batch, seq_len, num_features)
        return out_seq

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# -------------------------
# Loss function helpers
# -------------------------
def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                      kld_weight: float = 1.0, reduction: str = "mean") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes MSE reconstruction + KL divergence.
    Returns (total_loss, recon_loss, kld_loss) -- each scalar tensors.
    reduction: 'mean' (per batch average) or 'sum'
    """
    # Reconstruction loss (MSE)
    mse = nn.MSELoss(reduction=reduction)
    recon_loss = mse(recon_x, x)

    # KL divergence per batch (mean over batch)
    # KLD = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) -> averaged over batch
    kld_element = 1 + logvar - mu.pow(2) - logvar.exp()
    kld_loss = -0.5 * torch.mean(torch.sum(kld_element, dim=1))  # mean over batch

    total_loss = recon_loss + kld_weight * kld_loss
    return total_loss, recon_loss, kld_loss

# -------------------------
# Training loop (callable)
# -------------------------
def train_vae_from_dataloader(dataloader: DataLoader,
                              epochs: int = 100,
                              lr: float = 1e-3,
                              kld_weight: float = 1.0,
                              kl_anneal: Optional[dict] = None,
                              model_savepath: str = 'models/vae_normal_manifold_weights.pth',
                              device: torch.device = DEVICE) -> Tuple[PIASVAE, dict]:
    """
    Train the PIASVAE on sequences provided by dataloader.
    dataloader should yield batches of shape (batch, seq_len, num_features) as float tensors.
    kl_anneal: optional dict {'start':0.0, 'end':1.0, 'n_epochs':50} to linearly anneal KLD weight.
    Returns trained model and history dict.
    """
    model = PIASVAE(seq_len=SEQUENCE_LENGTH, num_features=NUM_FEATURES, latent_dim=LATENT_DIM,
                    hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'total_loss': [], 'recon_loss': [], 'kld_loss': []}

    # Prepare KL annealing schedule if provided
    if kl_anneal is not None:
        start, end, n_epochs = kl_anneal.get('start', 0.0), kl_anneal.get('end', 1.0), kl_anneal.get('n_epochs', 50)
    else:
        start, end, n_epochs = kld_weight, kld_weight, 1

    n_batches = len(dataloader)
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0
        for batch_idx, batch in enumerate(dataloader):
            x_batch = batch[0].to(device)  # expecting TensorDataset with single tensor
            optimizer.zero_grad()

            recon_x, mu, logvar = model(x_batch)
            # compute current kld multiplier
            if kl_anneal is not None:
                # linear schedule across epochs
                alpha = float(min(epoch, n_epochs)) / float(max(1, n_epochs))
                cur_kld_weight = start + alpha * (end - start)
            else:
                cur_kld_weight = kld_weight

            total_loss, recon_loss, kld_loss = vae_loss_function(recon_x, x_batch, mu, logvar,
                                                                kld_weight=cur_kld_weight, reduction="mean")
            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kld += kld_loss.item()

        # average over batches
        history['total_loss'].append(epoch_total / n_batches)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['kld_loss'].append(epoch_kld / n_batches)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Total: {history['total_loss'][-1]:.6f} | Recon: {history['recon_loss'][-1]:.6f} | KLD: {history['kld_loss'][-1]:.6f} | kld_w: {cur_kld_weight:.4f}")

    os.makedirs(os.path.dirname(model_savepath) or '.', exist_ok=True)
    torch.save(model.state_dict(), model_savepath)
    print(f"Saved VAE weights to {model_savepath}")
    return model, history

# -------------------------
# Helpers: encode dataset -> latents, decode latents -> sequences
# -------------------------
def encode_dataset_to_latents(model: PIASVAE, data_tensor: torch.Tensor, batch_size: int = 128,
                              device: torch.device = DEVICE) -> np.ndarray:
    """
    Encodes a dataset of sequences (shape: N, seq_len, num_features) into latent vectors (N, latent_dim).
    """
    model.eval()
    loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)
    latents = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            mu, logvar = model.encode(x)
            # Use mu as latent embedding (deterministic)
            latents.append(mu.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    return latents  # shape (N, latent_dim)

def decode_latents_to_sequences(model: PIASVAE, latents: np.ndarray, batch_size: int = 128,
                                device: torch.device = DEVICE) -> np.ndarray:
    """
    Decode latents (N, latent_dim) back into sequences (N, seq_len, num_features).
    """
    model.eval()
    lat_t = torch.from_numpy(latents).float()
    loader = DataLoader(TensorDataset(lat_t), batch_size=batch_size, shuffle=False)
    decoded = []
    with torch.no_grad():
        for batch in loader:
            z = batch[0].to(device)
            recon = model.decode(z)   # (batch, seq_len, num_features)
            decoded.append(recon.cpu().numpy())
    decoded = np.concatenate(decoded, axis=0)
    return decoded

def interpolate_and_synthesize(model: PIASVAE,
                               normal_latents_center: np.ndarray,
                               anomaly_latents: np.ndarray,
                               n_per_anomaly: int = 50,
                               alphas: Optional[List[float]] = None,
                               device: torch.device = DEVICE) -> np.ndarray:
    """
    For each anomaly latent vector, interpolate between the normal center and the anomaly latent,
    decode multiple intermediate latents and produce synthetic sequences.

    normal_latents_center: (latent_dim,) center of normal latent space (e.g., mean of normals)
    anomaly_latents: (M, latent_dim) real anomaly latents
    Returns synthetic_sequences: (M * n_per_anomaly * len(alphas)?, seq_len, num_features)
    """

    if alphas is None:
        # default: create n_per_anomaly evenly spaced interpolation coefficients (excluding 0 to avoid pure normal)
        alphas = np.linspace(0.1, 1.0, n_per_anomaly)

    synth_list = []
    for a_idx, alpha in enumerate(alphas):
        # create a batch of interpolations for all anomaly latents at this alpha:
        # z_interp = (1-alpha) * z_normal_center + alpha * z_anomaly
        z_interp = (1.0 - alpha) * normal_latents_center[np.newaxis, :] + alpha * anomaly_latents
        # decode them in batches
        decoded = decode_latents_to_sequences(model, z_interp, device=device)
        synth_list.append(decoded)

    # synth_list is list of arrays (len(alphas), (M, seq_len, num_features))
    # reshape to (M * len(alphas), seq_len, num_features)
    synth_all = np.concatenate(synth_list, axis=0)
    return synth_all

# -------------------------
# Convenience pipeline to produce synthetic anomalies and save them
# -------------------------
def synthesize_and_save(model: PIASVAE,
                        X_all: np.ndarray,
                        y_all: np.ndarray,
                        save_dir: str = 'data/03_processed',
                        n_per_anomaly: int = 50,
                        alphas: Optional[List[float]] = None,
                        device: torch.device = DEVICE) -> None:
    """
    High-level helper:
    - X_all: numpy array (N, seq_len, num_features)
    - y_all: numpy array (N,) labels (0 - normal, 1 - anomaly)
    Produces synthetic anomalies and saves:
      X_train_combined.npy, y_train_combined.npy
    """
    os.makedirs(save_dir, exist_ok=True)

    X_tensor = torch.from_numpy(X_all).float()
    # encode normals and anomalies
    normal_mask = (y_all == 0)
    anomaly_mask = (y_all == 1)

    if normal_mask.sum() == 0 or anomaly_mask.sum() == 0:
        raise ValueError("Need both normal and anomaly samples to synthesize.")

    normals = X_tensor[normal_mask].to(device)
    anomalies = X_tensor[anomaly_mask].to(device)

    print("Encoding normal data to latents...")
    normal_latents = encode_dataset_to_latents(model, normals, device=device)
    normal_center = np.mean(normal_latents, axis=0)  # (latent_dim,)

    print("Encoding real anomalies to latents...")
    anomaly_latents = encode_dataset_to_latents(model, anomalies, device=device)

    print("Interpolating and decoding synthetic anomalies...")
    synthetic_seqs = interpolate_and_synthesize(model, normal_center, anomaly_latents,
                                                n_per_anomaly=n_per_anomaly, alphas=alphas, device=device)

    # Flatten sequences for downstream classifier if desired (or keep 3D)
    # Here we'll save flattened arrays for backward compatibility, but recommend keeping 3D.
    synth_flat = synthetic_seqs.reshape(synthetic_seqs.shape[0], -1)
    X_flat_all = X_all.reshape(X_all.shape[0], -1)

    # Combine: original data + synthetic anomalies
    X_combined = np.concatenate([X_flat_all, synth_flat], axis=0)
    y_synth = np.ones(synth_flat.shape[0], dtype=np.int64)
    y_combined = np.concatenate([y_all, y_synth], axis=0)

    print(f"Saving combined dataset to {save_dir} (X: {X_combined.shape}, y: {y_combined.shape})")
    np.save(os.path.join(save_dir, 'X_train_combined.npy'), X_combined)
    np.save(os.path.join(save_dir, 'y_train_combined.npy'), y_combined)

    print("Synthetic data saved. You can now run your classifier training script.")

# -------------------------
# Example small utility: build dataloader from flattened files (compatibility)
# -------------------------
def dataloader_from_flat_files(X_flat_path: str, y_path: str, batch_size: int = 128, seq_len: int = SEQUENCE_LENGTH,
                               num_features: int = NUM_FEATURES) -> DataLoader:
    """
    Loads flattened X, reshapes to (N, seq_len, num_features) and returns DataLoader (TensorDataset)
    """
    X_flat = np.load(X_flat_path)
    y = np.load(y_path)
    N = X_flat.shape[0]
    X_seq = X_flat.reshape(N, seq_len, num_features)
    X_tensor = torch.from_numpy(X_seq).float()
    y_tensor = torch.from_numpy(y).long()
    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
