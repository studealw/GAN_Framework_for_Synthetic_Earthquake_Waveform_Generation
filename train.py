import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from model import (
    DecoderGeneratorModel, DecoderGeneratorModelConfig,
    CriticModel, CriticConfig,
    EncoderModel, reparameterize
)
import pandas as pd
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

# Device configuration
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 100
batch_size = 64
learning_rate = 3e-4
beta1 = 0.5
channels = 3
sequence_length = 100
latent_dim = 32
cond_dim = 7

# Loss Weights
lambda_gp = 10.0      # Gradient Penalty weight
lambda_recon = 10.0   # Reconstruction loss weight
lambda_kl = 0.5       # KL divergence weight


def apply_butterworth(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


class EarthquakeDataset(Dataset):
    def __init__(self, metadata_csv="waveform.csv", waveform_folder="./waveform", sequence_length=100):
        self.df = pd.read_csv(metadata_csv)
        self.seq_length = sequence_length
        self.waveform_folder = waveform_folder  # Where your NS/EW/UP files are saved

        # 1. Extract 7 Real Conditional Parameters from the CSV
        cond_cols = [
            'Magnitude', 'EQ_Depth(km)', 'Epicentral_Distance_km',
            'Station Height(m)', 'Station_Lat', 'Station_Lon', 'PGA_UD(gal)'
        ]
        self.conditions = self.df[cond_cols].values

        # Normalize the conditions (Crucial for the Generator to learn)
        self.cond_mean = self.conditions.mean(axis=0)
        self.cond_std = self.conditions.std(axis=0) + 1e-8
        self.conditions = (self.conditions - self.cond_mean) / self.cond_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 2. Extract filenames from the absolute paths in the CSV
        # We use os.path.basename just in case your F:\ drive path changed
        ns_name = os.path.basename(row['NS_Waveform_File'])
        ew_name = os.path.basename(row['EW_Waveform_File'])
        up_name = os.path.basename(row['UP_Waveform_File'])

        ns_path = os.path.join(self.waveform_folder, ns_name)
        ew_path = os.path.join(self.waveform_folder, ew_name)
        up_path = os.path.join(self.waveform_folder, up_name)

        # 3. Read the external Waveform CSVs
        try:
            # Assuming these files have an 'Acceleration' column or are single-column
            ns_data = pd.read_csv(ns_path).iloc[:, 0].values
            ew_data = pd.read_csv(ew_path).iloc[:, 0].values
            up_data = pd.read_csv(up_path).iloc[:, 0].values
        except FileNotFoundError:
            # Fallback if a file is missing so training doesn't crash
            ns_data, ew_data, up_data = [
                torch.randn(self.seq_length).numpy()] * 3

        # 4. Filter and Normalize all 3 channels
        channels_data = []
        for raw_data in [ns_data, ew_data, up_data]:
            # Take the required sequence length
            data_window = raw_data[:self.seq_length]

            # Normalize the waveform
            data_window = (data_window - data_window.mean()) / \
                (data_window.std() + 1e-8)

            # Apply your custom Butterworth filter
            data_filtered = apply_butterworth(data_window, 20, 100)
            channels_data.append(data_filtered)

        # Stack into (Channels=3, Sequence_Length=100)
        waveforms = torch.tensor(channels_data, dtype=torch.float32)
        conditions = torch.tensor(self.conditions[idx], dtype=torch.float32)

        return waveforms, conditions


dataset = EarthquakeDataset(
    "Train Tab MAG Data 3 SEC.csv", waveform_folder="./waveforms")

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)
val_dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# Calling the Models
encoder = EncoderModel(sequence_length=sequence_length, channels=channels,
                       latent_dim=latent_dim, cond_dim=cond_dim).to(device)
decoder_config = DecoderGeneratorModelConfig(
    latent_dim=latent_dim, cond_dim=cond_dim, channels=channels)
decoder = DecoderGeneratorModel(decoder_config).to(device)
critic = CriticModel(CriticConfig(), channels=channels,
                     sequence_length=sequence_length, cond_dim=cond_dim).to(device)

# Optimizers
optimizer_E = optim.AdamW(encoder.parameters(),
                          lr=learning_rate, betas=(beta1, 0.999))
optimizer_G = optim.AdamW(decoder.parameters(),
                          lr=learning_rate, betas=(beta1, 0.999))
optimizer_C = optim.AdamW(
    critic.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Learning Rate Schedulers
scheduler_E = optim.lr_scheduler.CosineAnnealingLR(optimizer_E, T_max=epochs)
scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs)
scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=epochs)

# Reconstruction Loss (Mean Absolute Error)
criterion_recon = nn.L1Loss()

# GP function


def compute_gradient_penalty(critic, real_samples, fake_samples, conditions, device):
    alpha = torch.rand((real_samples.size(0), 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)

    d_interpolates = critic(interpolates, conditions)
    fake_grad_outputs = torch.ones(d_interpolates.size(), device=device)

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# TRAINING LOOP
for epoch in range(epochs):
    encoder.train()
    decoder.train()
    critic.train()

    for i, (real_waveforms, conditions) in enumerate(dataloader):
        real_waveforms = real_waveforms.to(device)
        conditions = conditions.to(device)

        # Train Critic
        optimizer_C.zero_grad()

        with torch.no_grad():
            mu, logvar = encoder(real_waveforms, conditions)
            z = reparameterize(mu, logvar)
            fake_waveforms = decoder(z, conditions)

        critic_real = critic(real_waveforms, conditions)
        critic_fake = critic(fake_waveforms.detach(), conditions)

        # Wasserstein Loss + GP
        w_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))
        gp = compute_gradient_penalty(
            critic, real_waveforms, fake_waveforms, conditions, device)

        loss_C = w_loss + (lambda_gp * gp)
        loss_C.backward()
        optimizer_C.step()

        # Train Encoder & Decoder (Generator)
        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        mu, logvar = encoder(real_waveforms, conditions)
        z = reparameterize(mu, logvar)
        fake_waveforms = decoder(z, conditions)

        critic_fake_G = critic(fake_waveforms, conditions)

        # Calculate the 3 Generator Losses
        loss_adv = -torch.mean(critic_fake_G)
        loss_recon = criterion_recon(fake_waveforms, real_waveforms)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss_G_total = loss_adv + \
            (lambda_recon * loss_recon) + (lambda_kl * loss_kl)

        loss_G_total.backward()
        optimizer_E.step()
        optimizer_G.step()

    # Step the learning rate down after every epoch completes
    scheduler_E.step()
    scheduler_G.step()
    scheduler_C.step()

# VALIDATION
    encoder.eval()
    decoder.eval()
    critic.eval()
    val_recon_error = 0.0

    with torch.inference_mode():
        for val_real, val_c in val_dataloader:
            val_real, val_c = val_real.to(device), val_c.to(device)

            # Check reconstruction quality
            mu, logvar = encoder(val_real, val_c)
            z = reparameterize(mu, logvar)
            val_fake = decoder(z, val_c)

            val_recon_error += criterion_recon(val_fake, val_real).item()

    avg_val_recon = val_recon_error / len(val_dataloader)

    if epoch % 1 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Critic Loss: {loss_C.item():.4f} | Gen Total Loss: {loss_G_total.item():.4f} | Val Recon Error: {avg_val_recon:.4f}")

# TESTING
print("\n--- Training Complete. Starting Testing Phase ---")
encoder.eval()
decoder.eval()
critic.eval()
test_recon_error = 0.0

with torch.inference_mode():
    for test_real, test_c in test_dataloader:
        test_real, test_c = test_real.to(device), test_c.to(device)

        mu, logvar = encoder(test_real, test_c)
        z = reparameterize(mu, logvar)
        test_fake = decoder(z, test_c)
        test_recon_error += criterion_recon(test_fake, test_real).item()

avg_test_recon = test_recon_error / len(test_dataloader)
print(f"Final Test Reconstruction Error (L1): {avg_test_recon:.4f}")

# Generate final pure synthetic waveforms from random noise
final_z = torch.randn(batch_size, latent_dim, device=device)
final_c = torch.randn(batch_size, cond_dim, device=device)
final_synthetic_waveforms = decoder(final_z, final_c)
print(
    f"Generated {final_synthetic_waveforms.size(0)} synthetic waveforms based on conditions.")

# SAVING THE MODELS

torch.save(encoder.state_dict(), "encoder_weights.pth")
torch.save(decoder.state_dict(), "decoder_weights.pth")
torch.save(critic.state_dict(), "critic_weights.pth")
print("SAVED THE MODEL")
