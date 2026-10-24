import random
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
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
import numpy as np


# Device configuration
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 100
batch_size = 64
learning_rate = 3e-4
beta1 = 0.5
channels = 3
sequence_length = 200
latent_dim = 32
cond_dim = 7
n_critic = 5  # Number of critic updates per generator update


# class EarthquakeNumpyDataset(Dataset):
#     def __init__(self, metadata_csv, waveform_npy, sequence_length=100):
#         self.seq_length = sequence_length

#         # 1. Read Metadata
#         self.df = pd.read_csv(metadata_csv)

#         # 2. Extract ONLY the 7 required columns (Matching your TF Notebook)
#         cond_cols = [
#             'Magnitude', 'Epicenter_Lat', 'Epicenter_Lon',
#             'EQ_Depth(km)', 'Station_Lat', 'Station_Lon', 'Epicentral_Distance_km'
#         ]
#         self.conditions = self.df[cond_cols].values.astype(np.float32)

#         # Normalize conditions (Z-score scaling)
#         self.cond_mean = self.conditions.mean(axis=0)
#         self.cond_std = self.conditions.std(axis=0) + 1e-8
#         self.conditions = (self.conditions - self.cond_mean) / self.cond_std

#         # 3. Load Waveforms from .npy (COMPLETELY IGNORES THE F: DRIVE)
#         raw_waveforms = np.load(waveform_npy).astype(np.float32)

#         # Truncate to N to match CSV length
#         N = len(self.df)
#         raw_waveforms = raw_waveforms[:N]

#         # Transpose from (Batch, Time, Channels) -> (Batch, Channels, Time) -> (N, 3, 100)
#         raw_waveforms = np.transpose(raw_waveforms, (0, 2, 1))

#         # Normalize waveforms (Z-score per channel, per waveform)
#         means = raw_waveforms.mean(axis=2, keepdims=True)
#         stds = raw_waveforms.std(axis=2, keepdims=True) + 1e-8
#         self.waveforms = (raw_waveforms - means) / stds

#         # 4. Convert to PyTorch Tensors
#         self.waveforms = torch.tensor(self.waveforms, dtype=torch.float32)
#         self.conditions = torch.tensor(self.conditions, dtype=torch.float32)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         # Returns the pre-loaded, pre-normalized tensors instantly
#         return self.waveforms[idx], self.conditions[idx]
class EarthquakeNumpyDataset(Dataset):
    def __init__(self, metadata_csv, waveform_npy, sequence_length=100):
        self.seq_length = sequence_length

        # 1. Read Metadata
        self.df = pd.read_csv(metadata_csv)

        # NEW: Compute Epicentral Distance if it's missing from the CSV
        if 'Epicentral_Distance_km' not in self.df.columns:
            lat1 = np.radians(self.df['Epicenter_Lat'])
            lon1 = np.radians(self.df['Epicenter_Lon'])
            lat2 = np.radians(self.df['Station_Lat'])
            lon2 = np.radians(self.df['Station_Lon'])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * \
                np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))

            # 6371.0 is the approximate radius of the Earth in km
            self.df['Epicentral_Distance_km'] = 6371.0 * c

        # 2. Extract ONLY the 7 required columns
        cond_cols = [
            'Magnitude', 'Epicenter_Lat', 'Epicenter_Lon',
            'EQ_Depth(km)', 'Station_Lat', 'Station_Lon', 'Epicentral_Distance_km'
        ]
        self.conditions = self.df[cond_cols].values.astype(np.float32)

        # Normalize conditions (Z-score scaling)
        self.cond_mean = self.conditions.mean(axis=0)
        self.cond_std = self.conditions.std(axis=0) + 1e-8
        self.conditions = (self.conditions - self.cond_mean) / self.cond_std

        # 3. Load Waveforms from .npy
        raw_waveforms = np.load(waveform_npy).astype(np.float32)

        # Truncate to N to match CSV length
        N = len(self.df)
        raw_waveforms = raw_waveforms[:N]

        # Transpose from (Batch, Time, Channels) -> (Batch, Channels, Time) -> (N, 3, 100)
        raw_waveforms = np.transpose(raw_waveforms, (0, 2, 1))

        # Normalize waveforms (Z-score per channel, per waveform)
        means = raw_waveforms.mean(axis=2, keepdims=True)
        stds = raw_waveforms.std(axis=2, keepdims=True) + 1e-8
        self.waveforms = (raw_waveforms - means) / stds

        # 4. Convert to PyTorch Tensors
        self.waveforms = torch.tensor(self.waveforms, dtype=torch.float32)
        self.conditions = torch.tensor(self.conditions, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Returns the pre-loaded, pre-normalized tensors instantly
        return self.waveforms[idx], self.conditions[idx]


# Dataloaders
# dataset = EarthquakeNumpyDataset(
#     metadata_csv=r"waveform_folder\Test_tab_MAG_data_3_SEC.csv",
#     # PUT THE EXACT PATH TO YOUR .NPY FILE HERE:
#     waveform_npy=r"C:\Users\JAY\Downloads\Test_waveform_MAG_data_3C.npy",
#     sequence_length=sequence_length
# )
dataset = EarthquakeNumpyDataset(
    metadata_csv="Test_tab_MAG_data_3_SEC.csv",
    waveform_npy="Test_waveform_MAG_data_3C.npy",
    sequence_length=sequence_length
)
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Calling the Models
encoder = EncoderModel(sequence_length=sequence_length,
                       latent_dim=latent_dim, cond_dim=cond_dim).to(device)
decoder = DecoderGeneratorModel(DecoderGeneratorModelConfig(
    sequence_length=sequence_length, latent_dim=latent_dim, cond_dim=cond_dim)).to(device)
critic = CriticModel(
    CriticConfig(), sequence_length=sequence_length).to(device)

# Loading the model weights
encoder.load_state_dict(torch.load("encoder_weights.pth"))
decoder.load_state_dict(torch.load("decoder_weights.pth"))
critic.load_state_dict(torch.load("critic_weights.pth"))

# TESTING
print("\n--- Model Loaded. Starting Testing Phase ---")
encoder.eval()
decoder.eval()
critic.eval()
test_recon_error = 0.0

# Reconstruction Loss (Mean Absolute Error)
criterion_recon = nn.L1Loss()

# 1. Create empty lists to collect the data
all_real = []
all_fake = []
all_cond = []

with torch.inference_mode():
    for test_real, test_c in test_dataloader:
        test_real, test_c = test_real.to(device), test_c.to(device)

        # Forward pass
        mu, logvar = encoder(test_real, test_c)
        z = reparameterize(mu, logvar)
        test_fake = decoder(z, test_c)

        test_recon_error += criterion_recon(test_fake, test_real).item()

        # 2. Move tensors to CPU, convert to numpy, and save to lists
        all_real.append(test_real.cpu().numpy())
        all_fake.append(test_fake.cpu().numpy())
        all_cond.append(test_c.cpu().numpy())

avg_test_recon = test_recon_error / len(test_dataloader)
print(f"Final Test Reconstruction Error (L1): {avg_test_recon:.4f}")

# 3. Stitch the lists into massive numpy arrays (Exactly what your plot needs!)
real_final = np.concatenate(all_real, axis=0)
pred_R_final = np.concatenate(all_fake, axis=0)
y_test_R = np.concatenate(all_cond, axis=0)

print(
    f"Shapes for plotting -> Real: {real_final.shape}, Fake: {pred_R_final.shape}")


# # CLEAN 4-SAMPLE PLOTTING SCRIPT

# print("Generating Clean Grid for 4 Earthquakes...")
# ch = 0  # 0: NS, 1: EW, 2: UD
# cmap = "viridis"
# N = pred_R_final.shape[0]

# # Sample exactly 4 random indices
# idxs = random.sample(range(N), 4)

# # Build an 8x2 grid (4 earthquakes * 2 rows each)
# fig, axes = plt.subplots(8, 2, figsize=(15, 16), constrained_layout=True)

# for r, k in enumerate(idxs):
#     real = real_final[k, ch]
#     fake = pred_R_final[k, ch]
#     mag = y_test_R[k, 0]

#     # ------------------ Waveforms ------------------
#     axes[2*r, 0].plot(real)
#     axes[2*r, 0].set_title(f"Real Wave | idx={k} | Norm M={mag:.2f}")
#     axes[2*r, 0].margins(x=0)

#     axes[2*r, 1].plot(fake)
#     axes[2*r, 1].set_title(f"Generated Wave | idx={k} | Norm M={mag:.2f}")
#     axes[2*r, 1].margins(x=0)

#     # ----------------- Spectrograms -----------------
#     f1, t1, S1 = spectrogram(real, fs=100, nperseg=20, noverlap=10)
#     f2, t2, S2 = spectrogram(fake, fs=100, nperseg=20, noverlap=10)

#     S1_db = 10 * np.log10(S1 + 1e-8)
#     S2_db = 10 * np.log10(S2 + 1e-8)

#     vmin = S1_db.min()
#     vmax = S1_db.max()

#     im1 = axes[2*r+1, 0].pcolormesh(t1, f1, S1_db,
#                                     shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
#     axes[2*r+1, 0].set_ylabel("Freq (Hz)")
#     fig.colorbar(im1, ax=axes[2*r+1, 0], label="dB")

#     im2 = axes[2*r+1, 1].pcolormesh(t2, f2, S2_db,
#                                     shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
#     axes[2*r+1, 1].set_ylabel("Freq (Hz)")
#     fig.colorbar(im2, ax=axes[2*r+1, 1], label="dB")

# # plt.show() for local testing, but save the figure for google colab


# # Save the figure to disk for gogole colab
# plt.savefig("earthquake_comparison.png", bbox_inches='tight')
# plt.close()

# ============================================================
# 1. EXPORT REAL & SYNTHETIC WAVEFORMS TO CSV
# ============================================================
print("Exporting waveforms to CSV...")

channel_names = ['NS', 'EW', 'UD']
time_steps = real_final.shape[2]  # typically 200


def export_waveforms_to_csv(data_array, filename):
    """
    Flattens (N, Channels, Time) into tabular rows:
    sample_idx | channel | t_0 | t_1 | ... | t_{time_steps-1}
    """
    rows = []
    col_names = ['sample_idx', 'channel'] + \
        [f't_{i}' for i in range(time_steps)]

    N_samples = data_array.shape[0]
    for i in range(N_samples):
        for c_idx, c_name in enumerate(channel_names):
            row = [i, c_name] + data_array[i, c_idx].tolist()
            rows.append(row)

    df_out = pd.DataFrame(rows, columns=col_names)
    df_out.to_csv(filename, index=False)
    print(f"Saved: {filename} (Shape: {df_out.shape})")


export_waveforms_to_csv(real_final, "real_waveforms.csv")
export_waveforms_to_csv(pred_R_final, "synthetic_waveforms.csv")


# ============================================================
# 2. 10-SAMPLE SPECTROGRAM & WAVEFORM COMPARISON GRID
# ============================================================
print("Generating 10-Sample Comparison Grid...")

num_samples = 10
ch = 0  # 0: NS, 1: EW, 2: UD
cmap = "viridis"

# Select 10 random indices
selected_idxs = random.sample(range(real_final.shape[0]), num_samples)

# 10 samples * 2 rows each (Waveform row + Spectrogram row) = 20 rows total
fig, axes = plt.subplots(
    num_samples * 2, 2, figsize=(16, 4 * num_samples), constrained_layout=True)

for r, k in enumerate(selected_idxs):
    real_wave = real_final[k, ch]
    fake_wave = pred_R_final[k, ch]
    mag = y_test_R[k, 0]

    # --- Waveform Row ---
    axes[2*r, 0].plot(real_wave, color='steelblue')
    axes[2*r, 0].set_title(
        f"Real Waveform | Sample #{k} | Norm M={mag:.2f} ({channel_names[ch]})")
    axes[2*r, 0].margins(x=0)
    axes[2*r, 0].set_ylabel("Amplitude")

    axes[2*r, 1].plot(fake_wave, color='darkorange')
    axes[2*r, 1].set_title(
        f"Synthetic Waveform | Sample #{k} | Norm M={mag:.2f} ({channel_names[ch]})")
    axes[2*r, 1].margins(x=0)
    axes[2*r, 1].set_ylabel("Amplitude")

    # --- Spectrogram Computation ---
    f_real, t_real, S_real = spectrogram(
        real_wave, fs=100, nperseg=20, noverlap=10)
    f_fake, t_fake, S_fake = spectrogram(
        fake_wave, fs=100, nperseg=20, noverlap=10)

    S_real_db = 10 * np.log10(S_real + 1e-8)
    S_fake_db = 10 * np.log10(S_fake + 1e-8)

    # Use a unified scale per sample pair for fair comparison
    vmin = min(S_real_db.min(), S_fake_db.min())
    vmax = max(S_real_db.max(), S_fake_db.max())

    # --- Spectrogram Row ---
    im_real = axes[2*r+1, 0].pcolormesh(t_real, f_real, S_real_db,
                                        shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2*r+1, 0].set_ylabel("Freq (Hz)")
    axes[2*r+1, 0].set_xlabel("Time (s)")
    fig.colorbar(im_real, ax=axes[2*r+1, 0], label="dB")

    im_fake = axes[2*r+1, 1].pcolormesh(t_fake, f_fake, S_fake_db,
                                        shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2*r+1, 1].set_ylabel("Freq (Hz)")
    axes[2*r+1, 1].set_xlabel("Time (s)")
    fig.colorbar(im_fake, ax=axes[2*r+1, 1], label="dB")

plt.savefig("spectrogram_10_samples_comparison.png",
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: spectrogram_10_samples_comparison.png")


# 1. Create a dedicated folder to avoid cluttering your directory
output_dir = "10_events_comparison_csv"
os.makedirs(output_dir, exist_ok=True)

channel_names = ['NS', 'EW', 'UD']
time_steps = real_final.shape[2]
time_array = np.arange(time_steps)

# 'selected_idxs' is the list of 10 random samples from the previous spectrogram step
print(f"Exporting 10 individual event CSV files to '{output_dir}/' ...")

for i, k in enumerate(selected_idxs):
    # Extract data for this specific earthquake event (shape: 3 x time_steps)
    real_wave = real_final[k]
    fake_wave = pred_R_final[k]

    # Initialize dictionary with time steps
    event_data = {'Time_Step': time_array}

    # Populate the dictionary with Real and Synthetic columns for each channel
    for c_idx, c_name in enumerate(channel_names):
        event_data[f'Real_{c_name}'] = real_wave[c_idx]
        event_data[f'Synthetic_{c_name}'] = fake_wave[c_idx]

    # Convert to DataFrame and save
    df_event = pd.DataFrame(event_data)

    # Name the file 1 through 10, but keep the original dataset index 'k' in the name for reference
    filename = os.path.join(output_dir, f"event_{i+1}_dataset_idx_{k}.csv")
    df_event.to_csv(filename, index=False)

    print(f"Saved {filename} | Shape: {df_event.shape}")

print("All 10 event CSVs successfully generated.")
