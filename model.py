import torch
from torch import nn
from dataclasses import dataclass
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Example input data with shape (batch_size, sequence_length, channels)
noise_input = torch.randn(32, 100)


@dataclass
class GeneratorModelConfig:
    batch_size: int = 32
    sequence_length: int = 200
    channels: int = 3
    kernel_size_1: int = 5
    stride_1: int = 5
    latent_dim: int = 100
    padding: int = 0


@dataclass
class DiscriminatorConfig:
    batch_size: int = 32


class GeneratorModel(nn.Module):
    def __init__(self, config: GeneratorModelConfig):
        super().__init__()
        self.config = config

        self.initial_dense = nn.Sequential(
            nn.Linear(self.config.latent_dim, 256 * 5),
            nn.ReLU()
        )

        self.conv_block_1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                stride=5,
                padding=0
            ),
            nn.GroupNorm(1, 128),
            nn.ReLU()
        )

        self.conv_block_2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.GroupNorm(1, 64),
            nn.Tanh()
        )

        self.conv_block_3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.initial_dense(x)
        x = x.view(-1, 256, 5)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        return x


model_0 = GeneratorModel(GeneratorModelConfig())

# print(model_0(test_data).shape)


class DiscriminatorModel(nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()

        self.config = config

        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=3,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(0.2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=5,
                padding=0
            ),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.classifier(x)

        return x


class EncoderModel(nn.Module):
    def __init__(self, sequence_length: int = 100, channels: int = 3, latent_dim: int = 32, cond_dim: int = 7):
        super().__init__()

        self.seq_length = sequence_length
        self.channels = channels
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        in_channels = self.channels + self.cond_dim

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(0.2),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=5,
                padding=0
            ),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(0.2)

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 5, self.latent_dim)
        self.fc_logvar = nn.Linear(256 * 5, self.latent_dim)

    def forward(self, x, cond):

        cond_expanded = cond.unsqueeze(-1).expand(-1, -1, self.seq_length)
        x_cond = torch.cat((x, cond_expanded), dim=1)

        x_encoded = self.conv_block(x_cond)

        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)

        return mu, logvar



model_1 = DiscriminatorModel(DiscriminatorConfig())

fake_waveforms = model_0(noise_input)

predictions = model_1(fake_waveforms)


print(predictions.shape)  # Output shape should be (batch_size, 1)


generator = model_0.to(device)
discriminator = model_1.to(device)

epochs = 10
learning_rate = 3e-4
beta1 = 0.5
batch_size = 32
latent_dim = 100
sequence_length = 100
channels = 3

criterion = nn.BCELoss()

optimizer_G = optim.AdamW(generator.parameters(),
                          lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.AdamW(discriminator.parameters(),
                          lr=learning_rate, betas=(beta1, 0.999))


dummy_real_data = torch.randn(500, channels, sequence_length)
dataset = TensorDataset(dummy_real_data)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)

# Validation Data
dummy_val_data = torch.randn(100, channels, sequence_length)
val_dataset = TensorDataset(dummy_val_data)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Test Data
dummy_test_data = torch.randn(100, channels, sequence_length)
test_dataset = TensorDataset(dummy_test_data)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

for epoch in range(epochs):
    # Set models to training mode
    generator.train()
    discriminator.train()
    for i, (real_waveforms,) in enumerate(dataloader):

        real_waveforms = real_waveforms.to(device)
        current_batch_size = real_waveforms.size(0)

        real_labels = torch.ones((current_batch_size, 1), device=device)
        fake_labels = torch.zeros((current_batch_size, 1), device=device)

        # Training the discriminator
        optimizer_D.zero_grad()

        # Loss on real data
        predictions_real = discriminator(real_waveforms)
        loss_D_real = criterion(predictions_real, real_labels)

        # Loss on fake data
        noise = torch.randn(current_batch_size, latent_dim, device=device)
        generated_waveforms = generator(noise)

        # Detach the generated waveforms for the discriminator step
        predictions_fake = discriminator(generated_waveforms.detach())
        loss_D_fake = criterion(predictions_fake, fake_labels)

        # Combine and backpropagate
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

       # Training the generator
        optimizer_G.zero_grad()

        # Loss on generated data
        predictions_for_G = discriminator(generated_waveforms)
        loss_G = criterion(predictions_for_G, real_labels)

        # Backpropagate
        loss_G.backward()
        optimizer_G.step()

   # Validation loop
    generator.eval()
    discriminator.eval()
    val_loss_D = 0.0

    with torch.inference_mode():
        for val_real, in val_dataloader:
            val_real = val_real.to(device)
            val_batch_size = val_real.size(0)

            val_real_labels = torch.ones((val_batch_size, 1), device=device)
            val_preds = discriminator(val_real)
            val_loss = criterion(val_preds, val_real_labels)
            val_loss_D += val_loss.item()

    avg_val_loss = val_loss_D / len(val_dataloader)

    # Metrics and logging
    if epoch % 10 == 0:
        print(
            f"Epoch:{epoch} | Discriminator Loss: {loss_D.item():.4f} | Generator Loss: {loss_G.item():.4f} | Validation Discriminator Loss: {avg_val_loss:.4f}")

# Testing
generator.eval()
discriminator.eval()

test_loss_D = 0.0

with torch.inference_mode():
    for test_real, in test_dataloader:
        test_real = test_real.to(device)
        test_batch_size = test_real.size(0)

        test_real_labels = torch.ones((test_batch_size, 1), device=device)
        test_preds = discriminator(test_real)
        test_loss = criterion(test_preds, test_real_labels)
        test_loss_D += test_loss.item()

    avg_test_loss = test_loss_D / len(test_dataloader)

    # Generate final synthetic waveforms using the fully trained Generator
    final_noise = torch.randn(batch_size, latent_dim, device=device)
    final_synthetic_waveforms = generator(final_noise)

print(f"Final Test Discriminator Loss: {avg_test_loss:.4f}")
print(f"Generated {final_synthetic_waveforms.size(0)} ")
