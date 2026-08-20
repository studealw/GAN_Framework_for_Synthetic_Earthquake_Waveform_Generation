import torch
from torch import nn
from dataclasses import dataclass
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DecoderGeneratorModelConfig:
    batch_size: int = 32
    sequence_length: int = 100
    channels: int = 3
    kernel_size_1: int = 5
    stride_1: int = 5
    latent_dim: int = 32
    padding: int = 0
    cond_dim: int = 7


@dataclass
class CriticConfig:
    batch_size: int = 32


class DecoderGeneratorModel(nn.Module):
    def __init__(self, config: DecoderGeneratorModelConfig):
        super().__init__()
        self.config = config

        in_features = self.config.latent_dim + self.config.cond_dim

        self.initial_dense = nn.Sequential(
            nn.Linear(in_features, 256 * 5),
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

    def forward(self, z, c):
        z_cond = torch.cat((z, c), dim=1)
        x = self.initial_dense(z_cond)
        x = x.view(-1, 256, 5)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        return x


class CriticModel(nn.Module):
    def __init__(self, config: CriticConfig, channels: int = 3, sequence_length: int = 100, cond_dim=7):
        super().__init__()

        self.config = config
        self.sequence_length = sequence_length

        in_channels = channels + cond_dim

        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
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
            nn.Linear(256 * 5, 1)
        )

    def forward(self, x, c):

        c_expanded = c.unsqueeze(-1).expand(-1, -1, self.sequence_length)

        x_cond = torch.cat((x, c_expanded), dim=1)

        x = self.conv_block_1(x_cond)
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
            nn.LeakyReLU(0.2),

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


def reparameterize(mu, logvar):

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    z = mu + eps * std

    return z


model_1 = CriticModel(CriticConfig())
