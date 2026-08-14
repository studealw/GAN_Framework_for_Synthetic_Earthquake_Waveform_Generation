import torch
from torch import nn
from dataclasses import dataclass

test_data = torch.randn(32, 100)  # Example input data with shape (batch_size, sequence_length, channels)

@dataclass
class GeneratorModelConfig:
    batch_size: int = 32
    sequence_length: int = 200
    channels: int = 3
    kernel_size_1: int = 5
    stride_1: int = 5
    latent_dim : int = 100
    padding : int = 0

class GeneratorModel(nn.Module):
    def __init__(self, config: GeneratorModelConfig):
        super().__init__()
        self.config = config

        self.initial_dense = nn.Sequential(
        nn.Linear(self.config.latent_dim, 256 * 25),
        nn.ReLU()
        )

        self.conv_block_1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels = 256,
                out_channels = 128,
                kernel_size = 5,
                stride = 5,
                padding = 0
            ),
            nn.GroupNorm(1,128),
            nn.ReLU()  
        )

        self.conv_block_2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels = 128,
                out_channels = 64,
                kernel_size = 4,
                stride = 2,
                padding = 1
            ),
            nn.GroupNorm(1,64),
            nn.Tanh()  
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels = 64,
                out_channels = 3,
                kernel_size = 4,
                stride =2,
                padding = 1
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

print(model_0(test_data).shape)  


