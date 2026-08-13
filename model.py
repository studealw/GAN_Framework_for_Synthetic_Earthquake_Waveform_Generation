import torch
from torch import nn
from dataclasses import dataclass

test_data = torch.randn(32, 200, 3)  # Example input data with shape (batch_size, sequence_length, channels)

@dataclass
class GeneratorModelConfig:
    batch_size: int = 32
    sequence_length: int = 200
    channels: int = 3
    kernel_size_1: int = 16
    stride_1: int = 2 
    latent_dim : int = 100
    padding : int = 7

class GeneratorModel(nn.Module):
    def __init__(self, config: GeneratorModelConfig):
        super().__init__()
        self.config = config

        # self.initial_dense = nn.Sequential(
        # nn.Linear(self.config.latent_dim, self.config.channels * 25),
        # nn.ReLU()
        # )

        self.conv_block_1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels = self.config.channels,
                out_channels = 128,
                kernel_size = self.config.kernel_size_1,
                stride = self.config.stride_1,
                padding = self.config.padding
            ),
            nn.GroupNorm(1,128),
            nn.ReLU()  
        )

        self.conv_block_2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels = 128,
                out_channels = 64,
                kernel_size = self.config.kernel_size_1,
                stride = self.config.stride_1,
                padding = self.config.padding
            ),
            nn.GroupNorm(1,64),
            nn.ReLU()  
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels = 64,
                out_channels = 3,
                kernel_size = self.config.kernel_size_1,
                stride = self.config.stride_1,
                padding = self.config.padding
            ),
            nn.ReLU()  
        )


    def forward(self, x):
        # x = self.initial_dense(x)
        x = x.view(-1, self.config.channels, 25)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        return x

model_0 = GeneratorModel(GeneratorModelConfig()) 

print(model_0(test_data).shape)  


