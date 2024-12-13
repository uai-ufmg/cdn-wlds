import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_size, output_channels):
        """
            Arguments:
            ---------
                - input_size (int): Number of channels coming from the input
                - output_channels (int): Number of channels generated data needs to have
            Return:
            ---------
                None
        """
        super(Generator, self).__init__()
        self.input_size = input_size
        self.output_channels = output_channels

        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose1d(in_channels=self.input_size, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4)
            nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(True),

            # State (512x8)
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(True),
            
            # State (256x16)
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),
            
            # State (128x32)
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            
            # State (64x64)
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(True),

            # State (32x128)
            nn.ConvTranspose1d(in_channels=32, out_channels=self.output_channels, kernel_size=10, stride=2, padding=4),
            nn.Sigmoid()
        )
            # output of main module --> Image (Cx256)


    def forward(self, x):
        """
            Arguments:
            ---------
                - x (torch.Tensor): Noise data
            Return:
            ---------
                - x (torch.Tensor): Generated well log sequence
        """
        x = self.main_module(x)
        return x

    def use_checkpointing(self):
        self.main_module = torch.utils.checkpoint(self.main_module)
