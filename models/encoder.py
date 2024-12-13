import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_channels, z_size):
        """
            Arguments:
            ---------
                - n_channels (int): Number of channels coming from the input data (number of logs)
                - z_size (int): Size of the noise vector z used to generate data
            Return:
            ---------
                None
        """
        super(Encoder, self).__init__()
        
        self.n_channels = n_channels

        self.main_module = nn.Sequential(
            # Image (Cx256)
            nn.Conv1d(in_channels=n_channels, out_channels=128, kernel_size=10, stride=2, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (128x128)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x64)
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x32)
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
            # outptut of main module --> State (1024x16)
        
        self.input_last_layer = 1024*16
        
        self.out_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_last_layer, z_size)
        )
    
    def forward(self, x):
        """
            Arguments:
            ---------
                - x (torch.Tensor): Well log sequence
            Return:
            ---------
                - x (torch.Tensor): Output embedding
        """
        x = self.main_module(x)
        x = self.out_feature(x)
        x = x.unsqueeze(2)
        return x

    def use_checkpointing(self):
        self.main_module = torch.utils.checkpoint(self.main_module)
        self.out_feature = torch.utils.checkpoint(self.out_feature)
