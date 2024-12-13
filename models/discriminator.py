import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_channels, z_dim):
        """
            Arguments:
            ---------
                - n_channels (int): Number of channels coming from the input data (number of logs)
                - z_dim (int): Size of the noise vector z used to generate data
            Return:
            ---------
                None
        """
        super(Discriminator, self).__init__()
        
        self.n_channels = n_channels

        self.main_x_module = nn.Sequential(
            # Image (Cx256)
            nn.Conv1d(in_channels=n_channels, out_channels=128, kernel_size=10, stride=2, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=0.2),
            
            # State (128x128)
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=0.2),

            # State (256x64)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=0.2),

            # State (512x32)
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=0.2),
            
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(p=0.2),
        )
            # outptut of main module --> State (1024x8)
        
        self.input_last_layer = 1024*8
        
        self.out_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_last_layer, 128)
        )

        self.main_z_module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z_dim, 128)
        )

        self.classification_layer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, z):
        """
            Arguments:
            ---------
                - x (torch.Tensor): well log sequence
                - z (int): Noise vector used to generate x (in case it's coming from the generator) or encoding E(x) from the Encoder (in case it's a real data)
            Return:
            ---------
                - output (torch.Tensor): Output classification (real or false) for each data from the batch 
        """
        x = self.main_x_module(x)
        x = self.out_feature(x)
        z = self.main_z_module(z)
        
        combined_input = torch.cat([x, z], dim=1)
        
        combined_input = self.classification_layer(combined_input)
        return combined_input

    def use_checkpointing(self):
        self.main_x_module = torch.utils.checkpoint(self.main_x_module)
        self.out_feature = torch.utils.checkpoint(self.out_feature)
        self.main_z_module = torch.utils.checkpoint(self.main_z_module)
        self.classification_layer = torch.utils.checkpoint(self.classification_layer)
