import torch
import torch.nn as nn

class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=0.0):
        super(Conv3D, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, l):
        return self.conv(l)

class Discriminator(nn.Module):
    def __init__(self, voxel_resolution=32):
        super(Discriminator, self).__init__()
        self.conv1 = Conv3D(1, 64)
        self.conv2 = Conv3D(64, 128)
        self.conv3 = Conv3D(128, 256, dropout=0.3)
        self.conv4 = Conv3D(256, 512, dropout=0.3)
        self.final = nn.Sequential(
            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, l):
        l = self.conv1(l)
        l = self.conv2(l)
        l = self.conv3(l)
        l = self.conv4(l)
        l = self.final(l)
        return l.view(-1, 1).squeeze(1)