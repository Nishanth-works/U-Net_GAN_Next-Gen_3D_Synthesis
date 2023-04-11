import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)

        # Decoder
        self.dec1 = UNetBlock(512, 256)
        self.dec2 = UNetBlock(256, 128)
        self.dec3 = UNetBlock(128, 64)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Decode
        dec1 = self.dec1(self.up(enc4))
        dec1 = torch.cat([enc3, dec1], 1)
        dec2 = self.dec2(self.up(dec1))
        dec2 = torch.cat([enc2, dec2], 1)
        dec3 = self.dec3(self.up(dec2))
        dec3 = torch.cat([enc1, dec3], 1)

        # Output
        output = self.final(dec3)
        return output