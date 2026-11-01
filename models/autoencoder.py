import torch
import torch.nn as nn


class LineAutoencoder(nn.Module):
    """
    Skip-connection Denoising Autoencoder tailored for micro-circuit trace recovery.
    Uses U-Net style skip concatenations between encoder and decoder to retain
    high-frequency spatial boundaries and faint micro-stripes while eliminating noise.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(LineAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/4, W/4

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # H/2, W/2
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32 (up1) + 32 (pool1)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # H, W
        self.dec2 = nn.Sequential(
            nn.Conv2d(48, out_channels, kernel_size=3, padding=1),  # 16 (up2) + 32 (enc1)
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)         # [B, 32, H, W]
        p1 = self.pool1(x1)       # [B, 32, H/2, W/2]
        x2 = self.enc2(p1)        # [B, 64, H/2, W/2]
        p2 = self.pool2(x2)       # [B, 64, H/4, W/4]

        # Decoder with skip connection cat1
        up_x1 = self.up1(p2)
        cat1 = torch.cat([up_x1, p1], dim=1)  # [B, 64, H/2, W/2]
        dec_x1 = self.dec1(cat1)              # [B, 32, H/2, W/2]

        # Decoder with skip connection cat2
        up_x2 = self.up2(dec_x1)
        cat2 = torch.cat([up_x2, x1], dim=1)  # [B, 48, H, W]
        output = self.dec2(cat2)              # [B, out_channels, H, W]
        return output

