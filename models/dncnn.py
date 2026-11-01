import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    Deep Denoising Convolutional Neural Network (DnCNN) with Residual Learning.
    Designed to predict residual noise from degraded PCB / industrial images:
        Clean Image = Input Noisy Image - Predicted Noise
    This formulation guarantees superior gradient stability, preserving subtle
    circuit micro-traces and raising PSNR/SSIM significantly.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_layers: int = 17, num_features: int = 64):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1

        layers = []
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Intermediate layers: Conv + BatchNorm + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Final layer: Conv to reconstruct residual noise map
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual learning: predict noise and subtract
        noise = self.dncnn(x)
        return torch.clamp(x - noise, 0.0, 1.0)

