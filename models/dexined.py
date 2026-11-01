import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _DenseExtremeInceptionBlock(nn.Module):
    """
    Sub-block for DexiNed containing concatenated residual convolutional passes
    to extract multi-scale micro-features and sub-pixel edge responses.
    """
    def __init__(self, in_features, out_features):
        super(_DenseExtremeInceptionBlock, self).__init__()
        self.conv1 = conv3x3(in_features, out_features)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_features, out_features)
        self.bn2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if residual.shape == out.shape:
            out += residual
        return self.relu(out)


class DexiNed(nn.Module):
    """
    Dense Extreme Inception Network (DexiNed) for fine sub-pixel trace edge detection.
    Extracts deep, multi-scale edge maps and fuses them to resolve low-contrast,
    faint micro-traces without heavy hand-crafted parameter tuning.
    """
    def __init__(self, in_channels: int = 1):
        super(DexiNed, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1_b1 = _DenseExtremeInceptionBlock(32, 32)
        self.side1 = nn.Conv2d(32, 1, kernel_size=1)

        # Block 2
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block2_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.block2_b1 = _DenseExtremeInceptionBlock(64, 64)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)

        # Block 3
        self.maxpool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block3_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.block3_b1 = _DenseExtremeInceptionBlock(128, 128)
        self.side3 = nn.Conv2d(128, 1, kernel_size=1)

        # Block 4
        self.maxpool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block4_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.block4_b1 = _DenseExtremeInceptionBlock(256, 256)
        self.side4 = nn.Conv2d(256, 1, kernel_size=1)

        # Fusion layer (combining 4 multi-scale side outputs)
        self.fusion = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]

        # Block 1
        x1 = F.relu(self.block1_conv1(x))
        x1 = self.block1_b1(x1)
        s1 = self.side1(x1)

        # Block 2
        x2 = self.maxpool1(x1)
        x2 = F.relu(self.block2_conv1(x2))
        x2 = self.block2_b1(x2)
        s2 = F.interpolate(self.side2(x2), size=(H, W), mode='bilinear', align_corners=False)

        # Block 3
        x3 = self.maxpool2(x2)
        x3 = F.relu(self.block3_conv1(x3))
        x3 = self.block3_b1(x3)
        s3 = F.interpolate(self.side3(x3), size=(H, W), mode='bilinear', align_corners=False)

        # Block 4
        x4 = self.maxpool3(x3)
        x4 = F.relu(self.block4_conv1(x4))
        x4 = self.block4_b1(x4)
        s4 = F.interpolate(self.side4(x4), size=(H, W), mode='bilinear', align_corners=False)

        # Fused Edge Map
        fused = self.fusion(torch.cat([s1, s2, s3, s4], dim=1))

        # Apply sigmoid to all edge maps
        outputs = [torch.sigmoid(s) for s in [s1, s2, s3, s4]]
        outputs.append(torch.sigmoid(fused))

        return outputs  # [side1, side2, side3, side4, fused]

