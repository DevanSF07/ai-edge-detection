import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchClassifier(nn.Module):
    """
    CNN-Based Line Presence Classifier for candidate Region of Interest (ROI) patches.
    Filters out background noise and non-trace patches before running fine edge fitment.
    Accepts 32x32 or 64x64 grayscale patches.
    """
    def __init__(self, input_size: int = 64, in_channels: int = 1, num_classes: int = 2):
        super(PatchClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        feature_dim = (input_size // 8) * (input_size // 8) * 128
        self.fc1 = nn.Linear(feature_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

