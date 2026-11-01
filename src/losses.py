import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int = 1) -> torch.Tensor:
    _1d_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, channel: int = 1) -> torch.Tensor:
    """
    Differentiable Structural Similarity (SSIM) Loss in PyTorch.
    Computes 1 - SSIM.
    """
    window = _create_window(window_size, channel).to(img1.device)
    padd = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return 1.0 - ssim_map.mean()


class HybridLoss(nn.Module):
    """
    Hybrid Loss Function combining Mean Squared Error (MSE) and Structural Similarity (SSIM):
        Loss = alpha * MSE + (1 - alpha) * (1 - SSIM)
    Preserves pixel-level trace accuracy while enforcing structural coherence and contrast,
    preventing edge washouts on micro-traces.
    """
    def __init__(self, alpha: float = 0.5, window_size: int = 11):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_mse = self.mse(pred, target)
        loss_ssim = ssim_loss(pred, target, window_size=self.window_size, channel=pred.shape[1])
        return self.alpha * loss_mse + (1.0 - self.alpha) * loss_ssim

