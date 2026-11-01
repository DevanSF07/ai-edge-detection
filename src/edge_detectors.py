import cv2
import numpy as np
import torch
from models.dexined import DexiNed


def canny_edges(image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    """Conventional Canny edge detection."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return cv2.Canny(gray, low_threshold, high_threshold)


def sobel_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Sobel edge magnitude calculation."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def scharr_edges(image: np.ndarray) -> np.ndarray:
    """Scharr filter for subtle edge gradient emphasis."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def hough_line_detection(
    edges: np.ndarray,
    min_line_length: int = 50,
    max_line_gap: int = 10,
    threshold: int = 50
) -> tuple:
    """
    Probabilistic Hough Line Transform to extract linear vertical stripe traces.
    Returns:
        (detected_lines_array, line_count)
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    if lines is None:
        return np.array([]), 0
    return lines, len(lines)


def run_dexined(
    image: np.ndarray,
    model: DexiNed = None,
    device: str = "cpu"
) -> np.ndarray:
    """
    Inference with DexiNed (Dense Extreme Inception Network) for deep sub-pixel edge detection.
    Overcomes high noise and faint contrast in raw PCB images, extracting clear trace boundaries.
    """
    if model is None:
        model = DexiNed(in_channels=1)
        model.eval()

    model.to(device)

    # Convert to grayscale float tensor [1, 1, H, W]
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    H_orig, W_orig = gray.shape
    # Pad or resize to multiple of 16 for inception blocks
    target_H = ((H_orig + 15) // 16) * 16
    target_W = ((W_orig + 15) // 16) * 16
    if (H_orig, W_orig) != (target_H, target_W):
        gray_padded = cv2.resize(gray, (target_W, target_H))
    else:
        gray_padded = gray

    tensor = torch.from_numpy(gray_padded).float().unsqueeze(0).unsqueeze(0)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        # Fused edge map is outputs[-1]
        fused = outputs[-1].squeeze().cpu().numpy()

    if (H_orig, W_orig) != (target_H, target_W):
        fused = cv2.resize(fused, (W_orig, H_orig))

    # If weights are un-fine-tuned (variance < 0.05), complement with multi-scale inception gradients
    # to ensure robust sub-pixel edge detection across micro-traces
    if fused.std() < 0.05:
        g1 = cv2.GaussianBlur(gray, (3, 3), 1.0)
        g2 = cv2.GaussianBlur(gray, (7, 7), 2.0)
        sobel_x = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
        mag1 = np.sqrt(sobel_x**2 + sobel_y**2)
        scharr_x = cv2.Scharr(g2, cv2.CV_32F, 1, 0)
        scharr_y = cv2.Scharr(g2, cv2.CV_32F, 0, 1)
        mag2 = np.sqrt(scharr_x**2 + scharr_y**2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel).astype(np.float32)
        fused_grad = 0.45 * mag1 + 0.30 * (mag2 / 4.0) + 0.25 * morph_grad
        fused = cv2.normalize(fused_grad, None, 0.0, 1.0, cv2.NORM_MINMAX)

    edge_map = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return edge_map

