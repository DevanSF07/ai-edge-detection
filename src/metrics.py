import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Mean Squared Error between two images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB.
    Higher values indicate better restoration quality.
    """
    mse = calculate_mse(img1, img2)
    if mse == 0.0:
        return float('inf')
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index Measure (SSIM) between two images.
    Captures structural, luminance, and contrast consistency (0 to 1).
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if img1.ndim == 3 and img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3 and img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, _ = ssim_fn(img1, img2, full=True, data_range=255)
    return float(score)


def calculate_laplacian_variance(img: np.ndarray) -> float:
    """
    Quantify image edge sharpness using Laplacian Variance.
    Metric used by Duravision inspection systems:
        Sharpness = Var(Laplacian(Image))
    Helps detect if denoising autoencoders have over-smoothened critical faint traces.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(laplacian.var())


def calculate_ods_f1(pred_edges: np.ndarray, gt_edges: np.ndarray, threshold: float = 0.35, max_dist: int = 2) -> float:
    """
    Calculate Optimal Dataset Scale (ODS) F1-score for edge detection.
    Adheres to the standard BSDS / DexiNed evaluation benchmark protocol with
    sub-pixel boundary tolerance (max_dist pixel matching margin):
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * Precision * Recall / (Precision + Recall)
    """
    if pred_edges.shape != gt_edges.shape:
        pred_edges = cv2.resize(pred_edges, (gt_edges.shape[1], gt_edges.shape[0]))

    pred_bin = (pred_edges >= (threshold * 255 if pred_edges.max() > 1 else threshold)).astype(np.uint8)
    gt_bin = (gt_edges >= (threshold * 255 if gt_edges.max() > 1 else threshold)).astype(np.uint8)

    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0

    # Tolerance margin via morphological dilation (BSDS500 standard)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * max_dist + 1, 2 * max_dist + 1))
    gt_dilated = cv2.dilate(gt_bin, kernel)
    pred_dilated = cv2.dilate(pred_bin, kernel)

    tp_pred = np.logical_and(pred_bin, gt_dilated).sum()
    tp_gt = np.logical_and(gt_bin, pred_dilated).sum()

    precision = float(tp_pred / (pred_bin.sum() + 1e-8))
    recall = float(tp_gt / (gt_bin.sum() + 1e-8))
    f1 = 2.0 * (precision * recall) / (precision + recall + 1e-8)
    return float(f1)

