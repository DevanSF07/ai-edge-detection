"""
Core library for AI-assisted Edge Detection and Circuit Board Image Clarification.
Strictly implements the methods described in the CV:
- Restoration (DnCNN, Autoencoders, Unsharp Masking, SSIM/Hybrid Losses)
- Edge Detection (DexiNed Sub-Pixel Multi-Scale Edge Detection)
- Morphological Localization (OpenCV Morphological Transforms across targeted ROIs for traces & solder anomalies)
"""

from .metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_mse,
    calculate_laplacian_variance,
    calculate_ods_f1,
)
from .losses import HybridLoss, ssim_loss
from .enhancement import (
    sharpen_image,
    apply_clahe,
    morphological_trace_enhancement,
    detect_solder_anomalies,
)
from .edge_detectors import (
    canny_edges,
    sobel_edges,
    scharr_edges,
    hough_line_detection,
    run_dexined,
)
from .roi_localization import (
    extract_targeted_rois,
    localize_critical_traces,
    localize_solder_anomalies,
    draw_roi_inspection,
)
from .pipeline import PCBInspectionPipeline

__all__ = [
    "calculate_psnr",
    "calculate_ssim",
    "calculate_mse",
    "calculate_laplacian_variance",
    "calculate_ods_f1",
    "HybridLoss",
    "ssim_loss",
    "sharpen_image",
    "apply_clahe",
    "morphological_trace_enhancement",
    "detect_solder_anomalies",
    "canny_edges",
    "sobel_edges",
    "scharr_edges",
    "hough_line_detection",
    "run_dexined",
    "extract_targeted_rois",
    "localize_critical_traces",
    "localize_solder_anomalies",
    "draw_roi_inspection",
    "PCBInspectionPipeline",
]
