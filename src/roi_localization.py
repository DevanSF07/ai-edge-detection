"""
ROI Localization and Solder Anomaly Detection via OpenCV Morphological Transforms.
Directly implements CV bullet point 3:
"Localised critical circuit traces and solder anomalies by integrating OpenCV
morphological transforms across targeted Regions of Interest."
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


def extract_targeted_rois(
    image: np.ndarray,
    patch_size: int = 64,
    stride: int = 64
) -> List[Tuple[int, int, int, int]]:
    """
    Extracts targeted Regions of Interest (ROIs) across the PCB surface
    using a sliding window inspection grid.
    Returns:
        List of bounding boxes: [(x, y, w, h), ...]
    """
    if image.ndim == 3:
        H, W = image.shape[:2]
    else:
        H, W = image.shape

    rois = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            rois.append((x, y, patch_size, patch_size))

    return rois


def localize_critical_traces(
    image: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    threshold_val: Optional[int] = None,
    min_density: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Localises critical micro-circuit traces inside targeted ROIs using
    OpenCV Top-Hat transform with directional structuring elements (vertical and horizontal).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Directional kernels for linear micro-traces
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))

    # Top-hat isolates bright copper traces from substrate
    tophat_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_vert)
    tophat_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_horiz)
    traces_map = cv2.add(tophat_v, tophat_h)

    # Adaptive threshold if not explicitly specified
    if threshold_val is None:
        threshold_val = max(6, int(np.mean(traces_map) + 0.8 * np.std(traces_map)))

    results = []
    for (x, y, w, h) in rois:
        patch = traces_map[y:y+h, x:x+w]
        if patch.size == 0:
            continue

        # Measure trace presence density inside the ROI
        trace_density = float(np.count_nonzero(patch > threshold_val) / patch.size)
        is_trace = trace_density > min_density

        results.append({
            "roi": (x, y, w, h),
            "is_critical_trace": is_trace,
            "trace_density": round(trace_density, 4),
        })

    return results


def localize_solder_anomalies(
    image: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    threshold_val: Optional[int] = None,
    min_density: float = 0.03
) -> List[Dict[str, Any]]:
    """
    Localises solder anomalies (bridges, voids, discontinuities, and pinholes)
    inside targeted ROIs using OpenCV Black-Hat transform and Morphological Closing.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Black-hat transform highlights dark anomalies, voids, and gaps in solder joints
    kernel_circ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_circ)

    # Adaptive threshold if not explicitly specified
    if threshold_val is None:
        threshold_val = max(5, int(np.mean(blackhat) + 1.2 * np.std(blackhat)))

    anomalies = []
    for (x, y, w, h) in rois:
        patch_bh = blackhat[y:y+h, x:x+w]
        if patch_bh.size == 0:
            continue

        void_ratio = float(np.count_nonzero(patch_bh > threshold_val) / patch_bh.size)
        is_anomaly = void_ratio > min_density

        anomalies.append({
            "roi": (x, y, w, h),
            "has_anomaly": is_anomaly,
            "void_ratio": round(void_ratio, 4),
            "defect_type": "solder_void/bridge" if is_anomaly else "normal"
        })

    return anomalies


def draw_roi_inspection(
    image: np.ndarray,
    trace_results: List[Dict[str, Any]],
    anomaly_results: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Renders an inspection overlay:
      - Green boxes: Localised critical circuit traces
      - Red boxes: Localised solder anomalies (voids, bridges, irregularities)
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Draw circuit traces in green
    for item in trace_results:
        if item["is_critical_trace"]:
            x, y, w, h = item["roi"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Draw solder anomalies in red (thicker border with label)
    for item in anomaly_results:
        if item["has_anomaly"]:
            x, y, w, h = item["roi"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, "ANOMALY", (x, y + h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    return vis
