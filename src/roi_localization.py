import cv2
import numpy as np
import torch


def construct_line_roi(pt1: tuple, pt2: tuple, padding: int = 12) -> tuple:
    """
    Construct a bounding rectangle (ROI) around expected micro-trace line endpoints (x1, y1) and (x2, y2).
    Returns:
        (x, y, w, h)
    """
    x1, y1 = pt1
    x2, y2 = pt2
    xmin = max(0, min(x1, x2) - padding)
    ymin = max(0, min(y1, y2) - padding)
    xmax = max(x1, x2) + padding
    ymax = max(y1, y2) + padding
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def extract_roi_patches(image: np.ndarray, roi_list: list, patch_size: tuple = (64, 64)) -> list:
    """
    Extract square patches from specified ROIs with resizing or center-cropping
    for input into the patch verification classifier.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    patches = []
    H, W = gray.shape

    for (x, y, w, h) in roi_list:
        x_end = min(W, x + w)
        y_end = min(H, y + h)
        x_start = max(0, x)
        y_start = max(0, y)

        crop = gray[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            continue
        patch = cv2.resize(crop, patch_size)
        patches.append(patch)

    return patches


def filter_candidate_rois(
    image: np.ndarray,
    roi_list: list,
    classifier: torch.nn.Module,
    patch_size: tuple = (64, 64),
    device: str = "cpu"
) -> list:
    """
    Evaluates candidate ROI patches using the CNN Line Presence Classifier.
    Returns:
        List of tuples: [(roi, is_line_present, confidence), ...]
    """
    patches = extract_roi_patches(image, roi_list, patch_size)
    if not patches:
        return []

    classifier.eval()
    classifier.to(device)

    results = []
    for roi, patch in zip(roi_list, patches):
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0) / 255.0
        patch_tensor = patch_tensor.to(device)

        with torch.no_grad():
            logits = classifier(patch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            conf = float(probs[pred_class])

        results.append({
            "roi": roi,
            "has_line": bool(pred_class == 1),
            "confidence": conf
        })

    return results


def draw_roi_overlays(image: np.ndarray, roi_results: list) -> np.ndarray:
    """Draw bounding boxes and status labels on the image for inspection visualization."""
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for item in roi_results:
        x, y, w, h = item["roi"]
        has_line = item["has_line"]
        conf = item["confidence"]

        color = (0, 255, 0) if has_line else (0, 0, 255)
        label = f"Trace: {conf:.2f}" if has_line else "No Trace"

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return vis

