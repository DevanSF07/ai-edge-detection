import cv2
import numpy as np


def sharpen_image(image: np.ndarray, sigma: float = 1.0, alpha: float = 1.5, beta: float = -0.5) -> np.ndarray:
    """
    Unsharp masking filter to restore high-frequency micro-trace edge gradients
    after autoencoder / DnCNN denoising.
        Sharpened = alpha * Image + beta * GaussianBlur(Image)
    Adjusted to bring Laplacian variance to the optimal target range (30-60)
    without amplifying random background noise.
    """
    if image.dtype != np.uint8 and image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
        is_float = True
    else:
        image_uint8 = image.copy()
        is_float = False

    blurred = cv2.GaussianBlur(image_uint8, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(image_uint8, alpha, blurred, beta, 0)

    if is_float:
        return sharpened.astype(np.float32) / 255.0
    return sharpened


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances contrast locally across faint micro-stripes and circuit tracks.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def morphological_trace_enhancement(image: np.ndarray, kernel_size: tuple = (3, 3), op: str = 'tophat') -> np.ndarray:
    """
    OpenCV Morphological transformations to isolate micro-traces from surface grain.
    Supported operations:
        - 'tophat': Isolates elements brighter than surroundings (traces, pads)
        - 'blackhat': Isolates dark anomalies / gaps between tracks
        - 'gradient': Computes morphological edge gradient (Dilation - Erosion)
        - 'close': Fills small discontinuities in faint broken micro-traces
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    if op.lower() == 'tophat':
        return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    elif op.lower() == 'blackhat':
        return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    elif op.lower() == 'gradient':
        return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    elif op.lower() == 'close':
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif op.lower() == 'open':
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    else:
        raise ValueError(f"Unsupported morphological operation: {op}")


def detect_solder_anomalies(image: np.ndarray, min_radius: int = 4, max_radius: int = 35) -> tuple:
    """
    Localises critical solder pads and circular shape anomalies
    using Morphological preprocessing combined with Hough Circle Transform.
    Returns:
        (annotated_image, detected_circles_list)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        annotated = image.copy()
    else:
        gray = image.copy()
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    # Enhance circular structures using morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.medianBlur(morph, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    detected_list = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            detected_list.append((cx, cy, r))
            # Draw circle perimeter (green) and center (red)
            cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), 3)

    return annotated, detected_list

