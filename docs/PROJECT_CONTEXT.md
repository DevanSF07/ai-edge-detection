# Project Context: AI-Assisted Edge Detection for Industrial PCB Inspection

## 1. Executive Summary & Client Context
* **Organization & Client Collaboration**: Developed for **Duravision Systems** in collaboration with **Guangdong Longtian Smart Instruments Co. Ltd.** (represented by Project Lead Kishor Durve).
* **Industrial Problem**: Automatic Optical Inspection (AOI) systems for Printed Circuit Boards (PCBs) must detect microscopic copper traces, vertical stripe boundaries, and solder joint circles. 
* **The Failure of Conventional Image Processing**:
  * Traditional edge detection methods (Canny, Sobel, Scharr, Hough Transform) fail when imaging conditions introduce high sensor grain, low illumination, or degraded contrast.
  * In degraded ("Bad Images"), micro-traces fade into the background noise, causing standard gradient operators to produce either completely fragmented edges or severe false-positive clutter.

---

## 2. Technical Milestones & Pipeline Evolution

### Stage 1: Classical Baseline & ROI Definition (NB-1 & NB-2)
* Tested standard OpenCV Canny, Sobel, and Hough Transform baselines.
* Formulated a **Region of Interest (ROI)** framework: because trace endpoints are approximately known in advance $(x_1, y_1)$ to $(x_2, y_2)$, rectangular bounding windows can constrain the search space.
* Built CNN patch classifiers ($32 \times 32$ and $64 \times 64$) to verify trace presence and eliminate false positives from background clutter.

### Stage 2: Super-Resolution & Denoising Exploration (BSRGAN / RRDBNet)
* Integrated an external deep super-resolution model (**RRDBNet / BasicSR** weights `Model.pth`, 67 MB) provided by Chinese academic partners.
* Deployed a desktop graphical interface (`app/SR_GUI.py`) for live operator inspection.
* Key finding: While 4x super-resolution effectively filtered noise, it introduced computational overhead (~10s/image on CPU), necessitating dedicated lightweight convolutional restoration.

### Stage 3: Autoencoder & DnCNN Image Restoration (NB-5 & Updated DnCNN)
* Engineered a **Denoising Autoencoder with skip connections** and **DnCNN** (Deep Convolutional Neural Network for Image Denoising).
* **Residual Learning in DnCNN**: Instead of directly outputting clean pixels, DnCNN predicts the residual noise map $\mathcal{R}(y) \approx v$, yielding $\hat{x} = y - \mathcal{R}(y)$.
* **Hybrid Loss Formulation**: Standard Mean Squared Error (MSE) tends to over-smooth fine edges. To preserve structural integrity and gradient contrast, a hybrid loss was enforced:
  $$\mathcal{L}_{\text{hybrid}} = 0.5 \cdot \mathcal{L}_{\text{MSE}} + 0.5 \cdot (1 - \text{SSIM})$$
* **Result**: Restored degraded images from **26.8 dB to 36.4 dB PSNR** and raised **SSIM from 0.74 to 0.94**.

### Stage 4: Sharpness Calibration via Laplacian Variance
* Denoising models occasionally blur high-frequency micro-trace transitions, causing the image to appear "over-smoothed".
* To monitor and counter this, **Laplacian Variance** was integrated as an automated quality gate:
  $$\text{Sharpness} = \operatorname{Var}(\nabla^2 I) = \operatorname{Var}\left( \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2} \right)$$
* An adaptive unsharp masking filter ($\alpha=1.5, \beta=-0.5, \sigma=1.0$) was introduced immediately after autoencoder reconstruction, bringing sharpness back into the ideal range (35–55) without re-amplifying background noise.

### Stage 5: Deep Sub-Pixel Edge Detection (DexiNed) & Morphological Localization
* Fine-tuned **DexiNed (Dense Extreme Inception Network)** across micro-circuit traces.
* DexiNed's dense multi-scale extreme inception blocks fuse edge representations across 4 scales, resolving faint sub-pixel trace boundaries that classical operators miss.
* Attained **0.88 ODS F1-Score** on micro-trace detection with an average inference speed of **4.61 seconds per image**.
* Applied OpenCV morphological transformations (Top-Hat, Gradient, Closing) and Hough Circle transforms across targeted ROIs to reliably pinpoint solder anomalies and trace gaps.

---

## 3. Placement CV Claims Verification Table

| CV Claim | Implementation in Repository | Supporting Evidence & Metrics |
| :--- | :--- | :--- |
| **"Restored degraded PCB surface images using DnCNN and Autoencoders"** | `models/autoencoder.py`<br>`models/dncnn.py`<br>`src/losses.py` | PyTorch U-Net skip autoencoder and DnCNN with residual learning and Hybrid SSIM+MSE loss. |
| **"raising PSNR score from 26.8 dB to 36.4 dB and SSIM from 0.74 to 0.94"** | `src/metrics.py`<br>`results/benchmark_results.csv` | Evaluated across 20 degraded PCB boards (`data/raw/bad_images/` vs `data/processed/corrected_bad/`). |
| **"Fine-tuned DexiNed (Dense Extreme Inception Network) on micro-circuit traces, capturing faint sub-pixel trace edges with 0.88 ODS F1-score"** | `models/dexined.py`<br>`src/edge_detectors.py` | 4-scale multi-level extreme inception fusion predicting continuous sub-pixel trace boundaries at 0.88 ODS F1. |
| **"Localised critical circuit traces and solder anomalies by integrating OpenCV morphological transforms across targeted Regions of Interest"** | `src/roi_localization.py`<br>`src/enhancement.py` | Top-hat / black-hat trace extraction, Hough circle solder anomaly detection, and candidate ROI patch classification. |

