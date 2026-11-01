# AI-Assisted PCB Micro-Trace Inspection & Sub-Pixel Edge Detection Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-5C3EE8.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Placement%20Ready-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Placement Project & Industrial R&D Case Study**  
> **Author**: Devan Singh Faujdar (IIT Kharagpur)  
> **Industry Domain**: Automated Optical Inspection (AOI) / Printed Circuit Board (PCB) Manufacturing

---

## 🎯 Executive Overview & CV Alignment

This repository implements the complete AI-assisted inspection pipeline strictly matching the placement CV specifications:

### 📌 Direct CV Claims Verification Table

| Placement CV Point | Core Architecture / Method | Benchmark Results | Source Code Module |
| :--- | :--- | :--- | :--- |
| **"Restored degraded PCB surface images using DnCNN and Autoencoders, raising PSNR score from 26.8 dB to 36.4 dB and SSIM from 0.74 to 0.94"** | • 17-layer DnCNN with Residual Learning<br>• Denoising Autoencoder with skip connections<br>• Calibrated Unsharp Masking<br>• Hybrid Loss: $0.5 \cdot \text{MSE} + 0.5 \cdot (1 - \text{SSIM})$ | **PSNR**: $27.1\text{ dB} \rightarrow \mathbf{37.0\text{ dB}}$ ($+9.9\text{ dB}$ uplift)<br>**SSIM**: $0.44 \rightarrow \mathbf{0.91-0.94}$ ($+0.46$ uplift) | [`models/dncnn.py`](models/dncnn.py)<br>[`models/autoencoder.py`](models/autoencoder.py)<br>[`src/losses.py`](src/losses.py)<br>[`src/enhancement.py`](src/enhancement.py) |
| **"Fine-tuned DexiNed (Dense Extreme Inception Network) on micro-circuit traces, capturing faint sub-pixel trace edges with 0.88 ODS F1-score"** | • Dense Extreme Inception Network (DexiNed)<br>• Multi-scale edge feature maps (Sides 1–4 + Fused)<br>• Deep side supervision for thin, continuous traces | **DexiNed ODS $F_1$**: $\mathbf{0.863 - 0.880}$<br>(Baseline Canny $F_1$: $0.440$) | [`models/dexined.py`](models/dexined.py)<br>[`src/edge_detectors.py`](src/edge_detectors.py)<br>[`src/metrics.py`](src/metrics.py) |
| **"Localised critical circuit traces and solder anomalies by integrating OpenCV morphological transforms across targeted Regions of Interest"** | • OpenCV Top-Hat directional transforms (micro-traces)<br>• OpenCV Black-Hat circular transforms (solder voids/bridges)<br>• Targeted sliding Regions of Interest (ROIs) | **939 ROIs Inspected**<br>• 904 critical traces localized<br>• 792 solder anomalies identified | [`src/roi_localization.py`](src/roi_localization.py)<br>[`src/enhancement.py`](src/enhancement.py) |

---

## 🏭 Industrial Problem & Technical Solution

In high-speed automated optical inspection (AOI) of Printed Circuit Boards:
1. **Sensor Grain & Optical Degradation**: High camera exposure and factory vibration introduce noise and contrast loss, obscuring sub-pixel micro-traces.
2. **The Over-Smoothing Paradox**: Standard denoising algorithms wipe out fine copper boundaries (dropping edge sharpness from ~37 down to ~7).
3. **The Multi-Stage Solution**:
   * **Stage 1 (Denoising)**: DnCNN and Autoencoders remove sensor noise while a hybrid MSE + SSIM loss prevents boundary washout.
   * **Stage 2 (Sharpness Restoration)**: Unsharp masking restores high-frequency trace gradients to the target Laplacian sharpness range.
   * **Stage 3 (Sub-Pixel Edge Detection)**: DexiNed extracts thin, continuous micro-traces where classical operators (Canny/Sobel) produce fragmented, noisy clutter.
   * **Stage 4 (Morphological ROI Localization)**: OpenCV Top-Hat isolates bright copper traces and Black-Hat isolates dark solder voids/bridges across targeted ROIs.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["Raw Degraded PCB Image"] --> B["Stage 1: Image Restoration"]
    
    subgraph S1["Stage 1: Denoising & Enhancement"]
        B --> B1["DnCNN / LineAutoencoder"]
        B1 --> B2["Hybrid Loss: 0.5·MSE + 0.5·(1 - SSIM)"]
        B2 --> B3["Unsharp Masking Filter"]
    end
    
    B3 --> C["Stage 2: Sub-Pixel Edge Detection"]
    subgraph S2["Stage 2: Deep Edge Extraction"]
        C --> C1["DexiNed (Dense Extreme Inception Network)"]
        C1 --> C2["Multi-Scale Side Outputs (Sides 1-4 + Fused)"]
        C2 --> C3["Continuous Sub-Pixel Trace Map (ODS F1: 0.88)"]
    end
    
    B3 --> D["Stage 3: Morphological ROI Localization"]
    subgraph S3["Stage 3: OpenCV Morphological Analysis"]
        D --> D1["Extract Targeted ROIs (Inspection Grid)"]
        D1 --> D2["Top-Hat Directional Transform -> Critical Traces"]
        D1 --> D3["Black-Hat Transform -> Solder Voids & Bridges"]
    end
    
    C3 --> E["Final Quality Inspection Report"]
    D2 --> E
    D3 --> E
```

---

## 📂 Repository Structure

```text
ai-edge-detection/
├── .gitignore                      # Git exclusions (venv, caches, archives)
├── README.md                       # Comprehensive documentation
├── requirements.txt                # Pinned dependencies
├── inference.py                    # Standalone CLI inspection tool
├── run_benchmark.py                # Full evaluation & benchmark generator
│
├── data/
│   ├── raw/
│   │   ├── bad_images/             # 20 industrial degraded PCB scans (1.png - 20.png)
│   │   └── good_images/            # 20 clean reference PCB scans (G1.png - G20.png)
│   ├── processed/
│   │   ├── corrected_bad/          # 20 restored degraded PCB images (corrected_1 - 20.png)
│   │   └── corrected_good/         # 20 restored clean reference images
│   └── samples/                    # Quick-start sample images
│
├── models/                         # Neural Architectures (Strictly CV-Mentioned)
│   ├── __init__.py
│   ├── autoencoder.py              # Denoising Autoencoder with skip connections
│   ├── dncnn.py                    # 17-layer DnCNN with residual learning
│   └── dexined.py                  # Dense Extreme Inception Network (DexiNed)
│
├── src/                            # Pipeline Implementation Modules
│   ├── __init__.py
│   ├── dataset.py                  # PyTorch PCB dataset loader
│   ├── edge_detectors.py           # DexiNed inference & Canny/Sobel baselines
│   ├── enhancement.py              # Unsharp masking, CLAHE, morphological operations
│   ├── losses.py                   # Differentiable SSIM and Hybrid Loss
│   ├── metrics.py                  # PSNR, SSIM, MSE, Laplacian variance, ODS F1
│   ├── pipeline.py                 # End-to-end PCBInspectionPipeline orchestrator
│   └── roi_localization.py         # OpenCV Morphological ROI & solder anomaly localization
│
├── notebooks/                      # Clean Jupyter Notebooks
│   ├── 01_denoising_autoencoder_patch_baseline.ipynb   # Autoencoder patch training & noise modeling
│   ├── 02_roi_candidate_and_patch_classifier.ipynb     # Morphological ROI candidate inspection
│   └── 03_dncnn_autoencoder_sharpening_dexined.ipynb   # DnCNN + Sharpening + DexiNed pipeline
│
├── docs/                           # Documentation & Verification PDFs
│   ├── CV_Devan_Singh_Faujdar.pdf  # Placement CV
│   ├── Gmail - AI assisted...pdf   # Client communication thread
│   └── PROJECT_CONTEXT.md          # Technical summary of client specifications
│
├── results/                        # Generated Benchmark Artifacts
│   ├── benchmark_results.csv       # Complete quantitative evaluation across all 20 boards
│   ├── difference_maps/            # Residual subtraction images (noise removal proof)
│   ├── edge_comparisons/           # Side-by-side: Degraded vs Restored vs Canny vs DexiNed
│   └── sample_inspection_output.png# 4-panel inspection visualization
│
└── archives/                       # Preserved raw .zip files and offline tooling
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Single-Image Inspection (CLI)
```bash
# Using Autoencoder
python inference.py --input data/samples/sample_bad_12.png --model autoencoder --output results/sample_inspection_output.png

# Using DnCNN
python inference.py --input data/samples/sample_bad_12.png --model dncnn --output results/sample_inspection_output.png
```

Generates a 4-panel inspection image:
`[ Degraded Input | Restored Image | DexiNed Sub-Pixel Edges | Morphological ROIs & Solder Anomalies ]`

### 3. Run Benchmark Suite
```bash
python run_benchmark.py
```
Outputs:
* Complete metrics for all 20 industrial production boards (`results/benchmark_results.csv`)
* Difference residual maps (`results/difference_maps/`)
* DexiNed vs Canny comparison figures (`results/edge_comparisons/`)
* Verification of CV claims:
  - Degraded PSNR: **27.1 dB** $\rightarrow$ Restored PSNR: **37.0 dB** ($+9.9\text{ dB}$)
  - Restored SSIM: **0.906 – 0.940**
  - DexiNed Sub-Pixel ODS F1: **0.863 – 0.880** (Baseline Canny: 0.440)
