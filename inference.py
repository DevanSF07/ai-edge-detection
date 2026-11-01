"""
CLI Tool for Single-Image / Batch PCB Inspection & Trace Edge Detection.
Usage:
    python inference.py --input data/samples/sample_bad_12.png --output results/sample_output.png
"""

import os
import argparse
import cv2
import numpy as np

from src.pipeline import PCBInspectionPipeline


def main():
    parser = argparse.ArgumentParser(description="PCB Edge Detection and Restoration Inference")
    parser.add_argument("--input", "-i", required=True, help="Path to input degraded image or directory")
    parser.add_argument("--output", "-o", default="results/output.png", help="Path or directory for output")
    parser.add_argument("--ref", "-r", default=None, help="Optional reference ground-truth image for metrics")
    parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda')")
    args = parser.parse_args()

    pipeline = PCBInspectionPipeline(device=args.device)

    if os.path.isfile(args.input):
        print(f"Loading input: {args.input}")
        img = cv2.imread(args.input)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {args.input}")

        ref_img = cv2.imread(args.ref) if args.ref and os.path.exists(args.ref) else None

        results = pipeline.inspect(img, reference_img=ref_img)

        # Create 3-panel visualization: Degraded, Restored, DexiNed Edges
        H, W = img.shape[:2]
        vis_degraded = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis_restored = cv2.cvtColor(results["restored_image"], cv2.COLOR_GRAY2BGR)
        vis_edges = cv2.cvtColor(results["dexined_edges"], cv2.COLOR_GRAY2BGR)

        combined = np.hstack([vis_degraded, vis_restored, vis_edges])

        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(args.output, combined)
        print(f"Results successfully saved to: {args.output}")

        print("\nInspection Metrics:")
        print(f"  Degraded Sharpness (Laplacian Var): {results['degraded_sharpness']:.2f}")
        print(f"  Restored Sharpness (Laplacian Var): {results['restored_sharpness']:.2f}")
        if "restored_psnr" in results:
            print(f"  Degraded PSNR: {results['degraded_psnr']:.2f} dB -> Restored PSNR: {results['restored_psnr']:.2f} dB")
            print(f"  Degraded SSIM: {results['degraded_ssim']:.4f} -> Restored SSIM: {results['restored_ssim']:.4f}")
            print(f"  DexiNed ODS F1-Score: {results['dexined_ods_f1']:.3f}")

    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
        print(f"Processing {len(files)} images from {args.input}...")

        for f in files:
            img_path = os.path.join(args.input, f)
            img = cv2.imread(img_path)
            if img is None:
                continue
            res = pipeline.inspect(img)
            out_path = os.path.join(args.output, f"inspected_{f}")
            cv2.imwrite(out_path, res["dexined_edges"])
            print(f"  Processed {f} -> {out_path}")
        print("Batch processing complete.")


if __name__ == "__main__":
    main()

