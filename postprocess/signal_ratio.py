"""
Per reconstructed 1536-pixel core, inside the circle of radius 764 pixels:
- intensity: mean grayscale value
- signal_ratio: fraction of grayscale pixels below 245 (non-white), the paper's signal metric
Writes one line per core and the means over the folder to --out_file. A lower signal ratio on
vessel-only or tumor cores means less off-target staining.
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
# Command-line arguments
parser = argparse.ArgumentParser(description="Compute the stained-pixel signal ratio inside a circular tissue mask (tumor or vessel cores; choose which via --patch_folder).")
parser.add_argument(
    "--patch_folder",
    type=str,
    required=True,
    help="Path to folder containing .png patch images (masked circular images)."
)
parser.add_argument(
    "--out_file",
    type=str,
    default="signal_ratio_circlemask.txt",
    help="Output text file name (default: signal_ratio_circlemask.txt)."
)
args = parser.parse_args()
PATCH_FOLDER = Path(args.patch_folder)
OUT_FILE = args.out_file
# Circular tissue mask
IMAGE_SIZE = 1536
CIRCLE_RADIUS = IMAGE_SIZE // 2 - 4
Y, X = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
CENTER = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
CIRCULAR_MASK = (np.sqrt((X - CENTER[1])**2 + (Y - CENTER[0])**2) <= CIRCLE_RADIUS)

def process_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Apply circular tissue mask
    img_masked = img[CIRCULAR_MASK]

    mean_intensity = np.mean(img_masked)
    signal_ratio = np.mean(img_masked < 245)

    return image_path.name, mean_intensity, signal_ratio


def main():
    images = sorted(PATCH_FOLDER.glob("*.png"))
    results = [process_image(p) for p in images if process_image(p) is not None]

    intensities = [r[1] for r in results]
    signal_ratios = [r[2] for r in results]

    mean_intensity = np.mean(intensities)
    mean_signal_ratio = np.mean(signal_ratios)

    with open(OUT_FILE, "w") as f:
        for fname, intensity, ratio  in results:
            f.write(f"{fname}: intensity={intensity:.2f}, signal_ratio={ratio:.3f}\n")
        f.write(f"\nMean intensity: {mean_intensity:.2f}\n")
        f.write(f"Mean signal ratio (within circular mask): {mean_signal_ratio:.3f}")

    print(f"Saved masked signal-ratio metrics to {OUT_FILE}")


if __name__ == "__main__":
    main()
