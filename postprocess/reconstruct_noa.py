import os
import re
import argparse
import numpy as np
from PIL import Image
from collections import defaultdict

# The 'noa' suffix is historical and carries no meaning; this stitches 256-pixel
# patches back into whole 1536-pixel cores.
def reconstruct_images(patch_dir, output_dir, patch_size=256, image_size=1536):
    os.makedirs(output_dir, exist_ok=True)

    # Group patch filenames by base image prefix
    image_groups = defaultdict(list)
    pattern = re.compile(r"(.+)_y(\d+)_x(\d+)\.png")

    for fname in os.listdir(patch_dir):
        match = pattern.match(fname)
        if match:
            prefix, _, _ = match.groups()
            image_groups[prefix].append(fname)

    print(f"Found {len(image_groups)} unique images.")

    for prefix, patch_list in image_groups.items():
        print(f"Reconstructing: {prefix} ({len(patch_list)} patches)")
        canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
        weight = np.zeros((image_size, image_size, 1), dtype=np.float32)

        for fname in patch_list:
            match = pattern.match(fname)
            if not match:
                continue
            _, y_str, x_str = match.groups()
            y = int(y_str)
            x = int(x_str)

            patch_path = os.path.join(patch_dir, fname)
            patch = Image.open(patch_path).convert("RGB")
            patch_np = np.array(patch).astype(np.float32)

            canvas[y:y+patch_size, x:x+patch_size] += patch_np
            weight[y:y+patch_size, x:x+patch_size] += 1.0

        # Normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            canvas = canvas / weight

        # Fill uncovered with white
        mask = (weight == 0)
        canvas[mask.repeat(3, axis=2)] = 255.0

        # Save
        canvas_img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        output_path = os.path.join(output_dir, f"{prefix}_reconstructed.png")
        canvas_img.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct images from overlapping patches.")
    parser.add_argument("--patch_dir", required=True, help="Directory containing patch PNGs")
    parser.add_argument("--output_dir", required=True, help="Directory to save reconstructed images")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size (default: 256)")
    parser.add_argument("--image_size", type=int, default=1536, help="Full image size (default: 1536)")
    
    args = parser.parse_args()
    reconstruct_images(
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        image_size=args.image_size
    )
