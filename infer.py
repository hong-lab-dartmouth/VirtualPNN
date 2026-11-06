#!/usr/bin/env python3
"""Run the released VirtualPNN fold generators on H&E image files.

The folds are averaged and one PNG is written per input, at the size of the input.

    ./fetch_weights.sh
    python infer.py --input my_slides/ --output virtual_pnn/

  --run-name  released model: VirtualPNN_AV6 (default), VirtualPNN_noAV, CUT, AIFFPE
  --folds     fold numbers to use (default: all found)
  --overlap   overlap in pixels between tiles, blended at the seams (default 0)
  --device    cuda or cpu

The output is an appearance model of aggrecan staining, not a per-cell PNN measure.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import networks

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Run name of the paper's model (VirtualPNN-P+AV); fetch_weights.sh writes its folds as
# weights/VirtualPNN_AV6_Fold<k>/ beside the other released models.
DEFAULT_RUN = "VirtualPNN_AV6"

# Architecture constants of the released generators, matching options/base_options.py.
ARCH = dict(input_nc=3, output_nc=3, ngf=64, netG="resnet_9blocks", norm="instance",
            use_dropout=False, init_type="xavier", init_gain=0.02,
            no_antialias=False, no_antialias_up=False)


def build_generator(weights_path, device):
    """Instantiate the generator and load one checkpoint into it."""
    net = networks.define_G(ARCH["input_nc"], ARCH["output_nc"], ARCH["ngf"], ARCH["netG"],
                            ARCH["norm"], ARCH["use_dropout"], ARCH["init_type"],
                            ARCH["init_gain"], ARCH["no_antialias"], ARCH["no_antialias_up"],
                            gpu_ids=[])
    state = torch.load(str(weights_path), map_location="cpu")
    if hasattr(state, "_metadata"):
        del state._metadata
    # Strip the "module." prefix that DataParallel checkpoints carry.
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    net.load_state_dict(state)
    return net.eval().to(device)


def find_folds(weights_dir, requested, run_name=DEFAULT_RUN):
    """Locate latest_net_G.pth for each requested fold of one model, or every fold present.

    An empty run_name takes every folder. Two folders naming the same fold are an error.
    """
    weights_dir = Path(weights_dir)
    found = {}
    for sub in sorted(weights_dir.iterdir()) if weights_dir.is_dir() else []:
        ckpt = sub / "latest_net_G.pth"
        if ckpt.is_file() and sub.name.startswith(run_name):
            # The fold number is read from the _Fold<k> marker.
            m = re.search(r"[Ff]old[_-]?(\d+)", sub.name) or re.search(r"(\d+)\s*$", sub.name)
            fold = int(m.group(1)) if m else len(found) + 1
            if fold in found:
                sys.exit(f"error: fold {fold} matches both {found[fold].parent.name} and {sub.name} "
                         f"under {weights_dir}; choose one model with --run-name.")
            found[fold] = ckpt
    if requested:
        missing = [f for f in requested if f not in found]
        if missing:
            sys.exit(f"error: no weights for fold(s) {missing} of {run_name or 'any run'} under "
                     f"{weights_dir}. Found: {sorted(found) or 'nothing'}. Run ./fetch_weights.sh first.")
        return {f: found[f] for f in requested}
    if not found:
        sys.exit(f"error: no '{run_name}*/latest_net_G.pth' under {weights_dir}. Run ./fetch_weights.sh, "
                 f"or point --weights at the directory holding the fold subdirectories.")
    return found


def to_tensor(img):
    """PIL RGB -> float tensor in [-1, 1], matching data/base_dataset.get_transform."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1) * 2.0 - 1.0


def to_image(t):
    """Convert a [-1, 1] tensor to an 8-bit RGB image, rounding to the nearest level."""
    a = t.clamp(-1.0, 1.0).cpu().numpy()
    return np.rint((np.transpose(a, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)


def tile_origins(extent, tile, stride):
    """Left/top coordinates covering `extent`, with the last tile flush to the edge."""
    if extent <= tile:
        return [0]
    xs = list(range(0, extent - tile + 1, stride))
    if xs[-1] != extent - tile:
        xs.append(extent - tile)
    return xs


def blend_window(tile, overlap):
    """A 2-D taper so overlapping tiles average smoothly instead of showing seams."""
    if overlap <= 0:
        return np.ones((tile, tile), dtype=np.float32)
    ramp = np.ones(tile, dtype=np.float32)
    edge = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


@torch.no_grad()
def translate(nets, img, tile, overlap, device, batch_size):
    """Run every generator over one image and return the fold-averaged result."""
    w, h = img.size
    pad_r, pad_b = (-w) % tile, (-h) % tile
    arr = np.asarray(img.convert("RGB"))
    if pad_r or pad_b:
        arr = np.pad(arr, ((0, pad_b), (0, pad_r), (0, 0)), mode="reflect")
    padded = Image.fromarray(arr)
    pw, ph = padded.size

    stride = tile - overlap
    xs, ys = tile_origins(pw, tile, stride), tile_origins(ph, tile, stride)
    window = blend_window(tile, overlap)

    acc = np.zeros((3, ph, pw), dtype=np.float32)
    wsum = np.zeros((ph, pw), dtype=np.float32)

    coords = [(x, y) for y in ys for x in xs]
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        batch = torch.stack([to_tensor(padded.crop((x, y, x + tile, y + tile)))
                             for x, y in chunk]).to(device)
        # Average the folds in the network's own [-1, 1] space, before quantisation.
        out = None
        for net in nets:
            pred = net(batch)
            out = pred if out is None else out + pred
        out = (out / len(nets)).cpu().numpy()
        for (x, y), patch in zip(chunk, out):
            acc[:, y:y + tile, x:x + tile] += patch * window
            wsum[y:y + tile, x:x + tile] += window

    merged = acc / np.maximum(wsum, 1e-8)
    return Image.fromarray(to_image(torch.from_numpy(merged))[:h, :w])


def collect_inputs(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        sys.exit(f"error: {path} is neither a file nor a directory")
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        sys.exit(f"error: no images under {path} "
                 f"(looked for {', '.join(sorted(IMAGE_SUFFIXES))})")
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Predict aggrecan-like perineuronal-net staining from H&E.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, help="image file, or a directory of images")
    ap.add_argument("--output", required=True, help="directory for the predicted images")
    ap.add_argument("--weights", default="weights",
                    help="directory holding one subdirectory per fold")
    ap.add_argument("--run-name", default=DEFAULT_RUN,
                    help="model to run, by the run-name prefix of its fold folders: VirtualPNN_AV6 "
                         "(VirtualPNN-P+AV, the paper's model), VirtualPNN_noAV, CUT or AIFFPE")
    ap.add_argument("--folds", type=int, nargs="*", default=None,
                    help="folds to average (default: every fold found)")
    ap.add_argument("--tile", type=int, default=256, help="tile size; the training patch size")
    ap.add_argument("--overlap", type=int, default=0,
                    help="tile overlap in pixels; 0 matches the validated pipeline")
    ap.add_argument("--batch-size", type=int, default=8, help="tiles per forward pass")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--suffix", default="_virtualpnn", help="appended to each output name")
    args = ap.parse_args()

    if args.overlap < 0 or args.overlap >= args.tile:
        sys.exit(f"error: --overlap must be between 0 and {args.tile - 1}")

    folds = find_folds(args.weights, args.folds, args.run_name)
    device = torch.device(args.device)
    print(f"device: {device}")
    nets = []
    for fold, ckpt in sorted(folds.items()):
        print(f"  fold {fold}: {ckpt}")
        nets.append(build_generator(ckpt, device))
    print(f"averaging {len(nets)} generator(s)")

    files = collect_inputs(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} image(s) -> {out_dir}")

    for i, src in enumerate(files, 1):
        img = Image.open(src)
        result = translate(nets, img, args.tile, args.overlap, device, args.batch_size)
        dst = out_dir / f"{src.stem}{args.suffix}.png"
        result.save(dst)
        print(f"  [{i}/{len(files)}] {src.name} {img.size[0]}x{img.size[1]} -> {dst.name}")
    print("done")


if __name__ == "__main__":
    main()
