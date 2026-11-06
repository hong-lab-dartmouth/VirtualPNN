"""End-to-end check with real weights: load released generators and translate a tile.

Skipped unless VPNN_TEST_WEIGHTS points at a directory laid out as fetch_weights.sh writes it
(<dir>/VirtualPNN_AV6_Fold<k>/latest_net_G.pth). Every checkpoint found is first checked
against its SHA-256 in WEIGHTS_MANIFEST.csv, so a corrupted or substituted file fails here.

    VPNN_TEST_WEIGHTS=weights python -m pytest tests/test_weights_smoke.py -v
"""
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import infer  # noqa: E402

WEIGHTS = os.environ.get("VPNN_TEST_WEIGHTS")
pytestmark = pytest.mark.skipif(not WEIGHTS, reason="set VPNN_TEST_WEIGHTS to run with real weights")

# fetch_weights.sh: archive name prefix -> checkpoint directory prefix
RUN_OF = {"VirtualPNN-P+AV": "VirtualPNN_AV6", "VirtualPNN-P": "VirtualPNN_noAV", "CUT": "CUT",
          "AI-FFPE": "AIFFPE"}


def manifest_sha():
    out = {}
    with open(ROOT / "WEIGHTS_MANIFEST.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] in RUN_OF:
                out[f"{RUN_OF[row['model']]}_Fold{row['fold']}"] = row["sha256"]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def he_like_tile(size=512, seed=0):
    """A synthetic H&E-coloured tile: pink background with darker purple blobs."""
    rng = np.random.default_rng(seed)
    img = np.empty((size, size, 3), dtype=np.float32)
    img[:] = (225, 170, 205)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(40):
        cy, cx, r = rng.integers(0, size, 2).tolist() + [int(rng.integers(4, 12))]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = (110, 60, 150)
    img += rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def test_checkpoints_match_the_manifest():
    want = manifest_sha()
    found = sorted(Path(WEIGHTS).glob("*/latest_net_G.pth"))
    assert found, f"no */latest_net_G.pth under {WEIGHTS}"
    for ckpt in found:
        run = ckpt.parent.name
        if run in want:
            assert sha256(ckpt) == want[run], f"{ckpt} does not match WEIGHTS_MANIFEST.csv"


def test_translate_a_tile_with_the_main_model():
    folds = infer.find_folds(WEIGHTS, None)      # the default run name, VirtualPNN_AV6
    want = manifest_sha()
    for k, p in sorted(folds.items()):
        # the file loaded for fold k is VirtualPNN-P+AV's fold-k generator, byte for byte
        assert p.parent.name == f"VirtualPNN_AV6_Fold{k}", p
        assert sha256(p) == want[f"VirtualPNN_AV6_Fold{k}"], p
    print("main-model checkpoints loaded:", ", ".join(p.parent.name for _, p in sorted(folds.items())))
    nets = [infer.build_generator(p, "cpu") for p in folds.values()]
    img = he_like_tile()
    out = infer.translate(nets, img, tile=256, overlap=0, device="cpu", batch_size=4)
    a = np.asarray(out)
    assert a.shape == (512, 512, 3) and a.dtype == np.uint8
    # Predictions are mostly near-white background.
    assert a.mean() > 128, f"implausibly dark output (mean {a.mean():.1f})"
    with torch.no_grad():
        again = np.asarray(infer.translate(nets, img, 256, 0, "cpu", 4))
    assert np.array_equal(a, again), "inference is not deterministic in eval mode"
