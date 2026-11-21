"""Build the feature bank fit.py reads, from the 96-pixel H&E crops of the annotated neurons.

The bank holds the Phikon-v2 feature of every crop. The paper's classifier uses these features alone. With
--copies 5 the bank also holds the features of five augmented copies of each crop (augment.py), the paper's
sensitivity analysis. Crops must be in cells.csv order.

    python build_bank.py --crops crops.npy --out bank.npz [--device cuda] [--copies 0] [--seed 0]

crops.npy: uint8, shape (n, 96, 96, 3), 0.4007 µm per pixel, each centred on an annotated neuron
(features.resample and features.crops make them from an image and the neurons' coordinates).
"""
import argparse

import numpy as np

import augment
import features


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--copies", type=int, default=0, help="augmented copies per crop (0, the paper's classifier; 5, its sensitivity analysis)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    crops = np.load(a.crops)
    assert crops.dtype == np.uint8 and crops.shape[1:] == (features.CROP_PX, features.CROP_PX, 3), crops.shape
    n = len(crops)
    proc, model = features.load_encoder(a.device)
    plain = features.embed(crops, proc, model)
    k = a.copies
    copies = features.embed(augment.bank(crops, k=k, seed=a.seed), proc, model) if k else np.zeros((0, plain.shape[1]), np.float32)
    np.savez_compressed(a.out, X=np.vstack([plain, copies]).astype(np.float32),
                        cell=np.concatenate([np.arange(n), np.arange(n * k) // max(k, 1)]).astype(np.int32),
                        augmented=np.concatenate([np.zeros(n, bool), np.ones(n * k, bool)]))
    print(f"{n} crops, {n * k} augmented copies; wrote {a.out}")


if __name__ == "__main__":
    main()
