"""Score neurons with the released weights of the reference classifier.

Input is either features already computed with features.embed (--features, .npy of shape (n, 2048)) or 96-pixel H&E
crops (--crops, .npy of shape (n, 96, 96, 3), uint8, in the 0.4007 µm-per-pixel frame), which are embedded here.
The output is the probability that each neuron is PNN-positive: the mean over the five fold models of one design,
or one fold with --fold. The paper scored the one patient that no generator fold trained on in the same way, by
the mean of the five patient-grouped models.

Every input must be an H&E crop centred on a neuron. The models were trained on annotated neurons only.

    python predict.py --features features.npy --out scores.csv
    python predict.py --crops crops.npy --device cuda --out scores.csv
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DESIGNS = ("patient_grouped", "core_split_patient_clean", "core_split_core_held_out")


def probabilities(X: np.ndarray, weights, design: str = "patient_grouped", fold=None) -> np.ndarray:
    """P(PNN-positive) for features X (n, 2048): mean over the five fold models, or fold `fold` (1 to 5)."""
    mean, scale = weights[f"{design}__mean"], weights[f"{design}__scale"]
    coef, b = weights[f"{design}__coef"], weights[f"{design}__intercept"]
    ks = range(5) if fold is None else [int(fold) - 1]
    p = [1.0 / (1.0 + np.exp(-(((X - mean[k]) / scale[k]) @ coef[k] + b[k]))) for k in ks]
    return np.mean(p, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--features")
    g.add_argument("--crops")
    ap.add_argument("--weights", default=str(HERE / "weights.npz"))
    ap.add_argument("--design", default="patient_grouped", choices=DESIGNS)
    ap.add_argument("--fold", type=int, choices=range(1, 6))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.features:
        X = np.load(a.features).astype(np.float64)
    else:
        import features
        proc, model = features.load_encoder(a.device)
        X = features.embed(np.load(a.crops), proc, model).astype(np.float64)
    assert X.ndim == 2 and X.shape[1] == 2048, X.shape
    p = probabilities(X, np.load(a.weights), a.design, a.fold)
    np.savetxt(a.out, p, fmt="%.6f", header="p_pnn_positive", comments="")
    print(f"scored {len(p)} neurons with design {a.design}, "
          f"{'fold ' + str(a.fold) if a.fold else 'mean of five folds'}; wrote {a.out}")


if __name__ == "__main__":
    main()
