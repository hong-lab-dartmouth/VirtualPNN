"""Fit the reference classifier on the folds of each generator design and score every annotated neuron.

The classifier is a logistic regression (L2, C = 0.1, balanced class weights, lbfgs, 3,000 iterations) on
standardized 2,048-d Phikon-v2 features of 96-pixel H&E crops (features.py). It is trained on the annotated
neurons, one feature per neuron, and on nothing else. The paper's classifier uses no augmentation. A bank that also
holds augmented copies (build_bank.py --copies 5) gives the paper's sensitivity analysis.

One model is fitted per design and fold, on the neurons the matching generator fold was allowed to train on
(folds.json):

  patient_grouped              folds of whole patients; a fold's model never sees its test or validation patients
  core_split_patient_clean     the submitted core split; a fold's model never sees a patient absent from that
                               generator fold's training set
  core_split_core_held_out     the submitted core split; a fold's model sees only cores of that generator fold's
                               training set (other cores of a test patient may remain, as for the generators)

A neuron's score is the mean probability over the folds listed for its core under `score_folds_of_core`: one fold
where a core is a test core of exactly one fold, several where several folds are clean for it, none where no fold is
(the neuron then has no score in that design).

Inputs
  --bank   bank.npz    X (n, 2048) float32 features; cell (n,) row of cells.csv each feature belongs to;
                       augmented (n,) bool, False for the one unaugmented feature of each neuron (all False
                       in the paper's classifier)
  --cells  cells.csv   cell, core, patient, label (1 = PNN-positive), size_gated, nucleus_diameter_um
  --folds  folds.json  training units and scoring folds per design
Outputs
  --out-weights weights.npz   per design: mean, scale, coef (5, 2048) and intercept (5,), fold order 1 to 5
  --out-scores  scores.csv    cells.csv columns plus one probability column per design

    python fit.py --bank bank.npz --out-weights weights.npz --out-scores scores.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
C, CLASS_WEIGHT, MAX_ITER = 0.1, "balanced", 3000
FOLDS = ("1", "2", "3", "4", "5")


def train_mask(cells: pd.DataFrame, spec: dict) -> np.ndarray:
    """Boolean per neuron: may the model of this fold train on it?"""
    if "train_patients" in spec:
        return cells["patient"].isin(spec["train_patients"]).to_numpy()
    return cells["core"].isin(spec["train_cores"]).to_numpy()


def fit_one(X, y, X_score):
    """(P(PNN-positive) for X_score, mean, scale, coef, intercept) of one standardized logistic regression."""
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, class_weight=CLASS_WEIGHT, max_iter=MAX_ITER).fit(sc.transform(X), y)
    assert list(m.classes_) == [0, 1]
    return m.predict_proba(sc.transform(X_score))[:, 1], sc.mean_, sc.scale_, m.coef_[0], float(m.intercept_[0])


def combine(P: dict, cells: pd.DataFrame, score_folds_of_core: dict) -> np.ndarray:
    """Per-neuron score: mean over the folds listed for the neuron's core (NaN where none is listed)."""
    s = np.full(len(cells), np.nan)
    for core, folds in score_folds_of_core.items():
        m = (cells["core"] == core).to_numpy()
        if folds:
            s[m] = np.mean([P[str(k)][m] for k in folds], axis=0)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--cells", default=str(HERE / "cells.csv"))
    ap.add_argument("--folds", default=str(HERE / "folds.json"))
    ap.add_argument("--out-weights", required=True)
    ap.add_argument("--out-scores", required=True)
    a = ap.parse_args()

    cells = pd.read_csv(a.cells, dtype={"cell": str, "core": str, "patient": str})
    designs = json.loads(Path(a.folds).read_text())["designs"]
    z = np.load(a.bank)
    X, row_cell, aug = z["X"].astype(np.float32), z["cell"].astype(int), z["augmented"].astype(bool)
    y_cell = cells["label"].to_numpy().astype(int)
    assert len(X) == len(row_cell) == len(aug) and row_cell.max() == len(cells) - 1
    plain = np.where(~aug)[0]
    assert (row_cell[plain] == np.arange(len(cells))).all(), "the unaugmented features must follow cells.csv order"
    X_score = X[plain]

    weights, out = {}, cells.copy()
    for name, d in designs.items():
        P, W = {}, []
        for k in FOLDS:
            rows = train_mask(cells, d["folds"][k])[row_cell]
            p, mean, scale, coef, b = fit_one(X[rows], y_cell[row_cell[rows]], X_score)
            P[k] = p
            W.append((mean, scale, coef, b))
            print(f"{name} fold {k}: {int(rows.sum())} training features, {int(y_cell[row_cell[rows]].sum())} "
                  f"PNN-positive", flush=True)
        out[name] = combine(P, cells, d["score_folds_of_core"])
        for j, key in enumerate(("mean", "scale", "coef")):
            weights[f"{name}__{key}"] = np.stack([w[j] for w in W]).astype(np.float64)
        weights[f"{name}__intercept"] = np.array([w[3] for w in W], np.float64)
    np.savez_compressed(a.out_weights, **weights)
    out.to_csv(a.out_scores, index=False, float_format="%.10g")
    print("wrote", a.out_weights, "and", a.out_scores)


if __name__ == "__main__":
    main()
