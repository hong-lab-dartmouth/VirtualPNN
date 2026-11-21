"""Reference classifier: the shipped scores give the paper's AUROC, the fold specification is consistent with the
cell table, and (once weights.npz is present) the weights file has the documented layout."""
import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
RC = Path(__file__).resolve().parents[1] / "reference_classifier"
DESIGNS = ("patient_grouped", "core_split_patient_clean", "core_split_core_held_out")
# design: (neurons, PNN-positive, patients, AUROC) on the size-gated neurons of the paper's comparison
PAPER = {"patient_grouped": (2023, 121, 9, 0.717), "core_split_patient_clean": (990, 46, 7, 0.757),
         "core_split_core_held_out": (2023, 121, 9, 0.739)}


def auroc(y, s):
    r = pd.Series(np.asarray(s, float)).rank().to_numpy()
    y = np.asarray(y).astype(bool)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def test_scores_reproduce_the_paper():
    s = pd.read_csv(RC / "scores.csv")
    assert (len(s), int(s.label.sum()), s.patient.nunique(), s.core.nunique()) == (2650, 358, 9, 14)
    for d, (n, pos, pat, auc) in PAPER.items():
        m = (s.size_gated == 1) & (s["compared_" + d] == 1)
        assert (int(m.sum()), int(s.label[m].sum()), s.patient[m].nunique()) == (n, pos, pat)
        assert abs(auroc(s.label[m], s[d][m]) - auc) < 0.0006


def test_folds_match_the_cell_table():
    s = pd.read_csv(RC / "scores.csv")
    F = json.loads((RC / "folds.json").read_text())["designs"]
    assert set(F) == set(DESIGNS)
    for d in DESIGNS:
        assert set(F[d]["folds"]) == set("12345") and set(F[d]["score_folds_of_core"]) == set(s.core)
        for core, folds in F[d]["score_folds_of_core"].items():
            g = s[s.core == core]
            assert g[d].notna().all() == bool(folds)                 # a core has scores exactly when a fold scores it
            for k in folds:                                          # and no model that scores a core trained on it
                spec = F[d]["folds"][str(k)]
                if len(folds) == 1 or d == "core_split_patient_clean":
                    assert not set(g.patient) & set(spec.get("train_patients", [])), (d, core, k)
                    assert core not in spec.get("train_cores", []), (d, core, k)


def test_weights_layout():
    w = RC / "weights.npz"
    if not w.exists():
        pytest.skip("weights.npz is not in this checkout")
    z = np.load(w)
    for d in DESIGNS:
        assert z[f"{d}__mean"].shape == z[f"{d}__scale"].shape == z[f"{d}__coef"].shape == (5, 2048)
        assert z[f"{d}__intercept"].shape == (5,)
