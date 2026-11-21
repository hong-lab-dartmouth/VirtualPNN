# Reference classifier

A supervised per-cell classifier that the paper uses as a reference for the generators. It asks one question:
does the H&E image of a neuron carry information about whether that neuron has a perineuronal net? It is not part
of VirtualPNN, it produces no image, and it is not a detector of perineuronal nets.

## What it is

A 96-pixel H&E crop around an annotated neuron is embedded by the frozen Phikon-v2 encoder (2,048 features), and a
logistic regression on the standardized features returns the probability that the neuron is PNN-positive. The
labels come from aggrecan immunofluorescence of the same tissue section, imaged before the section was restained
with H&E. Training used the annotated neurons, one crop each, without augmentation.

The classifier takes an annotated neuron as its input. It has no step that finds neurons or tells them from other
cells, so it cannot be run on a whole slide as it stands.

## Results in the paper

2,650 annotated neurons (358 PNN-positive) on 14 cores from 9 patients. The comparison with the generators uses
the size-gated neurons that every recipe scored. Size-gated means a nucleus at or above the 70th percentile of its
core, the range from which PNN-negative neurons were annotated.

| Design | Neurons (PNN-positive) | Patients | Classifier AUROC | Nucleus diameter alone |
|---|---|---|---|---|
| Patient-grouped folds | 2,023 (121) | 9 | 0.717 | 0.588 |
| Core split, patient-clean cores | 990 (46) | 7 | 0.757 | 0.514 |
| Core split, core held out | 2,023 (121) | 9 | 0.739 | 0.588 |

On the patient-grouped neurons the ten generator recipes reached 0.552 to 0.603 with the annulus score (the stained
fraction of a ring 4 to 12 µm from the neuron) and 0.490 to 0.527 with the WFA-detector score. `scores.csv`
reproduces the classifier column:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
s = pd.read_csv("scores.csv")
m = (s.size_gated == 1) & (s.compared_patient_grouped == 1)
roc_auc_score(s.label[m], s.patient_grouped[m])        # 0.717
```

## Files

| File | Content |
|---|---|
| `features.py` | Crop frame, Phikon-v2 feature, StarDist settings. |
| `build_bank.py` | Crops to feature bank: embeds each crop, and its augmented copies if asked. |
| `fit.py` | Fits one model per design and fold and scores every neuron. |
| `predict.py` | Scores features or crops with the released weights. |
| `augment.py` | The appearance augmentation of the sensitivity analysis. |
| `cells.csv` | The annotated neurons: core, patient, label, size gate, nucleus diameter, and whether each entered a design's comparison. |
| `folds.json` | Per design and fold, the patients or cores a model may train on, and the folds that score each core. |
| `scores.csv` | The paper's out-of-fold probability of every neuron under each design. |
| `weights.npz` | Fifteen fitted models (three designs, five folds): feature mean and scale, coefficients, intercept. |

Patient, core and neuron names are arbitrary. A core with no score in a design has an empty cell.

## Folds

Each model was fitted only on neurons that the matching generator fold was allowed to train on, so the classifier
and the generators are compared on the same held-out tissue. In the patient-grouped design a fold's model never
sees its test or validation patients. One patient was in no generator fold; its neurons are scored by the mean of
the five models, as the generators scored that core.

## Augmentation

The paper's classifier uses no augmentation: the generators were trained without appearance augmentation, the
classifier is trained and tested on one section, and the fit is then deterministic. Applied to the features, the
released weights reproduce `scores.csv` to within 1e-6. As a sensitivity analysis the paper also fitted the classifier
with five augmented copies of each crop (`build_bank.py --copies 5`); four such fits gave 0.67 to 0.69 on the
patient-grouped neurons.

## Use

```bash
pip install numpy pandas scikit-learn Pillow            # fit.py and predict.py --features
pip install torch transformers                          # features.embed, predict.py --crops
python predict.py --crops crops.npy --device cuda --out scores.csv
```

`crops.npy` holds uint8 crops of shape (n, 96, 96, 3) at 0.4007 µm per pixel, each centred on a neuron
(`features.resample` and `features.crops` make them from an image and nucleus centroids).

`fit.py` needs the feature bank of the annotated neurons (`bank.npz`, 2,650 features). It is derived from patient
tissue and is available from the corresponding authors on request, like the images.

## Limits

- Nine patients from one institution, one section, one annotator.
- The AUROC is modest. The classifier is evidence that H&E carries some cell-level signal, not a tool for
  counting or locating nets.
- The classifier was trained and tested on H&E restained after immunofluorescence. Performance on routine H&E,
  other stains, scanners or tissue is untested.

## License

Code, BSD 3-Clause, as the rest of this repository. Weights, CC BY 4.0. The weights are only usable with Phikon-v2
features, and Phikon-v2 is distributed by its authors under a non-commercial license
(https://huggingface.co/owkin/phikon-v2).
