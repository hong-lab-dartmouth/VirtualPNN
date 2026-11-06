# Baseline commands (CUT, CycleGAN, AI-FFPE)

The per-fold commands with which the three comparators of the paper were trained and tested, as they
were run. They ran in a checkout of the public AI-FFPE code
([DeepMIALab/AI-FFPE](https://github.com/DeepMIALab/AI-FFPE)), not in this repository: there the
model class is `--model cut` (this repository calls the same class `virtualpnn`), and every run
fine-tuned from AI-FFPE's published weights, placed in that checkout as `checkpoints/aiffpe_weights/`
(`--continue_train --pretrained_name aiffpe_weights --epoch 5`).

| Files | What |
|---|---|
| `runtrain_{AIFFPE,CUT,CycleGAN}_Fold{1..5}.sh` | training, 20 + 20 epochs, batch size 8, base learning rate 2e-4, on `cv_splits/fold_k` |
| `TEST_{AIFFPE,CUT,CycleGAN}_Fold{1..5}.sh` | inference on the fold's PNN test cores and on its vessel-only cores, stitching, FID and the vessel signal (some steps are commented out where the outputs already existed when the file was last run) |
| `TEST_{AIFFPE,CUT,CycleGAN}_tm_inference.sh` | inference on the tumour negative-control cores, all five folds |

`--dataroot ../PNN_finetune/...` is the authors' layout; substitute your own. `../reconstruct_noa.py`
and `../vessel_signal.py` are this repository's `postprocess/reconstruct_noa.py` and
`postprocess/signal_ratio.py`. Each run's options are recorded in its `train_opt.txt`
(`../train_opt/`); every option of these 15 training scripts equals that record. The two VirtualPNN models' commands are one folder
up, in `scripts/`.
