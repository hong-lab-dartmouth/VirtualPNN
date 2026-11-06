# Recorded training options of the 25 submitted runs

One file per run, copied verbatim from the `train_opt.txt` that `train.py` wrote at launch
(`<model>_Fold<k>.train_opt.txt`; models VirtualPNN_AV6 = VirtualPNN-P+AV, VirtualPNN_noAV = VirtualPNN-P, CUT,
CycleGAN, AIFFPE). Paths inside are relative to the training working directory as used at the time
(`../PNN_finetune/cv_splits_AV6/fold_k` for P+AV, `../PNN_finetune/cv_splits/fold_k` for the others; the
pretrained base is `aiffpe` or `aiffpe_weights`, the same AI-FFPE generator, sha256 7d469b5d…). These files are
the record of every hyperparameter. Every file
records `model: cut` (CycleGAN: `cycle_gan`): the runs were made before the model class was renamed, and `cut` is
the class this repository selects with `--model virtualpnn`.
