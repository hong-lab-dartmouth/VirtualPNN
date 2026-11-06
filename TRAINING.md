# Training

Five-fold cross-validation, five models. The exact per-fold VirtualPNN commands are in
[`scripts/`](scripts/); this file gives the settings and the parts that differ between models.

## Data

The commands below are as they were run, so `--dataroot` still points at
the directory the authors used, `../PNN_finetune/`. Substitute your own. The layout is the
standard CUT/CycleGAN one: each fold directory holds `trainA/` and `trainB/` (and `testA/`,
`testB/`) of unpaired 256-pixel patches, A being H&E and B the aggrecan channel. Training the
anti-vessel variants additionally reads `trainA_vessel/`, a pool of vessel-containing H&E
patches sampled at `--vessel_patch_ratio`.

- H&E and aggrecan (IF) patches, 256 x 256, unaligned. Folds live under `PNN_finetune/`:
  `cv_splits_AV6/fold_{1..5}` for VirtualPNN-P+AV, `cv_splits/fold_{1..5}` for the others.
- All five models start from the AI-FFPE base (`--continue_train --epoch 5`, with `--pretrained_name aiffpe` for the
  two VirtualPNN variants and `--pretrained_name aiffpe_weights` for the baselines). The base folders hold the
  same three files, `5_net_G.pth`, `5_net_D.pth` and `5_net_F.pth` (the generator is the same file, sha256
  `7d469b5d...`), and each model loads its generator, discriminator and, in the CUT family, patch-feature
  network from them (CycleGAN loads the generator and discriminator files under the names of its two generators
  and two discriminators and has no patch-feature network). The base is needed only to retrain; inference does
  not use it. `fetch_weights.sh` writes it to `weights/aiffpe/` and, with CycleGAN's file names added,
  `weights/aiffpe_weights/`; pass `--checkpoints_dir weights` to train from there.

## VirtualPNN-P+AV (main model), per fold k
Verbatim in `scripts/runtrain_VirtualPNN_AV6_Fold{k}.sh`:
```
python train.py \
  --dataroot ../PNN_finetune/cv_splits_AV6/fold_k --name VirtualPNN_AV6_Foldk \
  --model virtualpnn --CUT_mode CUT --continue_train --pretrained_name aiffpe --epoch 5 \
  --lr_G 0.00005 --lr_D 0.00002 --n_epochs 20 --n_epochs_decay 20 \
  --batch_size 8 --load_size 256 --crop_size 256 \
  --lambda_NCE 1 --lambda_VGG 2 --self_regularization 0 --display_id -1 \
  --use_vessel_patches --vessel_patch_ratio 0.3 --lambda_anti_vessel 5
```
`netG resnet_9blocks`, `nce_layers 0,4,8,12,16`, `nce_T 0.07`, `nce_idt True` are defaults.

## VirtualPNN-P (ablation: anti-vessel off), per fold k
`scripts/runtrain_VirtualPNN_noAV_Fold{k}.sh`. Same as above with `cv_splits/fold_k` and:
```
  --vessel_patch_ratio 0 --lambda_anti_vessel 0
```
VirtualPNN-P is therefore **not** a one-factor ablation of VirtualPNN-P+AV. Besides the
anti-vessel term, the two differ in their training data and length: VirtualPNN-P trains on
`cv_splits/` without the vessel sampler, so its dataset length (the larger of `trainA` and
`trainB`) is shorter and it takes 2.9-4.0 times fewer generator steps per run (mean 3.5 times),
and its `trainA` differs (fold 1: 3,945 patches against 3,331 for VirtualPNN-P+AV, whose
vessel-containing patches were moved to `trainA_vessel`). The two models differ in the anti-vessel
objective, the vessel sampler and the length of the schedule. The figures come from
the runs' own `train_opt.txt` (copied to `scripts/train_opt/`), `loss_log.txt` and fold directories.

## Baselines (per fold k)
Verbatim in [`scripts/baselines/`](scripts/baselines/). They were run in a checkout of the public
AI-FFPE code, where the model class is `--model cut` (the class this repository calls `virtualpnn`),
and each fine-tuned from AI-FFPE's published weights (`--continue_train --pretrained_name
aiffpe_weights --epoch 5`). The same folds as VirtualPNN-P (`cv_splits/fold_k`), `netG resnet_9blocks`:
- **CUT**: `runtrain_CUT_Fold{k}.sh`, `--model cut --CUT_mode CUT --self_regularization 0`.
- **AI-FFPE**: `runtrain_AIFFPE_Fold{k}.sh`, the published AI-FFPE recipe (`--CUT_mode CUT`).
- **CycleGAN**: `runtrain_CycleGAN_Fold{k}.sh`, `--model cycle_gan`.

None of the three has a perceptual or an anti-vessel term or reads vessel patches, so each ran the same number of
generator steps as VirtualPNN-P. All train for 20 + 20 epochs with batch
size 8 at the base learning rate of 2e-4, which applies to both generator and discriminator; the two
VirtualPNN models set `--lr_G 0.00005 --lr_D 0.00002` instead.

The full option record of every run, the two VirtualPNN models and the baselines, is the
`train_opt.txt` that `train.py` wrote at launch, copied verbatim to
[`scripts/train_opt/`](scripts/train_opt/). Those files record the training
configuration. They were written before the model class was renamed and therefore record
`model: cut` (CycleGAN: `cycle_gan`); `cut` is the class now selected by `--model virtualpnn`.

## Environment
See `requirements.txt`. Single GPU (`--gpu_id 0`). `--display_id -1` runs headless (no visdom).
GAN training is stochastic: retraining reproduces the method, not the published weights.
