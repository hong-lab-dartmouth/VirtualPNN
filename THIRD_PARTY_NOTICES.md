# Third-party code and models

Most of the training and inference framework in this repository is not ours. The full licence
texts are in [`LICENSES/`](LICENSES/), copied verbatim from the upstream repositories; they
apply to the files that originated there, and their conditions (retain the copyright notice,
the list of conditions and the disclaimer) are met by shipping those files.

## Contrastive Unpaired Translation (CUT), and pytorch-CycleGAN-and-pix2pix before it

- CUT: Copyright (c) 2020, Taesung Park and Jun-Yan Zhu. All rights reserved.
  https://github.com/taesungp/contrastive-unpaired-translation
- pytorch-CycleGAN-and-pix2pix, from which CUT is built: Copyright (c) 2017, Jun-Yan Zhu and
  Taesung Park. All rights reserved. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- That code carries forward notices for pix2pix (Copyright (c) 2016, Phillip Isola and Jun-Yan
  Zhu) and dcgan.torch (Copyright (c) 2015, Facebook, Inc.).

Licence: a BSD-style licence (two-clause text for CUT, CycleGAN and pix2pix; three-clause for
dcgan.torch). Full text: [`LICENSES/contrastive-unpaired-translation.txt`](LICENSES/contrastive-unpaired-translation.txt),
which contains all four notices. That file also contains the licences of stylegan2-pytorch (MIT)
and NVIDIA StyleGAN2 (non-commercial); the StyleGAN2 code they cover was in the upstream tree but
is not in this repository, so those two sections do not apply here.

Provides the training loop, the option parsers, the dataset and visualisation utilities, the
ResNet and PatchGAN architectures, the PatchNCE loss and the CycleGAN model: essentially all of
`data/`, `options/`, `util/`, `models/networks.py`, `models/patchnce.py`,
`models/cycle_gan_model.py`, `models/base_model.py`, `models/__init__.py`, `train.py` and
`test.py`, and the CUT core of `models/virtualpnn_model.py`.

## AI-FFPE

https://github.com/DeepMIALab/AI-FFPE (DeepMIA Lab). The code in this repository was forked
from AI-FFPE, which had adapted CUT for histology. AI-FFPE is distributed under the CUT licence
file unchanged (the same text as `LICENSES/contrastive-unpaired-translation.txt`, with the same
copyright holders); it adds no licence or copyright notice of its own.

From AI-FFPE this repository contains the spatial attention block (`SpatialAttention`, applied as
`self.SAB` in `ResnetGenerator`, `models/networks.py`) and the self-regularization option. The
attention block is part of the released generators (its weights are in every published checkpoint
as `SAB.conv1/2/3`). The CBAM attention module, the StyleGAN2 generator and discriminator, the
SinCUT and template models and a second copy of the network definitions are not included.

## torchvision

`data/image_folder.py` is modified from torchvision's `datasets/folder.py`.
Copyright (c) Soumith Chintala 2016. BSD 3-Clause; full text in
[`LICENSES/torchvision.txt`](LICENSES/torchvision.txt).

## Models and tools that are downloaded, not bundled

None of these is redistributed here; each is fetched by its own package under its own terms.

| What | Used by | Terms |
|---|---|---|
| ImageNet VGG16 weights (torchvision) | the perceptual loss during training (`--lambda_VGG`) | torchvision model weights terms |
| InceptionV3 FID weights (`pytorch-fid`) | FID (`INFERENCE.md`, Figure 3) | pytorch-fid: Apache 2.0 |

## What is ours

The anti-vessel loss (`--lambda_anti_vessel`, `calculate_anti_vessel_loss` in
`models/virtualpnn_model.py`), the vessel-patch sampling it depends on
(`--use_vessel_patches`, `--vessel_patch_ratio`, the `trainA_vessel` handling in
`data/unaligned_dataset.py`), the VGG perceptual term (`--lambda_VGG`), the perineuronal-net
fine-tuning recipe in `scripts/`, `infer.py`, `postprocess/` and `tests/`. These are released under the BSD 3-Clause licence in [`LICENSE`](LICENSE).

## Weights

The trained generators distributed through the weights archive are released separately under
CC BY 4.0; see `LICENSE.txt` in the archive. They are not covered by the code licences above. The archive also
redistributes AI-FFPE's published base weights unchanged, under the terms of the AI-FFPE repository's licence
file, the CUT licence text above (`NOTICE_AI-FFPE_BSD-3-Clause.txt` in the archive).
