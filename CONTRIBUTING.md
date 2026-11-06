# Contributing

Bug reports and pull requests are welcome.

## Reporting a problem

Open an issue with the command you ran, the full traceback, and your PyTorch and CUDA
versions (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`).
If it concerns the output rather than a crash, say which fold or folds you used and
whether you passed `--overlap`.

## Before opening a pull request

```bash
python -m pytest tests/          # must pass
python infer.py --help           # entry points must still load
```

The tests run without weights, a GPU or data, so they are quick.

## Things worth knowing before you change the model

The released weights fix the architecture. The generator key set is pinned by a test
(58 tensors, including the `SAB.*` spatial attention block). Any change to
`ResnetGenerator` or to `ARCH` in `infer.py` that alters that set will make all 25
published checkpoints fail to load. If you intend such a change, it belongs in a new
model class, not in this one.

Perineuronal-net images are never contrast-stretched. Predicted aggrecan is shown and
saved at native windowing, never percentile-stretched or gamma-adjusted. The same applies to
figures and README artwork.

If you change tiling, blending or the intensity conversion, run the same input through the
old and new code and report the difference.

## Scope

This repository is the model: training, inference, the recorded training commands and the
weights manifest. The evaluation analyses of the paper are not part of it.
