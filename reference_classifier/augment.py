"""Appearance augmentation of the 96-pixel H&E crops, used only in the sensitivity analysis of the reference classifier.

The paper's classifier is trained without augmentation. With five augmented copies per crop its AUROC on the
size-gated neurons of the patient-grouped design was 0.67 to 0.69 over four fits, against 0.717 without.

Eight operations are applied in pixel space, in the order below, each with probability 0.6, and the result is
embedded again by the frozen encoder. Five copies are made of every crop and the original is kept. There is no
geometric augmentation: the aim was variation in stain and scanner appearance, not orientation.

  brightness   multiply by U(0.7, 1.3)
  contrast     scale the deviation from the image mean by U(0.7, 1.3)
  gamma        exponent U(0.7, 1.5) on the 0 to 1 intensity
  blur         Gaussian, sigma U(0, 1.5) pixels
  downsample   resize by U(0.5, 1.0) and back to 96 pixels (bilinear)
  jpeg         re-encode at quality U(30, 90)
  noise        add Gaussian noise, sigma U(2, 12) gray levels
  hed          stain jitter: in hematoxylin-eosin-DAB optical density, each stain is scaled by 1 + U(-0.2, 0.2)
               and shifted by U(-0.1, 0.1)

This file implements the operations from the ranges, order, probability and copy number recorded for the bank.
Where that record is silent the choices are the ones written above and in the code: the contrast pivot, the
resampling filter, and the stain basis (Ruifrok and Johnston) with natural-log optical density.

    copies = augment.bank(crops, k=5, seed=0)        # (n * 5, 96, 96, 3) uint8, five copies per crop in crop order
"""
import io

import numpy as np
from PIL import Image, ImageFilter

K, P_APPLY = 5, 0.6
# rows: unit optical-density vectors of hematoxylin, eosin and DAB (Ruifrok and Johnston 2001)
_M = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_M = _M / np.linalg.norm(_M, axis=1, keepdims=True)
_Mi = np.linalg.inv(_M)


def _u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def brightness(a, r):
    return _u8(a.astype(np.float64) * r.uniform(0.7, 1.3))


def contrast(a, r):
    a = a.astype(np.float64)
    m = a.mean()
    return _u8((a - m) * r.uniform(0.7, 1.3) + m)


def gamma(a, r):
    return _u8(255.0 * (a.astype(np.float64) / 255.0) ** r.uniform(0.7, 1.5))


def blur(a, r):
    return np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(float(r.uniform(0.0, 1.5)))))


def downsample(a, r):
    n = a.shape[0]
    s = max(8, int(round(n * r.uniform(0.5, 1.0))))
    return np.asarray(Image.fromarray(a).resize((s, s), Image.BILINEAR).resize((n, n), Image.BILINEAR))


def jpeg(a, r):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=int(round(r.uniform(30, 90))))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def noise(a, r):
    return _u8(a.astype(np.float64) + r.normal(0.0, r.uniform(2, 12), a.shape))


def hed(a, r):
    od = -np.log((a.astype(np.float64) + 1.0) / 256.0)
    c = od.reshape(-1, 3) @ _Mi
    c = c * (1.0 + r.uniform(-0.2, 0.2, size=3)) + r.uniform(-0.1, 0.1, size=3)
    return _u8(256.0 * np.exp(-(c @ _M).reshape(a.shape)) - 1.0)


BANK = (brightness, contrast, gamma, blur, downsample, jpeg, noise, hed)


def compose(a: np.ndarray, r: np.random.Generator, ops=BANK, p: float = P_APPLY) -> np.ndarray:
    """One augmented copy of a uint8 RGB crop: each operation in turn with probability p."""
    for op in ops:
        if r.uniform() < p:
            a = op(a, r)
    return a


def bank(crops: np.ndarray, k: int = K, seed: int = 0) -> np.ndarray:
    """k augmented copies of each crop, crop-major: the copies of crop i are rows i*k to i*k + k - 1."""
    r = np.random.default_rng(seed)
    out = np.empty((len(crops) * k,) + crops.shape[1:], np.uint8)
    for i, c in enumerate(crops):
        for j in range(k):
            out[i * k + j] = compose(c.copy(), r)
    return out
