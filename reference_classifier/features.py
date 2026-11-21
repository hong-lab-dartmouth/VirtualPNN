"""Features of the reference classifier: 96-pixel H&E crops around nuclei, embedded by the frozen Phikon-v2 encoder.

Frame. H&E is resampled (bicubic) to 0.400705248 µm per pixel. A crop is 96 x 96 pixels centred on the rounded
nucleus centroid; nuclei whose crop would leave the image are dropped.

Feature (2,048-d). Phikon-v2 `last_hidden_state` of the crop: the CLS token concatenated with the mean of the patch
tokens inside a disc of radius 30 pixels centred in the crop. The disc is drawn at crop size, resized bilinearly to
the token grid and thresholded at 64/255.

Nuclei in the paper were detected with StarDist `2D_versatile_he` (probability threshold 0.45, NMS threshold 0.3,
percentile normalization 1 to 99.8). The classifier was trained and tested on annotated neurons only, so a nucleus
passed to it should be a neuron; it has no step that tells neurons from other cells.

Needs numpy and Pillow; `load_encoder` and `embed` need torch and transformers, `detect_nuclei` needs stardist and
csbdeep. Phikon-v2 (owkin/phikon-v2 on Hugging Face) is distributed by its authors under a non-commercial license.
"""
import numpy as np
from PIL import Image

MPP = 0.400705248                 # µm per pixel of the crop frame
CROP_PX, HALF_PX, DISC_R_PX = 96, 48, 30
ENCODER = "owkin/phikon-v2"
STARDIST = dict(model="2D_versatile_he", prob_thresh=0.45, nms_thresh=0.3)
BLOCK, OVERLAP, CONTEXT = 2048, 128, 64


def resample(rgb: np.ndarray, um_per_px: float) -> np.ndarray:
    """RGB uint8 image resampled (bicubic) from um_per_px to the crop frame."""
    h, w = rgb.shape[:2]
    s = um_per_px / MPP
    return np.asarray(Image.fromarray(rgb).resize((round(w * s), round(h * s)), Image.BICUBIC))


def crops(rgb: np.ndarray, x, y, half: int = HALF_PX):
    """(n_kept, 96, 96, 3) uint8 crops centred on rounded (x, y) in the crop frame, and the boolean kept mask."""
    h, w = rgb.shape[:2]
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    keep = (xi - half >= 0) & (yi - half >= 0) & (xi + half < w) & (yi + half < h)
    out = np.stack([rgb[b - half:b + half, a - half:a + half] for a, b in zip(xi[keep], yi[keep])]) \
        if keep.any() else np.zeros((0, 2 * half, 2 * half, 3), np.uint8)
    return out.astype(np.uint8), keep


def disc_tokens(grid: int, crop_px: int = CROP_PX, r_px: int = DISC_R_PX) -> np.ndarray:
    """Boolean (grid*grid,) mask of the patch tokens inside the centred disc (all tokens if none)."""
    c = crop_px // 2
    d = np.hypot(*np.meshgrid(np.arange(crop_px) - c, np.arange(crop_px) - c)) <= r_px
    m = np.asarray(Image.fromarray(d.astype(np.uint8) * 255).resize((grid, grid), Image.BILINEAR)) > 64
    return (m if m.any() else np.ones_like(m)).reshape(-1)


def load_encoder(device: str = "cuda"):
    """(processor, frozen Phikon-v2) on `device`; the weights are downloaded from Hugging Face."""
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER, trust_remote_code=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


def embed(crop_stack: np.ndarray, proc, model, batch: int = 128) -> np.ndarray:
    """float32 (n, 2048) features of uint8 crops (n, 96, 96, 3). bfloat16 autocast on CUDA, as in the paper."""
    import torch
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for i in range(0, len(crop_stack), batch):
            px = proc(images=[Image.fromarray(c) for c in crop_stack[i:i + batch]],
                      return_tensors="pt")["pixel_values"]
            h = model(pixel_values=px.to(dev)).last_hidden_state
            mask = torch.as_tensor(disc_tokens(int(round((h.shape[1] - 1) ** 0.5))), device=dev)
            out.append(torch.cat([h[:, 0].float(), h[:, 1:][:, mask].float().mean(1)], 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 2048), np.float32)


def detect_nuclei(rgb: np.ndarray) -> dict:
    """StarDist nuclei of an image already in the crop frame: {x, y (pixels), area_px, diam_um} as arrays."""
    from csbdeep.utils import normalize
    from scipy import ndimage as ndi
    from stardist.models import StarDist2D
    model = StarDist2D.from_pretrained(STARDIST["model"])
    x = normalize(rgb, 1, 99.8)
    kw = dict(axes="YXC", prob_thresh=STARDIST["prob_thresh"], nms_thresh=STARDIST["nms_thresh"])
    h, w = x.shape[:2]
    if min(h, w) >= BLOCK:
        lab, _ = model.predict_instances_big(x, block_size=BLOCK, min_overlap=OVERLAP, context=CONTEXT, **kw)
    else:
        lab, _ = model.predict_instances(x, n_tiles=(-(-h // BLOCK), -(-w // BLOCK), 1), **kw)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    one = np.ones_like(lab)
    com = np.array(ndi.center_of_mass(one, lab, ids)).reshape(-1, 2)
    area = np.asarray(ndi.sum(one, lab, ids), float)
    return {"x": com[:, 1], "y": com[:, 0], "area_px": area, "diam_um": 2 * np.sqrt(area / np.pi) * MPP}
