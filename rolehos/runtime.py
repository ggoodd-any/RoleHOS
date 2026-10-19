import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from rolehos import ROOT
from mmseg.registry import MODELS
from mmseg.models.utils.relation_state import foreground5_to_compat6

MEAN = np.array([106.01075, 95.40013, 87.42854], dtype=np.float32)
STD = np.array([64.356636, 60.888744, 61.41911], dtype=np.float32)
NAMES = ["BG", "LH", "RH", "LO", "RO", "TO"]
PALETTE = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255], [0, 255, 0]],
    dtype=np.uint8,
)


def deterministic(seed=0, strict=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(strict)
    return seed


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest():
    return json.loads((ROOT / "weights/manifest.json").read_text())


def record_for(variant="full", seed=0):
    return next(
        x for x in read_manifest() if x["variant"] == variant and x["seed"] == seed
    )


def load_model(variant="full", seed=0, device="cpu", checkpoint=None):
    deterministic(seed)
    record = record_for(variant, seed)
    cfg = Config.fromfile(str(ROOT / record["config"]))
    init_default_scope("mmseg")
    cfg.model.backbone.checkpoint = ""
    model = MODELS.build(cfg.model)
    path = Path(checkpoint) if checkpoint else ROOT / record["checkpoint"]
    if not checkpoint and sha256(path) != record["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {path.name}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), record


def preprocess(rgb):
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image")
    resized = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(
        ((resized.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy()
    ).unsqueeze(0)


@torch.inference_mode()
def predict(model, rgb):
    inputs = preprocess(rgb).to(next(model.parameters()).device)
    result = model.forward_relation(inputs, route="slow", profile=False)
    # Both fixed-role and anonymous-query models expose their compatible five foreground logits.
    logits = result["logits5"]
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite model output")
    return (
        foreground5_to_compat6(logits.sigmoid(), threshold=0.5)[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def overlay(rgb, mask, alpha=0.55):
    image = cv2.resize(np.asarray(rgb), (448, 448), interpolation=cv2.INTER_LINEAR)
    colors = PALETTE[mask]
    mixed = image.copy()
    fg = mask != 0
    mixed[fg] = np.rint((1 - alpha) * image[fg] + alpha * colors[fg]).astype(np.uint8)
    return mixed, colors


def confusion(pred, gt):
    valid = (gt != 255) & (gt >= 0) & (gt < 6)
    if not np.isin(gt, [0, 1, 2, 3, 4, 5, 255]).all():
        raise ValueError("Unexpected label ids; use six-class EgoHOS masks")
    return np.bincount(
        6 * gt[valid].astype(np.int64) + pred[valid], minlength=36
    ).reshape(6, 6)


def metrics(cm, split):
    union = cm.sum(0) + cm.sum(1) - cm.diagonal()
    iou = np.divide(
        cm.diagonal(), union, out=np.zeros(6, dtype=np.float64), where=union != 0
    )
    selected = [1, 2, 4, 5] if split == "minihoi4d" else [1, 2, 3, 4, 5]
    return {
        "mIoU": float(np.nanmean(iou[selected]) * 100),
        "IoU": {n: None if np.isnan(x) else float(x * 100) for n, x in zip(NAMES, iou)},
        "confusion": cm.tolist(),
    }
