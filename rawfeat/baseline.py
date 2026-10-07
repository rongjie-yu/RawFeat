"""Raw-SLAM's uint16 Linear Raw, shared MeanAD, and official SuperPoint baseline."""

from __future__ import annotations

import cv2
import colour_demosaicing
import numpy as np
import torch

from .features import extract_features
from .sensor import BLACK_LEVEL, SIGNAL_RANGE, WHITE_LEVEL


def linear_raw_uint16(noisy_bayer_dn: np.ndarray) -> np.ndarray:
    demosaiced = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(noisy_bayer_dn, "RGGB")
    linear = np.clip(demosaiced - BLACK_LEVEL, 0, SIGNAL_RANGE) / WHITE_LEVEL
    return np.clip(linear * 65535.0, 0, 65535).astype(np.uint16)


def meanad(rgb: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    mean = float(values.mean(dtype=np.float64))
    deviation = float(np.abs(values.astype(np.float64) - mean).mean(dtype=np.float64))
    if deviation <= epsilon:
        return np.zeros_like(values)
    return np.clip((values.astype(np.float64) - mean + 2 * deviation) / (4 * deviation + epsilon), 0, 1).astype(np.float32)


def baseline_gray(noisy_bayer_dn: np.ndarray) -> np.ndarray:
    encoded = linear_raw_uint16(noisy_bayer_dn)
    rgb = encoded.astype(np.float32) / 65535.0
    return cv2.cvtColor(meanad(rgb), cv2.COLOR_RGB2GRAY)


@torch.no_grad()
def baseline_features(teacher: torch.nn.Module, noisy_bayer_dn: torch.Tensor, valid: torch.Tensor | None = None) -> list[dict[str, torch.Tensor]]:
    device = next(teacher.parameters()).device
    grays = np.stack([baseline_gray(frame.cpu().numpy()) for frame in noisy_bayer_dn])
    inputs = torch.from_numpy(grays[:, None]).to(device)
    logits, descriptors = teacher(inputs)
    return extract_features(logits, descriptors, valid=valid)
