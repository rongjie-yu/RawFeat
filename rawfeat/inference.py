"""Deployment extraction from exposure-normalized Canon Bayer sensor DN."""

from __future__ import annotations

import torch

from .features import extract_features
from .sensor import pack_student


@torch.no_grad()
def extract_bayer(
    model: torch.nn.Module,
    bayer_sensor_dn: torch.Tensor,
    *,
    crop_offset: tuple[float, float] = (0.0, 0.0),
    threshold: float = 0.005,
    max_points: int = 1024,
) -> dict[str, torch.Tensor]:
    """Return original Bayer pixel points, scores and 128-D unit descriptors."""
    if bayer_sensor_dn.ndim != 2 or any(size % 8 for size in bayer_sensor_dn.shape):
        raise ValueError("Bayer input must be HxW with dimensions divisible by 8")
    device = next(model.parameters()).device
    packed = pack_student(bayer_sensor_dn.to(device, dtype=torch.float32)[None])
    output = model(packed)
    return extract_features(output["logits"], output["descriptors"], threshold=threshold,
                            max_points=max_points, crop_offset=crop_offset)[0]
