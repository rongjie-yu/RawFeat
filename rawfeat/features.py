"""Dense score decoding, keypoint selection, and descriptor interpolation."""

from __future__ import annotations

import torch
import numpy as np
from torch.nn import functional as F


def score_map(logits: torch.Tensor) -> torch.Tensor:
    n, channels, hc, wc = logits.shape
    if channels != 65:
        raise ValueError("detector logits must have 65 channels")
    scores = logits.softmax(dim=1)[:, :64]
    return scores.permute(0, 2, 3, 1).reshape(n, hc, wc, 8, 8).permute(0, 1, 3, 2, 4).reshape(n, hc * 8, wc * 8)


def sample_descriptors(grid: torch.Tensor, points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinear sampling at Bayer pixel coordinates, with half-pixel centers."""
    if points.numel() == 0:
        return grid.new_empty((0, grid.shape[0]))
    normalizer = points.new_tensor([width, height])
    positions = (points + 0.5) / normalizer * 2 - 1
    sampled = F.grid_sample(grid[None], positions[None, None], mode="bilinear", align_corners=False)
    return F.normalize(sampled[0, :, 0].T, dim=1)


def extract_features(
    logits: torch.Tensor,
    descriptors: torch.Tensor,
    *,
    threshold: float = 0.005,
    nms_radius: int = 4,
    max_points: int = 1024,
    border: int = 8,
    valid: torch.Tensor | None = None,
    crop_offset: tuple[float, float] = (0.0, 0.0),
) -> list[dict[str, torch.Tensor]]:
    scores = score_map(logits)
    n, height, width = scores.shape
    eligible = scores >= threshold
    if border:
        eligible[:, :border] = False
        eligible[:, height - border:] = False
        eligible[:, :, :border] = False
        eligible[:, :, width - border:] = False
    if valid is not None:
        eligible &= valid.bool()
    # Excluded pixels cannot suppress a valid keypoint.
    masked = scores.masked_fill(~eligible, float("-inf"))
    max_scores = F.max_pool2d(masked[:, None], 2 * nms_radius + 1, stride=1, padding=nms_radius)[:, 0]
    keep = eligible & (masked == max_scores)
    offset = scores.new_tensor(crop_offset)
    outputs = []
    for index in range(n):
        y, x = torch.where(keep[index])
        values = scores[index, y, x]
        order = torch.argsort(values, descending=True, stable=True)
        # Greedy tie handling retains spaced points across a plateau; choosing
        # each pixel's smallest neighbor index would erase its whole interior.
        candidates = torch.stack((y[order], x[order]), dim=1).cpu().numpy()
        suppressed = np.zeros((height, width), dtype=bool)
        selected = []
        for rank, (cy, cx) in enumerate(candidates):
            if len(selected) >= max_points:
                break
            if suppressed[cy, cx]:
                continue
            selected.append(rank)
            suppressed[max(0, cy - nms_radius):cy + nms_radius + 1,
                       max(0, cx - nms_radius):cx + nms_radius + 1] = True
        order = order[torch.tensor(selected, device=order.device, dtype=torch.long)]
        points = torch.stack((x[order], y[order]), dim=1).to(scores.dtype)
        outputs.append({
            "points": points + offset,
            "scores": values[order],
            "descriptors": sample_descriptors(descriptors[index], points, height, width),
        })
    return outputs
