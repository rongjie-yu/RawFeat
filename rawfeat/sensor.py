"""Official Canon InvISP to calibrated RGGB sensor and student domains."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


BLACK_LEVEL = 2048.0
WHITE_LEVEL = 16383.0
SIGNAL_RANGE = WHITE_LEVEL - BLACK_LEVEL
INVISP_CANON_WHITE = 4095.0
WHITE_BALANCE = (1.0, 1024.0 / 2020.0, 1024.0 / 2020.0, 1458.0 / 2020.0)


def load_invisp(repo: str | Path, device: torch.device | str = "cuda") -> torch.nn.Module:
    repo = Path(repo).resolve()
    if not (repo / "model" / "model.py").is_file() or not (repo / "pretrained" / "canon.pth").is_file():
        raise FileNotFoundError(f"official Canon InvISP source and checkpoint required in {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model.model import InvISPNet

    model = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    state = torch.load(repo / "pretrained" / "canon.pth", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_unused = {f"operations.{index}.actnorm.{name}" for index in range(8) for name in ("bias", "logs")}
    if missing or set(unexpected) != expected_unused:
        raise RuntimeError(f"Canon InvISP checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.rawfeat_repo_path = str(repo)
    return model.requires_grad_(False).eval().to(device)


def invisp_to_sensor_rgb(inverse_output: torch.Tensor) -> torch.Tensor:
    """Invert the official Canon 12-bit gamma/WB preprocessing, then map to EOS5D4 DN."""
    if inverse_output.shape[1] != 3:
        raise ValueError("InvISP output must be NCHW RGB")
    wb = inverse_output.new_tensor((WHITE_BALANCE[0], WHITE_BALANCE[1], WHITE_BALANCE[3]))[None, :, None, None]
    linear_12bit = inverse_output.clamp(0, 1).pow(2.2) * INVISP_CANON_WHITE
    sensor_signal = (linear_12bit / wb) * (SIGNAL_RANGE / INVISP_CANON_WHITE)
    return sensor_signal.clamp(0, SIGNAL_RANGE) + BLACK_LEVEL


def sample_bayer(sensor_rgb: torch.Tensor) -> torch.Tensor:
    """RGGB sampling at fixed phase, without warping the mosaic."""
    n, channels, height, width = sensor_rgb.shape
    if channels != 3 or height % 2 or width % 2:
        raise ValueError("sensor RGB must have three channels and even spatial dimensions")
    bayer = torch.empty((n, height, width), dtype=sensor_rgb.dtype, device=sensor_rgb.device)
    bayer[:, 0::2, 0::2] = sensor_rgb[:, 0, 0::2, 0::2]
    bayer[:, 0::2, 1::2] = sensor_rgb[:, 1, 0::2, 1::2]
    bayer[:, 1::2, 0::2] = sensor_rgb[:, 1, 1::2, 0::2]
    bayer[:, 1::2, 1::2] = sensor_rgb[:, 2, 1::2, 1::2]
    return bayer


def pack_student(noisy_bayer_dn: torch.Tensor) -> torch.Tensor:
    """Black subtraction, level normalization and Canon WB; preserve negatives."""
    normalized = (noisy_bayer_dn - BLACK_LEVEL) / SIGNAL_RANGE
    packed = torch.stack((normalized[:, 0::2, 0::2], normalized[:, 0::2, 1::2],
                          normalized[:, 1::2, 0::2], normalized[:, 1::2, 1::2]), dim=1)
    return packed * packed.new_tensor(WHITE_BALANCE)[None, :, None, None]


def clean_gray(sensor_rgb: torch.Tensor) -> torch.Tensor:
    """Full Bayer-resolution linear gray after the same level and WB mapping."""
    wb = sensor_rgb.new_tensor((WHITE_BALANCE[0], WHITE_BALANCE[1], WHITE_BALANCE[3]))[None, :, None, None]
    linear = (sensor_rgb - BLACK_LEVEL) / SIGNAL_RANGE * wb
    weights = sensor_rgb.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]
    return (linear * weights).sum(dim=1, keepdim=True)
