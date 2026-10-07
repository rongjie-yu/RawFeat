"""Frozen official MagicLeap SuperPoint checkpoint with raw 65-class logits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def load_teacher(repo: str | Path, device: torch.device | str = "cuda") -> torch.nn.Module:
    repo = Path(repo)
    source = repo / "demo_superpoint.py"
    weight = repo / "superpoint_v1.pth"
    if not source.is_file() or not weight.is_file():
        raise FileNotFoundError(f"official MagicLeap source and checkpoint required in {repo}")
    spec = importlib.util.spec_from_file_location("magicleap_superpoint", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official SuperPoint source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    teacher = module.SuperPointNet()
    teacher.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    teacher.rawfeat_checkpoint_path = str(weight)
    teacher.requires_grad_(False).eval()
    return teacher.to(device)
