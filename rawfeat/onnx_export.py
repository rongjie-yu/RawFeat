"""Export a fused, gray-free RawFeat deployment model to ONNX.

The training model returns a dictionary and contains a gray reconstruction head.
ONNX deployment exposes only the two inference tensors, so this module loads a
training or ``rawfeat.deployment.v1`` checkpoint, applies the existing RepVGG
fusion path, wraps the model with a tuple output, and optionally checks the
serialized graph with ONNX Runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .export import (
    DEPLOYMENT_FORMAT,
    DEPLOYMENT_OUTPUTS,
    assert_deployment_structure,
    load_export,
)
from .identity import check_schema
from .model import RawFeatureExtractor
from .validation import file_digest


ONNX_FORMAT = "rawfeat.onnx.v1"
DEFAULT_OPSET = 17
DEFAULT_BATCH_SIZE = 1
DEFAULT_INPUT_HEIGHT = 240
DEFAULT_INPUT_WIDTH = 320


class OnnxOutputWrapper(nn.Module):
    """Turn the deployment model's output dictionary into a stable tuple."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(packed)
        return output["logits"], output["descriptors"]


def _onnx_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir() or not candidate.suffix:
        return candidate / "deployment.onnx"
    if candidate.suffix.lower() != ".onnx":
        raise ValueError(f"ONNX output must end in .onnx: {candidate}")
    return candidate


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def _source_kind_and_step(stored: dict[str, Any]) -> tuple[str, int | None]:
    if stored.get("format") == DEPLOYMENT_FORMAT:
        metadata = stored.get("metadata", {})
        step = metadata.get("source_step")
        return "deployment", None if step is None else int(step)
    step = stored.get("step")
    return "training", None if step is None else int(step)


def load_deployment_model(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[RawFeatureExtractor, dict[str, Any]]:
    """Load a checkpoint and return the fused, gray-free deployment model.

    ``checkpoint`` may be a normal staged training checkpoint or an existing
    ``rawfeat.deployment.v1`` artifact.  The returned model is always checked
    with :func:`assert_deployment_structure` before it can be exported.
    """

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    target = torch.device(device)
    stored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_kind, source_step = _source_kind_and_step(stored)
    if source_kind == "deployment":
        model = load_export(checkpoint_path, target)
    else:
        check_schema(stored)
        model = RawFeatureExtractor().eval()
        model.load_state_dict(stored["student"], strict=True)
        model = model.to(target).eval().deploy_copy().to(target).eval()
    structure = assert_deployment_structure(model)
    return model, {
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_digest(checkpoint_path),
        "source_kind": source_kind,
        "source_step": source_step,
        "structure": structure,
    }


def _validate_dimensions(batch_size: int, input_height: int, input_width: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if input_height <= 0 or input_width <= 0:
        raise ValueError("input height and width must be positive")
    if input_height % 4 or input_width % 4:
        raise ValueError(
            "packed input height and width must be divisible by 4 "
            "(the two stride-2 stages reduce them to the output grid)"
        )


def _dynamic_axes(dynamic: bool) -> dict[str, dict[int, str]] | None:
    if not dynamic:
        return None
    return {
        "packed": {0: "batch", 2: "packed_height", 3: "packed_width"},
        "logits": {0: "batch", 2: "grid_height", 3: "grid_width"},
        "descriptors": {0: "batch", 2: "grid_height", 3: "grid_width"},
    }


def _graph_check(path: Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as error:  # pragma: no cover - depends on deployment environment
        raise RuntimeError(
            "ONNX graph validation requires the optional 'onnx' package; "
            "install it with `python -m pip install onnx`"
        ) from error
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    input_names = [value.name for value in graph.graph.input]
    output_names = [value.name for value in graph.graph.output]
    if input_names != ["packed"]:
        raise RuntimeError(f"unexpected ONNX inputs: {input_names}")
    if output_names != list(DEPLOYMENT_OUTPUTS):
        raise RuntimeError(f"unexpected ONNX outputs: {output_names}")
    gray_tokens = ("gray", "grey")
    graph_text = "\n".join(
        [node.name for node in graph.graph.node]
        + [value for node in graph.graph.node for value in list(node.input) + list(node.output)]
    ).lower()
    if any(token in graph_text for token in gray_tokens):
        raise RuntimeError("ONNX graph contains a gray branch")
    return {
        "checked": True,
        "node_count": len(graph.graph.node),
        "input_names": input_names,
        "output_names": output_names,
    }


def _runtime_check(
    path: Path,
    wrapper: OnnxOutputWrapper,
    example: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - depends on deployment environment
        return {"checked": False, "available": False, "reason": "onnxruntime is not installed"}
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = wrapper(example.to(device))
    actual = session.run(list(DEPLOYMENT_OUTPUTS), {"packed": example.detach().cpu().numpy()})
    max_abs: dict[str, float] = {}
    mean_abs: dict[str, float] = {}
    for name, expected_tensor, actual_array in zip(DEPLOYMENT_OUTPUTS, expected, actual):
        expected_array = expected_tensor.detach().cpu().numpy()
        if expected_array.shape != actual_array.shape:
            raise RuntimeError(
                f"ONNX Runtime shape mismatch for {name}: "
                f"{actual_array.shape} != {expected_array.shape}"
            )
        delta = np.abs(actual_array - expected_array)
        max_abs[name] = float(delta.max()) if delta.size else 0.0
        mean_abs[name] = float(delta.mean()) if delta.size else 0.0
        np.testing.assert_allclose(actual_array, expected_array, rtol=1e-3, atol=1e-5)
    return {
        "checked": True,
        "available": True,
        "providers": session.get_providers(),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


@torch.inference_mode()
def export_onnx(
    checkpoint: str | Path,
    output: str | Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    input_width: int = DEFAULT_INPUT_WIDTH,
    opset: int = DEFAULT_OPSET,
    device: torch.device | str = "cpu",
    dynamic: bool = True,
    validate: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export and validate a fused gray-free RawFeat ONNX model.

    Spatial dimensions describe the four-channel packed Bayer tensor, not the
    original single-channel Bayer image.  They therefore need to be divisible
    by four, and the deployment outputs have spatial size ``input / 4``.
    """

    if opset < 11:
        raise ValueError("opset must be at least 11")
    _validate_dimensions(batch_size, input_height, input_width)
    output_path = _onnx_path(output).resolve()
    metadata_path = _metadata_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"ONNX output already exists: {output_path}; pass overwrite=True")
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"ONNX metadata already exists: {metadata_path}; pass overwrite=True")

    target = torch.device(device)
    model, source = load_deployment_model(checkpoint, target)
    wrapper = OnnxOutputWrapper(model).eval().to(target)
    example = torch.randn(batch_size, 4, input_height, input_width, device=target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        torch.onnx.export(
            wrapper,
            (example,),
            str(temporary),
            input_names=["packed"],
            output_names=list(DEPLOYMENT_OUTPUTS),
            opset_version=opset,
            dynamic_axes=_dynamic_axes(dynamic),
            do_constant_folding=True,
            dynamo=False,
        )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    validation: dict[str, Any] = {"graph": {"checked": False}, "runtime": {"checked": False}}
    if validate:
        validation["graph"] = _graph_check(output_path)
        validation["runtime"] = _runtime_check(output_path, wrapper, example, target)
    dynamic_axes = _dynamic_axes(dynamic)
    metadata = {
        "format": ONNX_FORMAT,
        **source,
        "onnx_path": str(output_path),
        "onnx_sha256": file_digest(output_path),
        "opset": int(opset),
        "precision": "FP32",
        "fused": True,
        "gray_removed": True,
        "auxiliary": False,
        "inputs": [{
            "name": "packed",
            "shape": [batch_size, 4, input_height, input_width],
            "layout": "NCHW",
            "spatial_unit": "packed Bayer",
        }],
        "outputs": [
            {"name": "logits", "channels": 65, "spatial_scale": "packed / 4"},
            {"name": "descriptors", "channels": 128, "spatial_scale": "packed / 4"},
        ],
        "dynamic_axes": dynamic_axes,
        "validation": validation,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_INPUT_HEIGHT",
    "DEFAULT_INPUT_WIDTH",
    "DEFAULT_OPSET",
    "ONNX_FORMAT",
    "OnnxOutputWrapper",
    "export_onnx",
    "load_deployment_model",
]
