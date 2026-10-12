#!/usr/bin/env python3
"""Export a RawFeat checkpoint as a fused, gray-free ONNX deployment model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Allow ``python scripts/export_onnx.py`` from a source checkout without an
# editable package install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rawfeat.onnx_export import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_OPSET,
    export_onnx,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a fused RepVGG, gray-free RawFeat model to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="training checkpoint or deployment.pt")
    parser.add_argument("--output", required=True, help="destination .onnx file or directory")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_HEIGHT,
                        help="packed Bayer input height")
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_WIDTH,
                        help="packed Bayer input width")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--device", default="cpu", help="PyTorch export device, e.g. cpu or cuda:0")
    parser.add_argument("--static-shape", action="store_true",
                        help="omit dynamic batch and spatial axes")
    parser.add_argument("--skip-validation", action="store_true",
                        help="skip ONNX checker and optional ONNX Runtime numerical validation")
    parser.add_argument("--force", action="store_true", help="overwrite the ONNX file and metadata sidecar")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = export_onnx(
        args.checkpoint,
        args.output,
        batch_size=args.batch_size,
        input_height=args.input_height,
        input_width=args.input_width,
        opset=args.opset,
        device=args.device,
        dynamic=not args.static_shape,
        validate=not args.skip_validation,
        overwrite=args.force,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
