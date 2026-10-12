import importlib.util
import json

import pytest
import torch

from rawfeat.model import RawFeatureExtractor
from rawfeat.onnx_export import export_onnx
from rawfeat.export import DEPLOYMENT_FORMAT


def _deployment_checkpoint(path):
    model = RawFeatureExtractor().eval().deploy_copy()
    metadata = {
        "source_checkpoint": "/tmp/checkpoint.pt",
        "source_checkpoint_sha256": "0" * 64,
        "source_step": 96000,
        "code_identity_sha256": "1" * 64,
        "protocol": "rawfeat.hpatches.synthetic_raw.v1",
        "manifest_sha256": "2" * 64,
        "fused": True,
        "gray_removed": True,
        "auxiliary": False,
    }
    torch.save({"format": DEPLOYMENT_FORMAT, "metadata": metadata,
                "student": model.state_dict()}, path)


def test_onnx_export_records_fused_gray_free_contract(tmp_path):
    checkpoint = tmp_path / "deployment.pt"
    output = tmp_path / "deployment.onnx"
    _deployment_checkpoint(checkpoint)
    report = export_onnx(checkpoint, output, input_height=32, input_width=40,
                         validate=False)
    assert report["fused"] and report["gray_removed"] and not report["auxiliary"]
    assert report["structure"]["branch_block_indices"] == []
    assert report["structure"]["gray_modules"] == []
    assert report["outputs"][0]["name"] == "logits"
    assert output.is_file()
    sidecar = json.loads((tmp_path / "deployment.onnx.metadata.json").read_text())
    assert sidecar["format"] == "rawfeat.onnx.v1"
    assert sidecar["dynamic_axes"]["packed"]["2"] == "packed_height"


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx and onnxruntime are optional export validation dependencies",
)
def test_onnx_export_runtime_matches_pytorch(tmp_path):
    checkpoint = tmp_path / "deployment.pt"
    _deployment_checkpoint(checkpoint)
    report = export_onnx(checkpoint, tmp_path / "runtime.onnx",
                         input_height=32, input_width=40)
    assert report["validation"]["graph"]["checked"]
    assert report["validation"]["runtime"]["checked"]
    assert max(report["validation"]["runtime"]["max_abs"].values()) < 1e-4


def test_onnx_export_requires_stride_compatible_packed_shape(tmp_path):
    checkpoint = tmp_path / "deployment.pt"
    _deployment_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="divisible by 4"):
        export_onnx(checkpoint, tmp_path / "bad.onnx", input_height=30, input_width=40,
                    validate=False)
