import numpy as np
import pytest
import torch

from rawfeat.export import (
    DEPLOYMENT_FORMAT,
    assert_deployment_structure,
    deployment_structure,
    load_export,
    _timing_stats,
    verify_discrete_case,
)
from rawfeat.model import RawFeatureExtractor


def features(points, scores=None):
    return {**{f'points_{v}': np.array(points, dtype=np.float32) for v in ('a','b')},
            **{f'descriptors_{v}': np.eye(len(points), dtype=np.float32) for v in ('a','b')},
            **{f'match_indices_{v}': np.arange(len(points)) for v in ('a','b')}}


def test_deployment_structure_fuses_all_blocks_and_removes_gray():
    training = RawFeatureExtractor().eval()
    deployed = training.deploy_copy()
    structure = deployment_structure(deployed)
    assert structure["repvgg_block_count"] == 11
    assert structure["fused_block_count"] == 11
    assert structure["branch_block_indices"] == []
    assert structure["gray_modules"] == []
    assert structure["is_deployed"]
    assert_deployment_structure(deployed)
    assert set(deployed(torch.randn(1, 4, 32, 40))) == {"logits", "descriptors"}


def test_load_export_strictly_constructs_gray_free_deployment(tmp_path):
    training = RawFeatureExtractor().eval()
    deployed = training.deploy_copy()
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
    path = tmp_path / "deployment.pt"
    torch.save({"format": DEPLOYMENT_FORMAT, "metadata": metadata,
                "student": deployed.state_dict()}, path)
    loaded = load_export(path, "cpu")
    assert assert_deployment_structure(loaded)["is_deployed"]
    assert set(loaded(torch.randn(1, 4, 32, 40))) == {"logits", "descriptors"}


def test_load_export_rejects_training_payload(tmp_path):
    path = tmp_path / "training.pt"
    torch.save({"student": RawFeatureExtractor().state_dict()}, path)
    with pytest.raises(RuntimeError, match="rawfeat.deployment.v1"):
        load_export(path, "cpu")


def test_latency_summary_fields_are_distribution_and_batch_based(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test-gpu")
    summary = _timing_stats(
        [1.0, 2.0, 3.0], stage="network_forward", condition="ratio4", batch_size=4,
        warmup=20, device=torch.device("cuda:0"), input_shape=[4, 4, 240, 320],
        peak_allocated=10, peak_reserved=20,
    )
    assert {"mean_ms", "median_ms", "p10_ms", "p90_ms", "std_ms", "images_per_second",
            "peak_allocated_bytes", "peak_reserved_bytes"} <= summary.keys()
    assert summary["batch_size"] == 4 and summary["repeats"] == 3


def test_fusion_accepts_measured_near_tie_order_and_rejects_large_inversion():
    a=features([[8,8],[20,20]]);b={k: v[::-1].copy() if k.startswith(('points','descriptors')) else v.copy() for k,v in a.items()}
    old=np.zeros((2,32,32),np.float32);new=old.copy()
    old[:,8,8]=.02+1e-8;old[:,20,20]=.02
    new[:,8,8]=.02;new[:,20,20]=.02+1e-8
    r=verify_discrete_case(a,b,old,new)
    assert r['point_order_changed'] and not r['point_set_changed'] and not r['match_coordinate_set_changed']
    old[:,8,8]=.03;new=old.copy()
    with pytest.raises(RuntimeError,match='non-tied score ranking'):verify_discrete_case(a,b,old,new)


def test_fusion_accepts_neighbor_nms_tie_but_rejects_unexplained_shift():
    a=features([[8,8]]);b=features([[8,9]])
    old=np.zeros((2,32,32),np.float32);new=old.copy()
    old[:,8,8]=.02+1e-8;old[:,9,8]=.02
    new[:,8,8]=.02;new[:,9,8]=.02+1e-8
    assert verify_discrete_case(a,b,old,new)['point_set_changed']
    old[:,8,8]=.03;new=old.copy()
    with pytest.raises(RuntimeError,match='away from NMS'):verify_discrete_case(a,b,old,new)
