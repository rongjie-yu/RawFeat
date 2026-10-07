
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from thop import profile

from .features import extract_features, score_map
from .identity import check_schema
from .metrics import mutual_nearest
from .model import RawFeatureExtractor, RepVGGBlock
from .runtime import configure_fp32
from .sensor import pack_student
from .validation import cache_path as coco_cache_path
from .validation import file_digest, read_manifest as read_coco_manifest


DEPLOYMENT_FORMAT = "rawfeat.deployment.v1"
DEPLOYMENT_OUTPUTS = ("logits", "descriptors")
REPVGG_BLOCK_COUNT = 11
DEFAULT_LATENCY_WARMUP = 20
DEFAULT_LATENCY_REPEATS = 100


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _deployment_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir() or not candidate.suffix:
        return candidate / "deployment.pt"
    return candidate


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def deployment_structure(model: torch.nn.Module) -> dict[str, Any]:
    blocks = [module for module in model.modules() if isinstance(module, RepVGGBlock)]
    branch_names = ("conv3", "conv1", "identity")
    branch_blocks = [
        index for index, block in enumerate(blocks)
        if any(hasattr(block, name) for name in branch_names)
    ]
    fused_blocks = [
        index for index, block in enumerate(blocks)
        if hasattr(block, "fused") and isinstance(block.fused, torch.nn.Conv2d)
        and block.fused.bias is not None
        and tuple(block.fused.kernel_size) == (3, 3)
    ]
    gray_modules = [
        name for name in ("gray_reduce", "gray_conv", "gray_output")
        if hasattr(model, name)
    ]
    return {
        "repvgg_block_count": len(blocks),
        "expected_repvgg_block_count": REPVGG_BLOCK_COUNT,
        "fused_block_count": len(fused_blocks),
        "fused_block_indices": fused_blocks,
        "branch_block_indices": branch_blocks,
        "gray_modules": gray_modules,
        "auxiliary": bool(getattr(model, "auxiliary", False)),
        "outputs": list(DEPLOYMENT_OUTPUTS),
        "is_deployed": (
            len(blocks) == REPVGG_BLOCK_COUNT
            and len(fused_blocks) == REPVGG_BLOCK_COUNT
            and not branch_blocks
            and not gray_modules
            and not bool(getattr(model, "auxiliary", False))
        ),
    }


def assert_deployment_structure(model: torch.nn.Module) -> dict[str, Any]:
    structure = deployment_structure(model)
    if not structure["is_deployed"]:
        raise RuntimeError(f"model is not a fused deployment model: {structure}")
    return structure


def _code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "rawfeat/model.py",
        "rawfeat/export.py",
        "rawfeat/features.py",
        "rawfeat/inference.py",
        "rawfeat/hpatches.py",
    )
    files = {path: file_digest(root / path) for path in relative_paths}
    return {"files": files, "sha256": _digest_json(files)}


def _protocol_identity(manifest_path: str | Path, cache_root: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    raw_manifest = json.loads(manifest_file.read_text())
    protocol = raw_manifest.get("protocol")
    if protocol == "rawfeat.hpatches.synthetic_raw.v1":
        from .hpatches import read_manifest

        manifest, manifest_sha256 = read_manifest(manifest_file)
    else:
        manifest, manifest_sha256 = read_coco_manifest(manifest_file)
    identity_file = Path(cache_root) / "identity.json"
    identity = json.loads(identity_file.read_text()) if identity_file.is_file() else None
    return {
        "protocol": manifest["protocol"],
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_sha256,
        "cache_identity_sha256": None if identity is None else _digest_json(identity),
        "cache_identity": identity,
    }


def _load_sample(
    manifest_path: str | Path,
    cache_root: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("protocol") == "rawfeat.hpatches.synthetic_raw.v1":
        from .hpatches import cache_path, iter_manifest_images

        images = iter_manifest_images(manifest)
        image = images[0]
        entry = torch.load(
            cache_path(cache_root, image["id"], "clean"),
            map_location="cpu",
            weights_only=True,
        )
        bayer = entry["noisy_bayer_dn"][None].to(device)
        return pack_student(bayer), bayer, {"image_id": image["id"], "condition": "clean"}
    entry = torch.load(
        coco_cache_path(cache_root, 0, "clean"),
        map_location="cpu",
        weights_only=True,
    )
    packed = entry["packed"].to(device)
    bayer = entry.get("noisy_bayer_dn")
    if bayer is not None:
        bayer = bayer[:1].to(device)
    return packed[:1], bayer, {"image_position": 0, "condition": "clean"}


def load_export(path: str | Path, device: torch.device | str = "cuda") -> RawFeatureExtractor:
    """Load only the strict fused deployment format."""
    export_path = _deployment_path(path)
    stored = torch.load(export_path, map_location="cpu", weights_only=False)
    if stored.get("format") != DEPLOYMENT_FORMAT:
        raise RuntimeError("deployment file must use rawfeat.deployment.v1")
    metadata = stored.get("metadata", {})
    required = {
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_step",
        "code_identity_sha256",
        "protocol",
        "manifest_sha256",
        "fused",
        "gray_removed",
        "auxiliary",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise RuntimeError(f"deployment metadata is incomplete: {missing}")
    if not metadata["fused"] or not metadata["gray_removed"] or metadata["auxiliary"]:
        raise RuntimeError("deployment metadata does not describe a fused gray-free model")
    model = RawFeatureExtractor(auxiliary=False, deploy=True)
    model.load_state_dict(stored["student"], strict=True)
    assert_deployment_structure(model)
    return model.eval().to(device)


def _run_timed(
    device: torch.device,
    function: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    cuda_events: bool,
) -> tuple[list[float], int, int]:
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for deployment latency measurement")
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    if cuda_events:
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(device))
            function()
            end.record(torch.cuda.current_stream(device))
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    else:
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            function()
            torch.cuda.synchronize(device)
            timings.append(1000.0 * (time.perf_counter() - started))
    return timings, int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device))


def _timing_stats(
    timings: list[float],
    *,
    stage: str,
    condition: str,
    batch_size: int,
    warmup: int,
    device: torch.device,
    input_shape: list[int],
    peak_allocated: int,
    peak_reserved: int,
) -> dict[str, Any]:
    if not timings:
        raise ValueError("latency timing requires at least one measured repetition")
    values = np.asarray(timings, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "stage": stage,
        "condition": condition,
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": int(len(values)),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "images_per_second": float(batch_size / (mean_ms / 1000.0)) if mean_ms else None,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "device": torch.cuda.get_device_name(device),
        "input_shape": input_shape,
    }


def _timed(
    device: torch.device,
    function: Callable[[], Any],
    warmup: int = DEFAULT_LATENCY_WARMUP,
    repeats: int = DEFAULT_LATENCY_REPEATS,
) -> dict[str, Any]:
    timings, allocated, reserved = _run_timed(
        device, function, warmup=warmup, repeats=repeats, cuda_events=True
    )
    return _timing_stats(
        timings,
        stage="network_forward",
        condition="unspecified",
        batch_size=1,
        warmup=warmup,
        device=device,
        input_shape=[],
        peak_allocated=allocated,
        peak_reserved=reserved,
    )


def _feature_for_output(output: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    features = extract_features(output["logits"], output["descriptors"])
    return [
        {name: value.detach().cpu() for name, value in feature.items()}
        for feature in features
    ]


def benchmark_model_latency(
    model: torch.nn.Module,
    batches_by_condition: dict[str, dict[int, list[dict[str, Any]]]],
    pair_inputs_by_condition: dict[str, list[dict[str, Any]]],
    device: torch.device,
    *,
    warmup: int = DEFAULT_LATENCY_WARMUP,
    repeats: int = DEFAULT_LATENCY_REPEATS,
) -> dict[str, Any]:
    """Measure each stage with one record per executed batch, never per-image copies."""
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for deployment latency measurement")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    measurements: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for condition, batches_by_size in batches_by_condition.items():
        for batch_size, batches in sorted(batches_by_size.items()):
            if not batches:
                raise ValueError(f"no latency batches for {condition}, batch {batch_size}")
            for stage in ("network_forward", "feature_extraction", "bayer_to_features"):
                for _ in range(warmup):
                    batch = batches[0]
                    if stage == "network_forward":
                        model(batch["packed"])
                    elif stage == "feature_extraction":
                        _feature_for_output(model(batch["packed"]))
                    else:
                        _feature_for_output(model(pack_student(batch["bayer"])))
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                stage_timings: list[float] = []
                for repeat_index in range(repeats):
                    batch_index = repeat_index % len(batches)
                    if stage == "network_forward":
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record(torch.cuda.current_stream(device))
                        model(batches[batch_index]["packed"])
                        end.record(torch.cuda.current_stream(device))
                        end.synchronize()
                        elapsed = float(start.elapsed_time(end))
                    else:
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        if stage == "feature_extraction":
                            _feature_for_output(model(batches[batch_index]["packed"]))
                        else:
                            _feature_for_output(model(pack_student(batches[batch_index]["bayer"])))
                        torch.cuda.synchronize(device)
                        elapsed = 1000.0 * (time.perf_counter() - started)
                    stage_timings.append(elapsed)
                    measurements.append({
                        "stage": stage,
                        "condition": condition,
                        "batch_size": batch_size,
                        "repeat_index": repeat_index,
                        "batch_index": batch_index,
                        "image_ids": batches[batch_index]["image_ids"],
                        "elapsed_ms": elapsed,
                    })
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
                summaries.append(_timing_stats(
                    stage_timings,
                    stage=stage,
                    condition=condition,
                    batch_size=batch_size,
                    warmup=warmup,
                    device=device,
                    input_shape=list(batches[0]["packed"].shape),
                    peak_allocated=peak_allocated,
                    peak_reserved=peak_reserved,
                ))
        pair_inputs = pair_inputs_by_condition.get(condition, [])
        if pair_inputs:
            for _ in range(warmup):
                pair = pair_inputs[0]
                _pair_pipeline(pair)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            stage_timings = []
            for repeat_index in range(repeats):
                pair_index = repeat_index % len(pair_inputs)
                started = time.perf_counter()
                _pair_pipeline(pair_inputs[pair_index])
                elapsed = 1000.0 * (time.perf_counter() - started)
                stage_timings.append(elapsed)
                measurements.append({
                    "stage": "pair_pipeline",
                    "condition": condition,
                    "batch_size": 2,
                    "repeat_index": repeat_index,
                    "batch_index": pair_index,
                    "image_ids": pair_inputs[pair_index]["image_ids"],
                    "elapsed_ms": elapsed,
                })
            summaries.append(_timing_stats(
                stage_timings,
                stage="pair_pipeline",
                condition=condition,
                batch_size=2,
                warmup=warmup,
                device=device,
                input_shape=[],
                peak_allocated=int(torch.cuda.max_memory_allocated(device)),
                peak_reserved=int(torch.cuda.max_memory_reserved(device)),
            ))
    return {"measurements": measurements, "summary": summaries}


def _pair_pipeline(pair: dict[str, Any]) -> None:
    import cv2

    feature_a = pair["feature_a"]
    feature_b = pair["feature_b"]
    indices_a, indices_b = mutual_nearest(feature_a["descriptors"], feature_b["descriptors"])
    points_a = feature_a["points"][indices_a]
    points_b = feature_b["points"][indices_b]
    if len(points_a) >= 4:
        cv2.setRNGSeed(int(pair.get("seed", 2027)))
        cv2.findHomography(
            points_a,
            points_b,
            cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=10000,
            confidence=0.999,
        )


def compare_feature_pair(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    old_scores: np.ndarray,
    new_scores: np.ndarray,
    *,
    threshold: float = 0.005,
    nms_radius: int = 4,
    max_points: int = 1024,
) -> dict[str, Any]:
    """Compare selection and MNN changes, requiring every change to have a boundary cause."""
    score_delta = np.asarray(new_scores, dtype=np.float32) - np.asarray(old_scores, dtype=np.float32)
    score_error = float(np.abs(score_delta).max()) if score_delta.size else 0.0
    score_mean_error = float(np.abs(score_delta).mean()) if score_delta.size else 0.0
    tolerance = 2.0 * score_error + np.finfo(np.float32).eps
    point_order_changed = not np.array_equal(before["points"], after["points"])
    before_indices = {tuple(point): index for index, point in enumerate(before["points"])}
    after_indices = {tuple(point): index for index, point in enumerate(after["points"])}
    point_set_changed = before_indices.keys() != after_indices.keys()
    if point_order_changed and not point_set_changed:
        ordered_values = np.asarray([
            old_scores[int(point[1]), int(point[0])] for point in after["points"]
        ])
        if len(ordered_values) > 1:
            inversion = float(np.max(ordered_values[1:] - np.minimum.accumulate(ordered_values[:-1])))
            if inversion > tolerance:
                raise RuntimeError("fusion changed a non-tied score ranking")
        reasons = {"score_order_near_tie"}
    else:
        reasons = set()
    descriptor_max = descriptor_mean = 0.0
    common = sorted(before_indices.keys() & after_indices.keys())
    score_max = score_mean = 0.0
    if common:
        before_scores = np.asarray([
            before.get("scores", old_scores)[before_indices[point]]
            if "scores" in before else old_scores[int(point[1]), int(point[0])]
            for point in common
        ])
        after_scores = np.asarray([
            after.get("scores", new_scores)[after_indices[point]]
            if "scores" in after else new_scores[int(point[1]), int(point[0])]
            for point in common
        ])
        score_delta = np.abs(after_scores - before_scores)
        score_max = float(score_delta.max())
        score_mean = float(score_delta.mean())
        before_descriptors = before["descriptors"][[before_indices[point] for point in common]]
        after_descriptors = after["descriptors"][[after_indices[point] for point in common]]
        descriptor_delta = after_descriptors - before_descriptors
        descriptor_max = float(np.abs(descriptor_delta).max())
        descriptor_mean = float(np.abs(descriptor_delta).mean())
    for original_indices, changed_indices, own_scores, other_scores in (
        (before_indices, after_indices, old_scores, new_scores),
        (after_indices, before_indices, new_scores, old_scores),
    ):
        for point in original_indices.keys() - changed_indices.keys():
            x, y = map(int, point)
            value = float(own_scores[y, x])
            neighbors = [
                candidate for candidate in changed_indices
                if max(abs(candidate[0] - x), abs(candidate[1] - y)) <= nms_radius
            ]
            near_nms = any(
                abs(float(own_scores[int(candidate[1]), int(candidate[0])]) - value) <= tolerance
                for candidate in neighbors
            )
            near_threshold = abs(value - threshold) <= tolerance
            selected_other_values = [
                float(other_scores[int(candidate[1]), int(candidate[0])])
                for candidate in changed_indices
            ]
            near_topk = (
                len(changed_indices) >= max_points
                and bool(selected_other_values)
                and abs(value - min(selected_other_values)) <= tolerance
            )
            if near_nms:
                reasons.add("nms_near_tie")
            if near_threshold:
                reasons.add("threshold_boundary")
            if near_topk:
                reasons.add("topk_boundary")
            if not (near_nms or near_threshold or near_topk):
                raise RuntimeError("fusion changed a keypoint away from NMS/threshold/TopK boundaries")
    mnn_changed = False
    before_mnn = after_mnn = set()
    if len(before["descriptors"]) and len(after["descriptors"]):
        before_a, before_b = mutual_nearest(
            before["descriptors"], before["descriptors"]
        )
        after_a, after_b = mutual_nearest(
            after["descriptors"], after["descriptors"]
        )
        before_mnn = {(tuple(before["points"][a]), tuple(before["points"][b])) for a, b in zip(before_a, before_b)}
        after_mnn = {(tuple(after["points"][a]), tuple(after["points"][b])) for a, b in zip(after_a, after_b)}
        mnn_changed = before_mnn != after_mnn
    return {
        "score_map_max_abs_error": score_error,
        "score_map_mean_abs_error": score_mean_error,
        "point_order_changed": bool(point_order_changed),
        "point_set_changed": bool(point_set_changed),
        "detected_before": int(len(before["points"])),
        "detected_after": int(len(after["points"])),
        "detected_delta": int(len(after["points"]) - len(before["points"])),
        "descriptor_max_abs_error": descriptor_max,
        "descriptor_mean_abs_error": descriptor_mean,
        "selected_score_max_abs_error": score_max,
        "selected_score_mean_abs_error": score_mean,
        "mnn_coordinate_set_changed": bool(mnn_changed),
        "discrete_change_reasons": sorted(reasons),
        "numeric_boundaries_verified": True,
    }


def verify_discrete_case(
    before: dict[str, Any],
    after: dict[str, Any],
    old_scores: np.ndarray,
    new_scores: np.ndarray,
) -> dict[str, Any]:
    """Compatibility wrapper for the fixed-COCO export comparison."""
    result = {
        "score_error": float(np.abs(old_scores - new_scores).max()),
        "point_order_changed": False,
        "point_set_changed": False,
        "match_coordinate_set_changed": False,
        "common_descriptor_max_error": 0.0,
        "numeric_boundaries_verified": True,
    }
    tolerance = 2.0 * result["score_error"] + np.finfo(np.float32).eps
    for view, index in (("a", 0), ("b", 1)):
        old = {
            "points": np.asarray(before[f"points_{view}"]),
            "descriptors": np.asarray(before[f"descriptors_{view}"]),
        }
        new = {
            "points": np.asarray(after[f"points_{view}"]),
            "descriptors": np.asarray(after[f"descriptors_{view}"]),
        }
        comparison = compare_feature_pair(
            old,
            new,
            old_scores[index],
            new_scores[index],
        )
        if comparison["point_order_changed"]:
            result["point_order_changed"] = True
        if comparison["point_set_changed"]:
            result["point_set_changed"] = True
        result["common_descriptor_max_error"] = max(
            result["common_descriptor_max_error"],
            comparison["descriptor_max_abs_error"],
        )
        if comparison["discrete_change_reasons"]:
            result.setdefault("discrete_change_reasons", set()).update(
                comparison["discrete_change_reasons"]
            )
    before_matches = {
        (*before["points_a"][i], *before["points_b"][j])
        for i, j in zip(before["match_indices_a"], before["match_indices_b"])
    }
    after_matches = {
        (*after["points_a"][i], *after["points_b"][j])
        for i, j in zip(after["match_indices_a"], after["match_indices_b"])
    }
    result["match_coordinate_set_changed"] = before_matches != after_matches
    if isinstance(result.get("discrete_change_reasons"), set):
        result["discrete_change_reasons"] = sorted(result["discrete_change_reasons"])
    return result


@torch.no_grad()
def export_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    cache_root: str | Path,
    manifest: str | Path,
    device: torch.device | str = "cuda:0",
    max_images: int = 256,
    benchmark_repeats: int = DEFAULT_LATENCY_REPEATS,
    warmup: int = DEFAULT_LATENCY_WARMUP,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export one strict deployment artifact and validate its in-memory reload."""
    del max_images
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deployment export")
    if benchmark_repeats <= 0 or warmup < 0:
        raise ValueError("benchmark repeats must be positive and warmup must be non-negative")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("deployment export requires a CUDA device")
    torch.cuda.set_device(device)
    configure_fp32()
    checkpoint_path = Path(checkpoint).resolve()
    output_path = _deployment_path(output).resolve()
    if checkpoint_path == output_path:
        raise ValueError("deployment output must be different from the training checkpoint")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"deployment output already exists: {output_path}; pass --force to replace it")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    check_schema(source)
    if source["step"] <= source["config"]["schedule"]["detection_updates"]:
        raise ValueError("export requires a B/C checkpoint with an active trained descriptor")
    training = RawFeatureExtractor().eval()
    training.load_state_dict(source["student"], strict=True)
    training = training.to(device)
    deployed = training.deploy_copy().eval().to(device)
    structure = assert_deployment_structure(deployed)
    protocol = _protocol_identity(manifest, cache_root)
    code_identity = _code_identity()
    metadata = {
        "format": DEPLOYMENT_FORMAT,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_digest(checkpoint_path),
        "source_step": int(source["step"]),
        "source_schema": source["schema"],
        "code_identity": code_identity,
        "code_identity_sha256": code_identity["sha256"],
        "protocol": protocol["protocol"],
        "manifest_path": protocol["manifest_path"],
        "manifest_sha256": protocol["manifest_sha256"],
        "cache_identity_sha256": protocol["cache_identity_sha256"],
        "fused": True,
        "gray_removed": True,
        "auxiliary": False,
        "gray_computed": False,
        "repvgg_block_count": structure["repvgg_block_count"],
        "fused_repvgg_block_count": structure["fused_block_count"],
        "outputs": list(DEPLOYMENT_OUTPUTS),
        "precision": "FP32",
    }
    sample: dict[str, Any] = {"available": False}
    if Path(cache_root).is_dir():
        packed, bayer, sample_identity = _load_sample(manifest, cache_root, device)
        before = training(packed)
        after = deployed(packed)
        differences = {}
        for name in DEPLOYMENT_OUTPUTS:
            delta = (before[name] - after[name]).abs()
            differences[name] = {
                "max_abs": float(delta.max().item()),
                "mean_abs": float(delta.mean().item()),
            }
            torch.testing.assert_close(before[name], after[name], atol=1e-4, rtol=1e-4)
        sample = {
            "available": True,
            **sample_identity,
            "input_shape": list(packed.shape),
            "dense_output_difference": differences,
        }
    payload = {
        "format": DEPLOYMENT_FORMAT,
        "metadata": metadata,
        "student": {name: tensor.detach().cpu() for name, tensor in deployed.state_dict().items()},
        "config": source["config"],
    }
    _atomic_torch_save(payload, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    restored = load_export(output_path, device)
    reload_check = {"exact": True, "max_abs": {}}
    if sample["available"]:
        restored_output = restored(packed)
        for name in DEPLOYMENT_OUTPUTS:
            delta = (after[name] - restored_output[name]).abs()
            reload_check["max_abs"][name] = float(delta.max().item())
            torch.testing.assert_close(after[name], restored_output[name], atol=0, rtol=0)
    one = packed[:1] if sample["available"] else torch.randn(1, 4, 240, 320, device=device)
    macs, parameters = profile(restored, inputs=(one,), verbose=False)
    timings, allocated, reserved = _run_timed(
        device, lambda: restored(one), warmup=warmup, repeats=benchmark_repeats, cuda_events=True
    )
    network = _timing_stats(
        timings,
        stage="network_forward",
        condition="sample",
        batch_size=one.shape[0],
        warmup=warmup,
        device=device,
        input_shape=list(one.shape),
        peak_allocated=allocated,
        peak_reserved=reserved,
    )
    report = {
        "format": DEPLOYMENT_FORMAT,
        "deployment_path": str(output_path),
        "metadata": metadata,
        "structure": structure,
        "sample_check": sample,
        "reload_check": reload_check,
        "parameter_count": int(parameters),
        "macs_batch1": int(macs),
        "network_forward": network,
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "source_step": metadata["source_step"],
    }
    (output_path.parent / "export_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
