"""RepVGG fusion, fixed-protocol check, and FP32 efficiency measurement."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np
from thop import profile

from .identity import check_schema
from .features import extract_features, score_map
from .model import RawFeatureExtractor
from .runtime import configure_fp32
from .sensor import pack_student
from .validation import cache_path, evaluate, read_manifest


def load_export(path: str | Path, device: torch.device | str = "cuda") -> RawFeatureExtractor:
    stored = torch.load(path, map_location="cpu", weights_only=False)
    model = RawFeatureExtractor(auxiliary=False, deploy=True)
    model.load_state_dict(stored["student"], strict=True)
    return model.eval().to(device)


def _timed(device: torch.device, function, warmup: int = 10, repeats: int = 50) -> dict:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(repeats):
        start.record()
        function()
        end.record()
        torch.cuda.synchronize(device)
        timings.append(start.elapsed_time(end))
    return {'median_ms': float(np.median(timings)), 'p10_ms': float(np.percentile(timings, 10)),
            'p90_ms': float(np.percentile(timings, 90)), 'repeats': repeats, 'warmup': warmup}


def verify_discrete_case(a: dict, b: dict, old_scores: np.ndarray, new_scores: np.ndarray) -> dict:
    """Verify rank/NMS/MNN changes at measured FP32 boundaries, retaining actual geometry."""
    error = float(np.abs(old_scores-new_scores).max())
    tolerance = 2*error + np.finfo(np.float32).eps
    order_changed, set_changed, descriptor_error = False, False, 0.
    aligned = {}
    for view, index in [('a', 0), ('b', 1)]:
        pa, pb = a['points_'+view], b['points_'+view]
        ia = {tuple(p): i for i, p in enumerate(pa)}; ib = {tuple(p): i for i, p in enumerate(pb)}
        order_changed |= not np.array_equal(pa, pb)
        set_changed |= ia.keys() != ib.keys()
        common = [p for p in ib if p in ia]
        if common:
            da = a['descriptors_'+view][[ia[p] for p in common]]
            db = b['descriptors_'+view][[ib[p] for p in common]]
            np.testing.assert_allclose(da, db, atol=1e-4, rtol=1e-4)
            descriptor_error = max(descriptor_error, float(np.abs(da-db).max()))
            values = np.array([old_scores[index, int(p[1]), int(p[0])] for p in common])
            inversion = float(np.max(values[1:]-np.minimum.accumulate(values[:-1]))) if len(values)>1 else 0.
            if inversion > tolerance:
                raise RuntimeError('fusion changed a non-tied score ranking')
        for original, changed, own, other in [(ia, ib, old_scores[index], new_scores[index]),
                                              (ib, ia, new_scores[index], old_scores[index])]:
            for point in original.keys()-changed.keys():
                x, y = map(int, point)
                value = own[y, x]
                rivals = [p for p in changed if max(abs(p[0]-x), abs(p[1]-y)) <= 4]
                near_nms = any(abs(float(own[int(p[1]), int(p[0])])-float(value)) <= tolerance for p in rivals)
                near_threshold = abs(float(value)-.005) <= tolerance
                near_cutoff = (len(changed)==1024 and abs(float(value)-min(float(other[int(p[1]), int(p[0])]) for p in changed)) <= tolerance)
                if not (near_nms or near_threshold or near_cutoff):
                    raise RuntimeError('fusion changed a keypoint away from NMS/threshold/TopK boundaries')
        if ia.keys() == ib.keys():
            aligned[view] = (a['descriptors_'+view], b['descriptors_'+view][[ib[tuple(p)] for p in pa]])
    if len(aligned) == 2 and len(aligned['a'][0]) and len(aligned['b'][0]):
        before = aligned['a'][0] @ aligned['b'][0].T
        after = aligned['a'][1] @ aligned['b'][1].T
        affinity_error = float(np.abs(before-after).max())
        for old, new in [(before, after), (before.T, after.T)]:
            changed = np.where(old.argmax(1) != new.argmax(1))[0]
            if len(changed):
                gaps = old[changed, old[changed].argmax(1)] - old[changed, new[changed].argmax(1)]
                if float(gaps.max()) > 2*affinity_error + 8*np.finfo(np.float32).eps:
                    raise RuntimeError('fusion changed a nearest neighbor away from a numeric tie')
    def matched_coordinates(features):
        return {(*features['points_a'][i], *features['points_b'][j]) for i, j in zip(features['match_indices_a'], features['match_indices_b'])}
    return {'score_error': error, 'point_order_changed': bool(order_changed), 'point_set_changed': bool(set_changed),
            'match_coordinate_set_changed': matched_coordinates(a) != matched_coordinates(b),
            'common_descriptor_max_error': descriptor_error, 'numeric_boundaries_verified': True}


@torch.no_grad()
def export_checkpoint(
    checkpoint: str | Path, output: str | Path, *, cache_root: str | Path,
    manifest: str | Path, device: torch.device | str = "cuda:0", max_images: int = 256,
    benchmark_repeats: int = 50,
) -> dict:
    if benchmark_repeats <= 0:
        raise ValueError('benchmark repeats must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deployment validation and latency measurement")
    device = torch.device(device)
    torch.cuda.set_device(device)
    configure_fp32()
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    check_schema(source)
    if source['step'] <= source['config']['schedule']['detection_updates']:
        raise ValueError('export requires a B/C checkpoint with an active trained descriptor')
    training = RawFeatureExtractor()
    training.load_state_dict(source["student"], strict=True)
    training = training.eval().to(device)
    deployed = training.deploy_copy()
    pair = torch.load(cache_path(cache_root, 0, "clean"), map_location="cpu", weights_only=True)
    packed = pair["packed"].to(device)
    before = training(packed)
    after = deployed(packed)
    differences = {}
    for name in ("logits", "descriptors"):
        differences[name] = float((before[name] - after[name]).abs().max())
        torch.testing.assert_close(before[name], after[name], atol=1e-4, rtol=1e-4)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student": deployed.state_dict(), "source_checkpoint": str(checkpoint),
                "source_step": source["step"], "config": source["config"]}, output)
    restored = load_export(output, device)
    restored_outputs = restored(packed)
    for name in ("logits", "descriptors"):
        torch.testing.assert_close(after[name], restored_outputs[name], atol=0, rtol=0)
    eval_root = output.parent / "export_validation"
    original_metrics = evaluate(manifest, cache_root, eval_root / "original", model=training, max_images=max_images)
    deployment_metrics = evaluate(manifest, cache_root, eval_root / "deployed", model=deployed, max_images=max_images)
    protocol, _ = read_manifest(manifest)
    discrete_rows = []
    for position, image in enumerate(protocol['images'][:max_images]):
        for case in image['cases']:
            stored_pair = torch.load(cache_path(cache_root, position, case['bucket']), map_location='cpu', weights_only=True)
            inputs = stored_pair['packed'].to(device)
            original, fused = training(inputs), deployed(inputs)
            for name in ('logits', 'descriptors'):
                torch.testing.assert_close(original[name], fused[name], atol=1e-4, rtol=1e-4)
                differences[name] = max(differences[name], float((original[name]-fused[name]).abs().max()))
            filename = f"{position:03d}_{case['bucket']}.npz"
            with np.load(eval_root/'original/predictions'/filename) as a, np.load(eval_root/'deployed/predictions'/filename) as b:
                row = verify_discrete_case(a, b, score_map(original['logits']).cpu().numpy(), score_map(fused['logits']).cpu().numpy())
            discrete_rows.append({'image_position': position, 'bucket': case['bucket'], **row})
    (output.parent/'fusion_per_pair.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in discrete_rows))
    discrete = {'cases': len(discrete_rows),
                **{key: sum(row[key] for row in discrete_rows) for key in ('point_order_changed', 'point_set_changed', 'match_coordinate_set_changed')},
                'max_common_descriptor_error': max(row['common_descriptor_max_error'] for row in discrete_rows),
                'max_decoded_score_error': max(row['score_error'] for row in discrete_rows),
                'numeric_boundaries_verified': True}
    # RANSAC is index/order sensitive even for identical match coordinate sets.
    # Keep both protocol results; never round scores or alter the extraction math.
    one = packed[:1]
    bayer = pair["noisy_bayer_dn"][:1].to(device)
    macs, parameters = profile(deployed, inputs=(one,), verbose=False)
    forward_ms = _timed(device, lambda: deployed(one), repeats=benchmark_repeats)

    def full_extraction():
        processed = pack_student(bayer)
        output_tensors = deployed(processed)
        extract_features(output_tensors["logits"], output_tensors["descriptors"])

    full_ms = _timed(device, full_extraction, repeats=benchmark_repeats)
    report = {"source_checkpoint": str(checkpoint), "source_step": source["step"],
              "max_absolute_difference": differences, "parameter_count": int(parameters),
              "macs_batch1": int(macs), "network_latency": forward_ms,
              "full_extraction_latency": full_ms, "discrete_fusion_check": discrete,
              "fusion_tolerance": {"logits_descriptors_atol_rtol": 1e-4, "discrete_score_gap": "2*measured_score_error + FP32 epsilon", "nearest_neighbor_gap": "2*measured_affinity_error + 8*FP32 epsilon"},
              "discrete_note": "All case dense outputs/common-point descriptors are checked. Discrete changes must be at measured numeric ties. RANSAC samples indices, so point ordering can change geometry even for identical match coordinate sets; both raw results and score delta are retained.", "batch": 1, "precision": "FP32",
              "input_shape": list(one.shape), "device": torch.cuda.get_device_name(device),
              "validation_score": deployment_metrics["score"], "original_validation_score": original_metrics["score"],
              "score_delta_percentage_points": 100*(deployment_metrics["score"]-original_metrics["score"]),
              "full_validation": deployment_metrics["full_protocol"]}
    (output.parent / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
