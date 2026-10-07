"""Fixed 256-image, 1,536-pair COCO protocol and baseline/student evaluation."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch

from .baseline import baseline_features
from .data import PairGenerator, PairRequest, image_paths
from .features import extract_features
from .metrics import pair_metrics, mutual_nearest


RATIOS = (1, 4, 16, 64, 100)
PROTOCOL = "rawfeat.coco.fixed.v2"
BASELINE_PREPROCESSING = "rawslam.meanad.uint16.v1"
CACHE_RECIPE = "rawfeat.coco.cache.v2"
PERCENT_METRICS = ("h_auc_1", "h_auc_3", "h_auc_5", "repeatability", "match_precision")
COUNT_METRICS = ("correct_matches", "detected_a", "detected_b", "matches", "homography_failures")
PIXEL_METRICS = ("corner_error", "localization_error")


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_digest() -> str:
    digest = hashlib.sha256()
    for name in ("data.py", "sensor.py", "noise.py"):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    versions = {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                "albumentations": version("albumentations"), "kornia": version("kornia")}
    digest.update(json.dumps(versions, sort_keys=True).encode())
    return digest.hexdigest()


def baseline_code_digest() -> str:
    digest = hashlib.sha256()
    for name in ("baseline.py", "features.py", "metrics.py", "validation.py"):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def cache_identity(manifest_sha256: str, invisp_repo: str | Path, eld_calibration: str | Path) -> dict:
    repo = Path(invisp_repo)
    return {"protocol": PROTOCOL, "recipe": CACHE_RECIPE,
            "manifest_sha256": manifest_sha256, "pipeline_sha256": pipeline_digest(),
            "invisp_checkpoint_sha256": file_digest(repo / "pretrained" / "canon.pth"),
            "invisp_source_sha256": [file_digest(repo / "model" / name) for name in ("model.py", "modules.py")],
            "eld_calibration_sha256": file_digest(eld_calibration)}


def identity_digest(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_cache_identity(root: str | Path, manifest_sha256: str) -> dict:
    identity = json.loads((Path(root) / "identity.json").read_text())
    required = {"protocol": PROTOCOL, "recipe": CACHE_RECIPE,
                "manifest_sha256": manifest_sha256, "pipeline_sha256": pipeline_digest()}
    if any(identity.get(key) != value for key, value in required.items()):
        raise RuntimeError("validation cache identity differs from current manifest or data pipeline")
    return identity


def create_manifest(coco_root: str | Path, output: str | Path) -> dict:
    paths = image_paths(coco_root, "val2017")
    if len(paths) < 256:
        raise ValueError("fixed validation requires at least 256 COCO val2017 images")
    rng = np.random.default_rng(2027)
    selected = rng.choice(len(paths), size=256, replace=False)
    images = []
    for position, index in enumerate(selected):
        relative = f"val2017/{paths[int(index)].name}"
        seeds = rng.integers(0, 2**31 - 1, size=5).tolist()
        base = {"image": relative, "crop_seed": seeds[0], "appearance_a_seed": seeds[1],
                "appearance_b_seed": seeds[2], "geometry_seed": seeds[3], "base_seed": seeds[4]}
        cases = []
        for bucket in ("clean", *map(str, RATIOS)):
            if bucket == "clean":
                ratio_a = ratio_b = 1.0
                clean = True
            else:
                ratio = float(bucket)
                if position < 128:
                    ratio_a = ratio_b = ratio
                elif position < 192:
                    ratio_a, ratio_b = 1.0, ratio
                else:
                    ratio_a, ratio_b = ratio, 1.0
                clean = False
            noise_seed = rng.integers(0, 2**31 - 1, size=2).tolist()
            cases.append({"bucket": bucket, "ratio_a": ratio_a, "ratio_b": ratio_b,
                          "noise_a_seed": noise_seed[0], "noise_b_seed": noise_seed[1], "clean": clean})
        images.append({**base, "cases": cases})
    manifest = {"protocol": PROTOCOL, "seed": 2027, "image_count": 256,
                "case_count": 1536, "images": images}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_manifest(path: str | Path) -> tuple[dict, str]:
    manifest = json.loads(Path(path).read_text())
    if manifest["protocol"] != PROTOCOL or manifest["image_count"] != 256 or manifest["case_count"] != 1536:
        raise ValueError("manifest does not match the fixed COCO validation protocol")
    if len(manifest["images"]) != 256 or any(len(image["cases"]) != 6 for image in manifest["images"]):
        raise ValueError("manifest image/case counts are incomplete")
    sources = [image["image"] for image in manifest["images"]]
    if manifest.get("seed") != 2027 or len(set(sources)) != 256:
        raise ValueError("manifest requires seed 2027 and 256 distinct validation sources")
    for position, image in enumerate(manifest["images"]):
        source = Path(image["image"])
        if source.parent != Path("val2017") or source.suffix != ".jpg":
            raise ValueError("manifest sources must be COCO val2017 JPEGs")
        for name in ("crop_seed", "appearance_a_seed", "appearance_b_seed", "geometry_seed", "base_seed"):
            if not isinstance(image.get(name), int) or not 0 <= image[name] < 2**31:
                raise ValueError(f"invalid fixed seed: {name}")
        if [case["bucket"] for case in image["cases"]] != ["clean", *map(str, RATIOS)]:
            raise ValueError("manifest requires one clean case and all five ordered ratio buckets")
        for case in image["cases"]:
            clean = case["bucket"] == "clean"
            ratio = 1.0 if clean else float(case["bucket"])
            expected = ((ratio, ratio) if clean or position < 128 else
                        (1.0, ratio) if position < 192 else (ratio, 1.0))
            if case.get("clean") is not clean or (case.get("ratio_a"), case.get("ratio_b")) != expected:
                raise ValueError("manifest clean flag or balanced ratio assignment differs from fixed protocol")
            for name in ("noise_a_seed", "noise_b_seed"):
                if not isinstance(case.get(name), int) or not 0 <= case[name] < 2**31:
                    raise ValueError(f"invalid fixed seed: {name}")
    return manifest, manifest_digest(manifest)


def case_request(image: dict, case: dict, coco_root: str | Path) -> PairRequest:
    return PairRequest(
        image=str(Path(coco_root) / image["image"]), crop_seed=image["crop_seed"],
        appearance_a_seed=image["appearance_a_seed"], appearance_b_seed=image["appearance_b_seed"],
        geometry_seed=image["geometry_seed"], noise_a_seed=case["noise_a_seed"],
        noise_b_seed=case["noise_b_seed"], ratio_a=case["ratio_a"], ratio_b=case["ratio_b"],
        clean=case["clean"],
    )


def cache_path(root: str | Path, image_position: int, bucket: str) -> Path:
    return Path(root) / f"{image_position:03d}_{bucket}.pt"


@torch.no_grad()
def cache_validation(
    manifest_path: str | Path, coco_root: str | Path, cache_root: str | Path,
    generator: PairGenerator, *, max_images: int = 256,
) -> dict:
    if not 1 <= max_images <= 256:
        raise ValueError("max_images must be between 1 and 256")
    manifest, digest = read_manifest(manifest_path)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    identity = cache_root / "identity.json"
    identity_data = cache_identity(digest, generator.invisp.rawfeat_repo_path, generator.eld.calibration)
    if identity.exists() and json.loads(identity.read_text()) != identity_data:
        raise RuntimeError("cache identity differs from manifest; use a new cache directory")
    identity.write_text(json.dumps(identity_data, indent=2) + "\n")
    completed = 0
    for position, image in enumerate(manifest["images"][:max_images]):
        paths = [cache_path(cache_root, position, case["bucket"]) for case in image["cases"]]
        if all(path.is_file() for path in paths):
            completed += len(paths)
            continue
        first = case_request(image, image["cases"][0], coco_root)
        base = generator.generate_base(first, fixed_crop=True)
        for case, path in zip(image["cases"], paths):
            if path.is_file():
                completed += 1
                continue
            result = generator.apply_noise(base, case_request(image, case, coco_root))
            stored = {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                      for name, value in result.items() if name != "clean_bayer_dn"}
            temporary = path.with_suffix(".tmp")
            torch.save(stored, temporary)
            temporary.replace(path)
            completed += 1
    return {"manifest_sha256": digest, "cached_cases": completed, "expected_cases": 1536,
            "full_protocol": completed == 1536}


def _numpy_features(features: list[dict[str, torch.Tensor]]) -> list[dict[str, np.ndarray]]:
    return [{name: tensor.detach().cpu().numpy() for name, tensor in view.items()} for view in features]


def _serialize_metric(value: float | int) -> float | int | None:
    return value if np.isfinite(value) else None


def baseline_comparison(student_buckets: dict, baseline_buckets: dict) -> dict:
    """Per-bucket baseline values and signed student-minus-baseline differences."""
    differences = {}
    for bucket, student in student_buckets.items():
        baseline = baseline_buckets[bucket]
        difference = {}
        for name in PERCENT_METRICS:
            difference[f"{name}_percentage_points"] = 100 * (student[name] - baseline[name])
        for name in COUNT_METRICS + PIXEL_METRICS:
            left, right = student[name], baseline[name]
            difference[name] = left - right if left is not None and right is not None else None
        differences[bucket] = difference
    return differences


def summarize_buckets(metrics_by_bucket: dict[str, list[dict]]) -> dict:
    buckets = {}
    for bucket, samples in metrics_by_bucket.items():
        values = {}
        for name in (*PERCENT_METRICS, *PIXEL_METRICS, *COUNT_METRICS[:-1]):
            finite = [float(sample[name]) for sample in samples
                      if sample[name] is not None and np.isfinite(sample[name])]
            values[name] = float(np.mean(finite)) if finite else None
        values["cases"] = len(samples)
        values["homography_failures"] = sum(sample["corner_error"] is None
                                            or not np.isfinite(sample["corner_error"]) for sample in samples)
        buckets[bucket] = values
    return buckets


def baseline_subset(summary_path: str | Path, max_images: int) -> dict:
    rows_path = Path(summary_path).parent / "per_pair.jsonl"
    samples = {bucket: [] for bucket in ("clean", *map(str, RATIOS))}
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    selected = [row for row in rows if 0 <= row["image_position"] < max_images]
    identities = {(row["image_position"], row["bucket"]) for row in selected}
    expected = {(position, bucket) for position in range(max_images) for bucket in samples}
    if identities != expected or len(selected) != len(expected):
        raise RuntimeError("baseline per-pair results do not cover the requested validation subset")
    for row in selected:
        samples[row["bucket"]].append(row)
    return summarize_buckets(samples)


def _visualize(pair: dict, features: list[dict[str, np.ndarray]], path: Path) -> None:
    gray = pair["srgb_gray"][:, 0].numpy()
    canvas = np.concatenate([np.clip(frame * 255, 0, 255).astype(np.uint8) for frame in gray], axis=1)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    pts_a, pts_b = features[0]["points"], features[1]["points"]
    ia, ib = mutual_nearest(features[0]["descriptors"], features[1]["descriptors"])
    for left, right in zip(pts_a[ia[:100]], pts_b[ib[:100]]):
        cv2.line(canvas, tuple(np.rint(left).astype(int)), tuple(np.rint(right + [640, 0]).astype(int)), (0, 160, 0), 1)
    for view, offset in zip(features, (0, 640)):
        for point in view["points"][:1024]:
            cv2.circle(canvas, (round(float(point[0])) + offset, round(float(point[1]))), 1, (0, 0, 255), -1)
    cv2.imwrite(str(path), canvas)


@torch.no_grad()
def evaluate(
    manifest_path: str | Path, cache_root: str | Path, output: str | Path,
    *, model: torch.nn.Module | None = None, teacher: torch.nn.Module | None = None,
    baseline_summary: str | Path | None = None, max_images: int = 256,
) -> dict:
    if (model is None) == (teacher is None):
        raise ValueError("pass exactly one of student model or baseline teacher")
    if not 1 <= max_images <= 256:
        raise ValueError("max_images must be between 1 and 256")
    if model is not None and model.training:
        raise ValueError("fixed validation requires student eval mode")
    manifest, digest = read_manifest(manifest_path)
    cache_root = Path(cache_root)
    identity = read_cache_identity(cache_root, digest)
    baseline_data = None
    if baseline_summary is not None:
        baseline_data = json.loads(Path(baseline_summary).read_text())
        if (baseline_data["manifest_sha256"] != digest or baseline_data["preprocessing"] != BASELINE_PREPROCESSING
                or baseline_data.get("cache_identity_sha256") != identity_digest(identity)
                or baseline_data.get("baseline_code_sha256") != baseline_code_digest()):
            raise RuntimeError("baseline was run with a different manifest or preprocessing")
        if not baseline_data["full_protocol"]:
            raise RuntimeError("partial baseline cannot be used for formal comparison")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions"
    predictions.mkdir(exist_ok=True)
    visualizations = output / "visualizations"
    visualizations.mkdir(exist_ok=True)
    device = next((model or teacher).parameters()).device
    metrics_by_bucket: dict[str, list[dict]] = {name: [] for name in ("clean", *map(str, RATIOS))}
    rows = []
    for position, image in enumerate(manifest["images"][:max_images]):
        for case in image["cases"]:
            bucket = case["bucket"]
            path = cache_path(cache_root, position, bucket)
            if not path.is_file():
                raise FileNotFoundError(f"missing fixed validation case: {path}")
            pair = torch.load(path, map_location="cpu", weights_only=True)
            valid = pair["valid_pixels"][:, 0].to(device)
            if model is not None:
                result = model(pair["packed"].to(device))
                features = extract_features(result["logits"], result["descriptors"], valid=valid)
            else:
                features = baseline_features(teacher, pair["noisy_bayer_dn"], valid=valid)
            numpy_features = _numpy_features(features)
            valid_np = pair["valid_pixels"][:, 0].numpy()
            metrics = pair_metrics(numpy_features[0], numpy_features[1], pair["homography"].numpy(),
                                   valid_np[0], valid_np[1])
            metrics_by_bucket[bucket].append(metrics)
            row = {"image_position": position, "source": image["image"], "bucket": bucket,
                   "ratio_a": case["ratio_a"], "ratio_b": case["ratio_b"],
                   **{key: _serialize_metric(value) for key, value in metrics.items()}}
            rows.append(row)
            match_a, match_b = mutual_nearest(numpy_features[0]["descriptors"], numpy_features[1]["descriptors"])
            np.savez_compressed(predictions / f"{position:03d}_{bucket}.npz",
                                points_a=numpy_features[0]["points"], scores_a=numpy_features[0]["scores"],
                                descriptors_a=numpy_features[0]["descriptors"],
                                points_b=numpy_features[1]["points"], scores_b=numpy_features[1]["scores"],
                                descriptors_b=numpy_features[1]["descriptors"],
                                match_indices_a=match_a, match_indices_b=match_b)
            if model is not None and bucket == str(RATIOS[-1]) and position < 16:
                _visualize(pair, numpy_features, visualizations / f"{position:03d}.png")
    (output / "per_pair.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    buckets = summarize_buckets(metrics_by_bucket)
    score = float(np.mean([buckets[str(ratio)]["h_auc_5"] for ratio in RATIOS]))
    summary = {"protocol": PROTOCOL, "manifest_sha256": digest,
               "cache_identity_sha256": identity_digest(identity),
               "preprocessing": BASELINE_PREPROCESSING if teacher is not None else "rawfeat.student.v1",
               "full_protocol": max_images == 256 and len(rows) == 1536,
               "score": score, "buckets": buckets, "cases": len(rows)}
    if teacher is not None:
        summary["superpoint_sha256"] = file_digest(teacher.rawfeat_checkpoint_path)
        summary["baseline_code_sha256"] = baseline_code_digest()
    if baseline_data is not None:
        baseline_buckets = (baseline_data["buckets"] if max_images == 256 else
                            baseline_subset(baseline_summary, max_images))
        baseline_score = float(np.mean([baseline_buckets[str(ratio)]["h_auc_5"] for ratio in RATIOS]))
        summary["baseline_images"] = max_images
        summary["baseline_score"] = baseline_score
        summary["score_delta_percentage_points"] = 100 * (score - baseline_score)
        summary["baseline_buckets"] = baseline_buckets
        summary["bucket_deltas"] = baseline_comparison(buckets, baseline_buckets)
        summary["bucket_deltas_percentage_points"] = {
            key: 100 * (buckets[key]["h_auc_5"] - baseline_buckets[key]["h_auc_5"])
            for key in map(str, RATIOS)
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
