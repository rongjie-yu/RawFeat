"""Independent full-image HPatches synthetic Raw evaluation protocol."""

from __future__ import annotations

import hashlib
import html
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

from .baseline import baseline_gray
from .features import extract_features, score_map
from .metrics import mutual_nearest, pair_metrics, warp_points
from .noise import CanonELD, no_noise
from .sensor import invisp_to_sensor_rgb, pack_student, sample_bayer


HPATCHES_PROTOCOL = "rawfeat.hpatches.synthetic_raw.v1"
HPATCHES_CACHE_RECIPE = "rawfeat.hpatches.image_cache.v1"
HPATCHES_HEIGHT = 480
HPATCHES_WIDTH = 640
HPATCHES_SEED = 2027
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_ITERATIONS = 10000
CONDITIONS = ("clean", "ratio1", "ratio4", "ratio16", "ratio64", "ratio100")
NOISY_CONDITIONS = ("ratio1", "ratio4", "ratio16", "ratio64", "ratio100")
NOISY_RATIOS = (1, 4, 16, 64, 100)
AVERAGE_METRICS = (
    "h_auc_1", "h_auc_3", "h_auc_5", "threshold_success_1", "threshold_success_3",
    "threshold_success_5", "repeatability", "localization_error", "match_precision",
    "corner_error", "detected_a", "detected_b", "matches", "correct_matches",
)
COUNT_METRICS = ("correct_matches", "detected_a", "detected_b", "matches")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(manifest: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(manifest).encode())


def stable_seed(global_seed: int, sequence_position: int, image_index: int, ratio: int) -> int:
    """Derive a reproducible 31-bit ELD seed without process-dependent hashing."""
    if ratio not in NOISY_RATIOS:
        raise ValueError(f"unsupported noisy ratio: {ratio}")
    state = np.random.SeedSequence(
        [global_seed, sequence_position, image_index, ratio, 0x4850]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def condition_specs() -> list[dict[str, Any]]:
    specs = [{"name": "clean", "clean": True, "ratio": None,
              "ratio_a": None, "ratio_b": None}]
    for ratio in NOISY_RATIOS:
        specs.append({"name": f"ratio{ratio}", "clean": False, "ratio": ratio,
                      "ratio_a": ratio, "ratio_b": ratio})
    return specs


def _read_homography(path: str | Path) -> np.ndarray:
    matrix = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64)
    if matrix.size != 9:
        raise ValueError(f"homography must contain nine values: {path}")
    matrix = matrix.reshape(3, 3)
    if (not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12
            or abs(float(matrix[2, 2])) < 1e-12):
        raise ValueError(f"homography is not finite and invertible: {path}")
    matrix /= matrix[2, 2]
    return matrix


def resize_transform(width: int, height: int) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    scale = max(HPATCHES_WIDTH / width, HPATCHES_HEIGHT / height)
    resized_width = max(HPATCHES_WIDTH, int(round(width * scale)))
    resized_height = max(HPATCHES_HEIGHT, int(round(height * scale)))
    crop_left = (resized_width - HPATCHES_WIDTH) // 2
    crop_top = (resized_height - HPATCHES_HEIGHT) // 2
    scale_x = resized_width / width
    scale_y = resized_height / height
    matrix = np.array([
        [scale_x, 0.0, (scale_x - 1.0) / 2.0 - crop_left],
        [0.0, scale_y, (scale_y - 1.0) / 2.0 - crop_top],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return {
        "source_width": int(width), "source_height": int(height),
        "resized_width": resized_width, "resized_height": resized_height,
        "scale": float(scale), "scale_x": float(scale_x), "scale_y": float(scale_y),
        "crop_left": int(crop_left), "crop_top": int(crop_top),
        "output_width": HPATCHES_WIDTH, "output_height": HPATCHES_HEIGHT,
        "T": matrix.tolist(),
    }


def transform_homography(transform_a: dict[str, Any], transform_b: dict[str, Any],
                         homography: np.ndarray) -> np.ndarray:
    """Map source-image H into the actual resize/crop pixel coordinates."""
    transformed = np.asarray(transform_b["T"], dtype=np.float64) @ homography @ np.linalg.inv(
        np.asarray(transform_a["T"], dtype=np.float64)
    )
    transformed /= transformed[2, 2]
    return transformed


def _integrity_entries(integrity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["path"]: entry for entry in integrity.get("files", [])}


def preflight_hpatches(
    dataset_root: str | Path,
    integrity_path: str | Path | None = None,
    *,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Validate all full-image files and homographies without loading a model."""
    root = Path(dataset_root).resolve()
    integrity_file = Path(integrity_path or root / "dataset_integrity.json").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HPatches dataset root does not exist: {root}")
    if not integrity_file.is_file():
        raise FileNotFoundError(f"dataset integrity file does not exist: {integrity_file}")
    integrity = json.loads(integrity_file.read_text())
    if integrity.get("variant") != "HPatches full image sequences":
        raise ValueError("dataset_integrity.json is not the full-image HPatches record")
    expected_root = Path(integrity.get("dataset_root", root)).resolve()
    if expected_root != root:
        raise ValueError(f"integrity root differs from requested root: {expected_root} != {root}")
    directories = sorted(path for path in root.iterdir() if path.is_dir() and path.name[:2] in ("i_", "v_"))
    if len(directories) != 116:
        raise ValueError(f"expected 116 HPatches sequences, found {len(directories)}")
    illumination = sum(path.name.startswith("i_") for path in directories)
    viewpoint = sum(path.name.startswith("v_") for path in directories)
    if (illumination, viewpoint) != (57, 59):
        raise ValueError(f"expected 57 illumination and 59 viewpoint sequences, found {illumination}/{viewpoint}")
    entries = _integrity_entries(integrity)
    sequences: list[dict[str, Any]] = []
    image_count = 0
    pair_count = 0
    checked_hashes = 0
    for sequence_position, directory in enumerate(directories):
        sequence_type = "illumination" if directory.name.startswith("i_") else "viewpoint"
        images: list[dict[str, Any]] = []
        for image_index in range(1, 7):
            image_path = directory / f"{image_index}.ppm"
            relative = f"{directory.name}/{image_index}.ppm"
            decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"cannot decode HPatches image: {image_path}")
            height, width = decoded.shape[:2]
            if decoded.ndim != 3 or decoded.shape[2] != 3:
                raise ValueError(f"HPatches image is not RGB: {image_path}")
            expected = entries.get(relative)
            if expected is not None and (expected["width"], expected["height"]) != (width, height):
                raise ValueError(f"image dimensions differ from integrity record: {relative}")
            images.append({"index": image_index, "path": relative, "width": width, "height": height})
            image_count += 1
        homographies: list[dict[str, Any]] = []
        for target_index in range(2, 7):
            path = directory / f"H_1_{target_index}"
            relative = f"{directory.name}/H_1_{target_index}"
            matrix = _read_homography(path)
            homographies.append({"target_index": target_index, "path": relative,
                                 "H": matrix.tolist()})
            pair_count += 1
        sequences.append({"name": directory.name, "type": sequence_type,
                          "position": sequence_position, "images": images,
                          "homographies": homographies})
    if (image_count, pair_count) != (696, 580):
        raise ValueError(f"expected 696 images and 580 pairs, found {image_count}/{pair_count}")
    if integrity.get("sequences") != 116 or integrity.get("images") != 696 or integrity.get("reference_target_pairs") != 580:
        raise ValueError("dataset_integrity.json counters do not match the full-image protocol")
    if verify_hashes:
        if len(entries) != 1276:
            raise ValueError(f"integrity record must contain 1276 files, found {len(entries)}")
        for relative, entry in entries.items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"integrity file is missing: {path}")
            if path.stat().st_size != entry["size_bytes"] or file_digest(path) != entry["sha256"]:
                raise ValueError(f"integrity hash or size mismatch: {relative}")
            checked_hashes += 1
    return {
        "dataset_root": str(root), "integrity_path": str(integrity_file),
        "integrity_sha256": file_digest(integrity_file),
        "archive_sha256": integrity.get("archive_sha256"),
        "sequences": sequences, "sequence_count": len(sequences),
        "illumination_count": illumination, "viewpoint_count": viewpoint,
        "image_count": image_count, "pair_count": pair_count,
        "checked_hashes": checked_hashes, "verify_hashes": verify_hashes,
    }


def create_manifest(
    dataset_root: str | Path,
    integrity_path: str | Path | None,
    output: str | Path,
    *,
    seed: int = HPATCHES_SEED,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    preflight = preflight_hpatches(dataset_root, integrity_path, verify_hashes=verify_hashes)
    root = Path(preflight["dataset_root"])
    sequences: list[dict[str, Any]] = []
    for sequence in preflight["sequences"]:
        image_records: list[dict[str, Any]] = []
        transforms: dict[int, dict[str, Any]] = {}
        for image in sequence["images"]:
            transform = resize_transform(image["width"], image["height"])
            transforms[image["index"]] = transform
            noise_seeds: dict[str, int | None] = {"clean": None}
            for ratio in NOISY_RATIOS:
                noise_seeds[f"ratio{ratio}"] = stable_seed(seed, sequence["position"], image["index"], ratio)
            image_records.append({
                "id": f"{sequence['name']}/{image['index']}", "index": image["index"],
                "path": image["path"], "width": image["width"], "height": image["height"],
                "transform": transform, "noise_seeds": noise_seeds,
            })
        pairs: list[dict[str, Any]] = []
        reference = transforms[1]
        for homography in sequence["homographies"]:
            target = homography["target_index"]
            raw = np.asarray(homography["H"], dtype=np.float64)
            processed = transform_homography(reference, transforms[target], raw)
            pairs.append({
                "id": f"{sequence['name']}/1->{target}", "reference_index": 1,
                "target_index": target, "reference_image_id": f"{sequence['name']}/1",
                "target_image_id": f"{sequence['name']}/{target}",
                "H_raw": raw.tolist(), "H_processed": processed.tolist(),
            })
        sequences.append({"name": sequence["name"], "type": sequence["type"],
                          "position": sequence["position"], "images": image_records, "pairs": pairs})
    manifest = {
        "protocol": HPATCHES_PROTOCOL, "seed": int(seed),
        "dataset_root": str(root), "integrity_path": preflight["integrity_path"],
        "dataset_integrity_sha256": preflight["integrity_sha256"],
        "dataset_archive_sha256": preflight["archive_sha256"],
        "target_size": {"height": HPATCHES_HEIGHT, "width": HPATCHES_WIDTH},
        "conditions": condition_specs(), "sequences": sequences,
        "sequence_count": 116, "illumination_count": 57, "viewpoint_count": 59,
        "image_count": 696, "pair_count": 580,
        "condition_pair_count": 580 * len(CONDITIONS),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def read_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("protocol") != HPATCHES_PROTOCOL:
        raise ValueError("manifest is not the current full-image HPatches protocol")
    if manifest.get("seed") != HPATCHES_SEED:
        raise ValueError("HPatches manifest seed must be 2027")
    if manifest.get("target_size") != {"height": HPATCHES_HEIGHT, "width": HPATCHES_WIDTH}:
        raise ValueError("HPatches manifest target size must be 480x640")
    if [item.get("name") for item in manifest.get("conditions", [])] != list(CONDITIONS):
        raise ValueError("manifest conditions must be clean, ratio1, ratio4, ratio16, ratio64, ratio100")
    for spec in manifest["conditions"]:
        if spec["name"] == "clean":
            expected = (True, None, None, None)
        else:
            ratio = int(spec["name"].removeprefix("ratio"))
            expected = (False, ratio, ratio, ratio)
        actual = (spec.get("clean"), spec.get("ratio"), spec.get("ratio_a"), spec.get("ratio_b"))
        if actual != expected:
            raise ValueError("HPatches conditions must contain symmetric ratios only")
    sequences = manifest.get("sequences", [])
    if (manifest.get("sequence_count"), manifest.get("illumination_count"),
            manifest.get("viewpoint_count"), manifest.get("image_count"),
            manifest.get("pair_count"), manifest.get("condition_pair_count")) != (116, 57, 59, 696, 580, 3480):
        raise ValueError("HPatches manifest counters are incomplete")
    names: set[str] = set()
    image_count = pair_count = 0
    for sequence in sequences:
        name = sequence.get("name", "")
        if name in names or (not name.startswith("i_") and not name.startswith("v_")):
            raise ValueError("HPatches sequence names must be unique i_/v_ directories")
        names.add(name)
        expected_type = "illumination" if name.startswith("i_") else "viewpoint"
        if sequence.get("type") != expected_type:
            raise ValueError(f"invalid HPatches sequence type: {name}")
        images = sequence.get("images", [])
        pairs = sequence.get("pairs", [])
        if [image.get("index") for image in images] != list(range(1, 7)) or len(pairs) != 5:
            raise ValueError(f"sequence does not contain six images and five pairs: {name}")
        for image in images:
            if image.get("id") != f"{name}/{image['index']}":
                raise ValueError("HPatches image id is inconsistent")
            if set(image.get("noise_seeds", {})) != set(CONDITIONS):
                raise ValueError("HPatches image noise seeds do not cover the six conditions")
            if image["noise_seeds"]["clean"] is not None or any(
                    not isinstance(image["noise_seeds"][condition], int)
                    or not 0 <= image["noise_seeds"][condition] < 2**32
                    for condition in NOISY_CONDITIONS):
                raise ValueError("HPatches noise seeds must be stable integer seeds")
            transform = image.get("transform", {})
            if transform.get("output_width") != HPATCHES_WIDTH or transform.get("output_height") != HPATCHES_HEIGHT:
                raise ValueError("HPatches image transform has the wrong output size")
            image_count += 1
        if [pair.get("target_index") for pair in pairs] != list(range(2, 7)):
            raise ValueError(f"HPatches pairs must be ordered 1->2...6: {name}")
        for pair in pairs:
            target = pair.get("target_index")
            if pair.get("reference_index") != 1 or target not in range(2, 7):
                raise ValueError("HPatches pair indices must be 1->2...6")
            raw = np.asarray(pair.get("H_raw"), dtype=np.float64)
            processed = np.asarray(pair.get("H_processed"), dtype=np.float64)
            if raw.shape != (3, 3) or processed.shape != (3, 3):
                raise ValueError("HPatches pair homographies must be 3x3")
            expected_processed = transform_homography(images[0]["transform"], images[target - 1]["transform"], raw)
            if not np.allclose(processed, expected_processed, rtol=0, atol=1e-10):
                raise ValueError(f"processed H does not match resize/crop transforms: {name} 1->{target}")
            pair_count += 1
    if len(sequences) != 116 or image_count != 696 or pair_count != 580:
        raise ValueError("HPatches manifest sequence/image/pair counts are incomplete")
    return manifest, manifest_digest(manifest)


def iter_manifest_images(manifest: dict[str, Any], sequence_names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    selected = None if sequence_names is None else set(sequence_names)
    result: list[dict[str, Any]] = []
    for sequence in manifest["sequences"]:
        if selected is not None and sequence["name"] not in selected:
            continue
        for image in sequence["images"]:
            result.append({**image, "sequence": sequence["name"], "type": sequence["type"],
                           "sequence_position": sequence["position"]})
    if selected is not None:
        missing = selected - {sequence["name"] for sequence in manifest["sequences"]}
        if missing:
            raise ValueError(f"unknown HPatches sequences: {sorted(missing)}")
    return result


def iter_manifest_pairs(
    manifest: dict[str, Any],
    sequence_names: Iterable[str] | None = None,
    max_pairs: int | None = None,
    target_indices: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    selected = None if sequence_names is None else set(sequence_names)
    selected_targets = None if target_indices is None else {int(target) for target in target_indices}
    if selected_targets is not None and not selected_targets.issubset(set(range(2, 7))):
        raise ValueError("HPatches target indices must be in 2..6")
    result: list[dict[str, Any]] = []
    for sequence in manifest["sequences"]:
        if selected is not None and sequence["name"] not in selected:
            continue
        for pair in sequence["pairs"]:
            if selected_targets is not None and pair["target_index"] not in selected_targets:
                continue
            result.append({**pair, "sequence": sequence["name"], "type": sequence["type"]})
    if selected is not None:
        missing = selected - {sequence["name"] for sequence in manifest["sequences"]}
        if missing:
            raise ValueError(f"unknown HPatches sequences: {sorted(missing)}")
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        result = result[:max_pairs]
    return result


def pipeline_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("hpatches.py", "sensor.py", "noise.py", "baseline.py", "features.py", "metrics.py", "teacher.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    versions = {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__}
    digest.update(_canonical_json(versions).encode())
    return digest.hexdigest()


def cache_identity(manifest_sha256: str, dataset_integrity_sha256: str,
                   invisp_repo: str | Path, eld_calibration: str | Path) -> dict[str, Any]:
    repo = Path(invisp_repo)
    return {
        "protocol": HPATCHES_PROTOCOL, "recipe": HPATCHES_CACHE_RECIPE,
        "manifest_sha256": manifest_sha256, "dataset_integrity_sha256": dataset_integrity_sha256,
        "pipeline_sha256": pipeline_digest(),
        "invisp_checkpoint_sha256": file_digest(repo / "pretrained" / "canon.pth"),
        "invisp_source_sha256": [file_digest(repo / "model" / name) for name in ("model.py", "modules.py")],
        "eld_calibration_sha256": file_digest(eld_calibration),
        "conditions": list(CONDITIONS), "dtype": "float32", "target_size": [HPATCHES_HEIGHT, HPATCHES_WIDTH],
    }


def identity_digest(identity: dict[str, Any]) -> str:
    return manifest_digest(identity)


def cache_path(root: str | Path, image_id: str, condition: str) -> Path:
    if condition not in CONDITIONS or "/" not in image_id:
        raise ValueError("invalid HPatches cache key")
    sequence, index = image_id.rsplit("/", 1)
    return Path(root) / "images" / sequence / f"{index}_{condition}.pt"


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _prepare_image(path: str | Path, transform: dict[str, Any]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot decode HPatches image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if transform["scale"] < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (transform["resized_width"], transform["resized_height"]), interpolation=interpolation)
    top = transform["crop_top"]
    left = transform["crop_left"]
    cropped = resized[top:top + HPATCHES_HEIGHT, left:left + HPATCHES_WIDTH]
    if cropped.shape[:2] != (HPATCHES_HEIGHT, HPATCHES_WIDTH):
        raise RuntimeError(f"resize/crop produced the wrong shape for {path}: {cropped.shape}")
    return np.ascontiguousarray(cropped).astype(np.float32) / 255.0


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_timed_forward(
    device: torch.device,
    function: Any,
) -> tuple[Any, float]:
    """Run one network forward and measure only the CUDA work it launches."""
    if device.type != "cuda":
        raise RuntimeError("HPatches inference timing requires CUDA")
    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    output = function()
    end.record(stream)
    end.synchronize()
    return output, float(start.elapsed_time(end))


def _single_image_inference(
    method: str,
    model: torch.nn.Module,
    noisy_bayer: torch.Tensor,
    image_id: str,
    condition: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Infer one cached Bayer image and return features plus a true image timing row."""
    if method not in {"rawfeat", "superpoint"}:
        raise ValueError(f"unsupported HPatches method: {method}")
    if tuple(noisy_bayer.shape) != (HPATCHES_HEIGHT, HPATCHES_WIDTH):
        raise ValueError("HPatches inference expects one 480x640 Bayer image")
    if device.type != "cuda":
        raise RuntimeError("HPatches inference requires CUDA")

    torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()
    if method == "rawfeat":
        input_started = time.perf_counter()
        packed = pack_student(noisy_bayer[None].to(device))
        input_processing_ms = 1000.0 * (time.perf_counter() - input_started)
        output, network_forward_ms = _cuda_timed_forward(device, lambda: model(packed))
        logits = output["logits"]
        descriptors = output["descriptors"]
    else:
        input_started = time.perf_counter()
        gray = baseline_gray(noisy_bayer.numpy())
        gray_tensor = torch.from_numpy(gray[None, None]).to(device)
        input_processing_ms = 1000.0 * (time.perf_counter() - input_started)
        output, network_forward_ms = _cuda_timed_forward(device, lambda: model(gray_tensor))
        logits, descriptors = output

    _cuda_sync(device)
    feature_started = time.perf_counter()
    features = extract_features(logits, descriptors)[0]
    _cuda_sync(device)
    feature_row = {name: value.detach().cpu().numpy() for name, value in features.items()}
    feature_extraction_ms = 1000.0 * (time.perf_counter() - feature_started)
    total_ms = 1000.0 * (time.perf_counter() - total_started)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    timing = {
        "method": method,
        "image_id": image_id,
        "condition": condition,
        "batch_size": 1,
        "single_image_inference": True,
        "network_forward_ms": network_forward_ms,
        "feature_extraction_ms": feature_extraction_ms,
        "input_processing_ms": input_processing_ms,
        "total_inference_ms": total_ms,
        "inference_ms": total_ms,
        "detected_points": int(len(feature_row["points"])),
        "detected_points_a": int(len(feature_row["points"])),
        "detected_points_b": None,
        "peak_memory_bytes": peak_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }
    return feature_row, timing


def load_cache_entry(path: str | Path) -> dict[str, Any]:
    entry = torch.load(path, map_location="cpu", weights_only=True)
    tensor = entry.get("noisy_bayer_dn")
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tuple(tensor.shape) != (HPATCHES_HEIGHT, HPATCHES_WIDTH):
        raise ValueError(f"HPatches cache must store FP32 480x640 Bayer DN: {path}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"HPatches cache contains non-finite Bayer values: {path}")
    return entry


@torch.no_grad()
def build_cache(
    manifest_path: str | Path,
    cache_root: str | Path,
    invisp: torch.nn.Module,
    eld: CanonELD,
    device: torch.device,
    *,
    sequence_names: Iterable[str] | None = None,
    batch_images: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("HPatches Raw cache generation requires CUDA")
    manifest, digest = read_manifest(manifest_path)
    selected_images = iter_manifest_images(manifest, sequence_names)
    cache_root = Path(cache_root)
    identity = cache_identity(digest, manifest["dataset_integrity_sha256"], invisp.rawfeat_repo_path, eld.calibration)
    identity_path = cache_root / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("HPatches cache identity differs; use a new cache directory")
    cache_root.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    timing_path = cache_root / "cache_timing.jsonl"
    timing_rows: list[str] = []
    completed = 0
    skipped = 0
    for image in selected_images:
        paths = {condition: cache_path(cache_root, image["id"], condition) for condition in CONDITIONS}
        if not force and all(path.is_file() for path in paths.values()):
            completed += len(CONDITIONS)
            skipped += len(CONDITIONS)
            continue
        rgb = torch.from_numpy(_prepare_image(Path(manifest["dataset_root"]) / image["path"], image["transform"]))
        rgb = rgb.permute(2, 0, 1).unsqueeze(0).to(device)
        _cuda_sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        sensor = invisp_to_sensor_rgb(invisp(rgb, rev=True))
        clean_bayer = sample_bayer(sensor)[0]
        _cuda_sync(device)
        base_ms = 1000.0 * (time.perf_counter() - start)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        for condition in CONDITIONS:
            if not force and paths[condition].is_file():
                completed += 1
                skipped += 1
                continue
            if condition == "clean":
                noisy = no_noise(clean_bayer)
                noise_seed = None
                ratio = None
            else:
                ratio = int(condition.removeprefix("ratio"))
                noise_seed = int(image["noise_seeds"][condition])
                noisy = eld.apply(clean_bayer, ratio, noise_seed)
            if noisy.dtype != torch.float32:
                noisy = noisy.float()
            stored = {
                "image_id": image["id"], "condition": condition, "ratio": ratio,
                "noise_seed": noise_seed, "transform": image["transform"],
                "noisy_bayer_dn": noisy.detach().cpu().contiguous(),
            }
            _atomic_torch_save(stored, paths[condition])
            completed += 1
            timing_rows.append(json.dumps({
                "image_id": image["id"], "condition": condition,
                "invisp_ms": base_ms, "peak_memory_bytes": peak_memory,
                "dtype": "float32", "path": str(paths[condition]),
            }, allow_nan=False))
        del rgb, sensor, clean_bayer
        torch.cuda.empty_cache()
    if timing_rows:
        with timing_path.open("a") as stream:
            stream.write("\n".join(timing_rows) + "\n")
    selected_sequences = sorted({image["sequence"] for image in selected_images})
    index = {
        "protocol": HPATCHES_PROTOCOL, "cache_identity_sha256": identity_digest(identity),
        "manifest_sha256": digest, "selected_sequences": selected_sequences,
        "selected_images": len(selected_images), "cached_cases": completed,
        "skipped_cases": skipped, "expected_cases": len(selected_images) * len(CONDITIONS),
        "full_protocol": len(selected_images) == 696 and completed == 696 * len(CONDITIONS),
    }
    (cache_root / "cache_index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def _load_features(
    method: str,
    images: list[dict[str, Any]],
    condition: str,
    cache_root: Path,
    device: torch.device,
    model: torch.nn.Module | None,
    superpoint: torch.nn.Module | None,
    batch_images: int,
    timing_rows: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    if batch_images != 1:
        raise ValueError("HPatches evaluation requires batch_size=1; larger batches are forbidden")
    feature_map: dict[str, dict[str, np.ndarray]] = {}
    selected_model = model if method == "rawfeat" else superpoint
    if selected_model is None:
        raise ValueError(f"{method} model is required")
    if images:
        warmup_image = images[0]
        warmup_bayer = load_cache_entry(
            cache_path(cache_root, warmup_image["id"], condition)
        )["noisy_bayer_dn"]
        _single_image_inference(
            method, selected_model, warmup_bayer, warmup_image["id"], condition, device
        )
    for image in images:
        noisy = load_cache_entry(cache_path(cache_root, image["id"], condition))["noisy_bayer_dn"]
        feature, timing = _single_image_inference(
            method, selected_model, noisy, image["id"], condition, device
        )
        feature_map[image["id"]] = feature
        timing_rows.append(timing)
    return feature_map


def _estimate_homography(feature_a: dict[str, np.ndarray], feature_b: dict[str, np.ndarray],
                         seed: int = HPATCHES_SEED) -> np.ndarray | None:
    index_a, index_b = mutual_nearest(feature_a["descriptors"], feature_b["descriptors"])
    if len(index_a) < 4:
        return None
    cv2.setRNGSeed(seed)
    estimated, _ = cv2.findHomography(
        feature_a["points"][index_a], feature_b["points"][index_b], cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=10000, confidence=0.999,
    )
    if estimated is None or not np.isfinite(estimated).all() or abs(float(np.linalg.det(estimated))) < 1e-12:
        return None
    estimated = estimated.astype(np.float64)
    estimated /= estimated[2, 2]
    return estimated


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _pair_row(method: str, pair: dict[str, Any], condition: str,
              feature_map: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    feature_a = feature_map[pair["reference_image_id"]]
    feature_b = feature_map[pair["target_image_id"]]
    homography = np.asarray(pair["H_processed"], dtype=np.float64)
    valid = np.ones((HPATCHES_HEIGHT, HPATCHES_WIDTH), dtype=bool)
    metrics = pair_metrics(feature_a, feature_b, homography, valid, valid,
                           reprojection_threshold=3.0, ransac_seed=HPATCHES_SEED)
    estimated = _estimate_homography(feature_a, feature_b)
    corner_error = metrics["corner_error"]
    finite_corner = np.isfinite(corner_error)
    row: dict[str, Any] = {
        "method": method, "sequence": pair["sequence"], "type": pair["type"],
        "reference_index": pair["reference_index"], "target_index": pair["target_index"],
        "condition": condition, "ratio_a": None if condition == "clean" else int(condition.removeprefix("ratio")),
        "ratio_b": None if condition == "clean" else int(condition.removeprefix("ratio")),
        "H_processed": homography.tolist(), "H_estimated": None if estimated is None else estimated.tolist(),
        "homography_failure": not finite_corner,
        "threshold_success_1": int(finite_corner and corner_error <= 1.0),
        "threshold_success_3": int(finite_corner and corner_error <= 3.0),
        "threshold_success_5": int(finite_corner and corner_error <= 5.0),
    }
    row.update({key: _serializable(value) for key, value in metrics.items()})
    return row


def _mean_finite(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row[key])]
    return float(np.mean(values)) if values else None


def _aggregate_sequence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["method"], row["sequence"], row["type"], row["condition"]) for row in rows})
    sequence_rows: list[dict[str, Any]] = []
    for method, sequence, sequence_type, condition in keys:
        group = [row for row in rows if (row["method"], row["sequence"], row["type"], row["condition"]) ==
                 (method, sequence, sequence_type, condition)]
        summary: dict[str, Any] = {
            "method": method, "sequence": sequence, "type": sequence_type,
            "condition": condition, "pair_count": len(group),
            "homography_failures": sum(row["homography_failure"] for row in group),
        }
        for key in AVERAGE_METRICS:
            summary[key] = _mean_finite(group, key)
        for key in COUNT_METRICS:
            summary[f"{key}_sum"] = int(sum(int(row[key]) for row in group))
        sequence_rows.append(summary)
    return sequence_rows


def _group_summary(sequence_rows: list[dict[str, Any]], method: str, group_name: str,
                   condition: str) -> dict[str, Any]:
    selected = [row for row in sequence_rows if row["method"] == method and row["condition"] == condition]
    if group_name != "all":
        selected = [row for row in selected if row["type"] == group_name]
    result: dict[str, Any] = {
        "method": method, "group": group_name, "condition": condition,
        "sequence_count": len(selected), "pair_count": int(sum(row["pair_count"] for row in selected)),
        "homography_failures": int(sum(row["homography_failures"] for row in selected)),
    }
    for key in AVERAGE_METRICS:
        result[key] = _mean_finite(selected, key)
    for key in COUNT_METRICS:
        result[f"{key}_mean_per_sequence"] = _mean_finite(selected, f"{key}_sum")
    return result


def _corner_distribution(rows: list[dict[str, Any]], method: str, group_name: str,
                         condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method and row["condition"] == condition
                and (group_name == "all" or row["type"] == group_name)]
    values = np.asarray([row["corner_error"] for row in selected
                         if row["corner_error"] is not None and np.isfinite(row["corner_error"])], dtype=np.float64)
    return {
        "corner_error_median": float(np.median(values)) if len(values) else None,
        "corner_error_p95": float(np.quantile(values, 0.95)) if len(values) else None,
        "corner_error_gt20": int((values > 20.0).sum()),
        "corner_error_finite": int(len(values)),
    }


def _bootstrap(sequence_rows: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    eligible = [row for row in sequence_rows if row["condition"] in NOISY_CONDITIONS
                and (group_name == "all" or row["type"] == group_name)]
    sequences = sorted({row["sequence"] for row in eligible})
    if not sequences:
        return {"group": group_name, "iterations": BOOTSTRAP_ITERATIONS, "sequence_count": 0,
                "estimate": None, "ci95": [None, None], "positive_sequences": 0, "negative_sequences": 0}
    differences = []
    for sequence in sequences:
        rawfeat_values = {row["condition"]: row["h_auc_5"] for row in sequence_rows
                          if row["method"] == "rawfeat" and row["sequence"] == sequence
                          and row["condition"] in NOISY_CONDITIONS}
        superpoint_values = {row["condition"]: row["h_auc_5"] for row in sequence_rows
                           if row["method"] == "superpoint" and row["sequence"] == sequence
                           and row["condition"] in NOISY_CONDITIONS}
        if set(rawfeat_values) != set(NOISY_CONDITIONS) or set(superpoint_values) != set(NOISY_CONDITIONS):
            continue
        differences.append(float(np.mean([rawfeat_values[name] - superpoint_values[name] for name in NOISY_CONDITIONS])))
    values = np.asarray(differences, dtype=np.float64)
    if not len(values):
        return {"group": group_name, "iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED,
                "sequence_count": 0, "estimate": None, "ci95": [None, None],
                "positive_sequences": 0, "negative_sequences": 0, "zero_sequences": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_ITERATIONS, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "group": group_name, "iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED,
        "sequence_count": int(len(values)), "estimate": float(values.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "positive_sequences": int((values > 0).sum()), "negative_sequences": int((values < 0).sum()),
        "zero_sequences": int((values == 0).sum()),
    }


def _visualize_pair(
    method: str, row: dict[str, Any], feature_map: dict[str, dict[str, np.ndarray]],
    cache_root: Path, output: Path, image_ids: tuple[str, str],
) -> None:
    entry_a = load_cache_entry(cache_path(cache_root, image_ids[0], row["condition"]))
    entry_b = load_cache_entry(cache_path(cache_root, image_ids[1], row["condition"]))
    if method == "superpoint":
        gray_a = np.clip(baseline_gray(entry_a["noisy_bayer_dn"].numpy()) * 255.0, 0, 255).astype(np.uint8)
        gray_b = np.clip(baseline_gray(entry_b["noisy_bayer_dn"].numpy()) * 255.0, 0, 255).astype(np.uint8)
    else:
        gray_a = np.clip((entry_a["noisy_bayer_dn"].numpy() - 2048.0) / (16383.0 - 2048.0) * 255.0, 0, 255).astype(np.uint8)
        gray_b = np.clip((entry_b["noisy_bayer_dn"].numpy() - 2048.0) / (16383.0 - 2048.0) * 255.0, 0, 255).astype(np.uint8)
    image_canvas = cv2.cvtColor(np.concatenate((gray_a, gray_b), axis=1), cv2.COLOR_GRAY2BGR)
    panel_height = 104
    canvas = np.zeros((panel_height + image_canvas.shape[0], image_canvas.shape[1], 3), dtype=np.uint8)
    canvas[panel_height:] = image_canvas
    cv2.putText(canvas, f"{method} | {row['sequence']} | target {row['target_index']} | {row['condition']}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "reference (A)", (10, panel_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(canvas, "target (B)", (HPATCHES_WIDTH + 10, panel_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)
    features_a, features_b = feature_map[image_ids[0]], feature_map[image_ids[1]]
    index_a, index_b = mutual_nearest(features_a["descriptors"], features_b["descriptors"])
    homography = np.asarray(row["H_processed"], dtype=np.float64)
    correct_matches = 0
    for left, right in zip(features_a["points"][index_a], features_b["points"][index_b]):
        projected = warp_points(np.asarray(left, dtype=np.float32)[None], homography)[0]
        correct = float(np.linalg.norm(projected - right)) <= 3.0
        correct_matches += int(correct)
        color = (0, 180, 0) if correct else (0, 0, 180)
        left_point = np.rint(left).astype(int) + [0, panel_height]
        right_point = np.rint(right).astype(int) + [HPATCHES_WIDTH, panel_height]
        cv2.line(canvas, tuple(left_point), tuple(right_point), color, 1)
    for points, offset in ((features_a["points"], 0), (features_b["points"], HPATCHES_WIDTH)):
        for point in points:
            cv2.circle(canvas, (round(float(point[0])) + offset, round(float(point[1])) + panel_height), 1, (0, 220, 255), -1)
    corners = np.array([[0, 0], [HPATCHES_WIDTH - 1, 0], [HPATCHES_WIDTH - 1, HPATCHES_HEIGHT - 1], [0, HPATCHES_HEIGHT - 1]], np.float32)
    true_corners = warp_points(corners, np.asarray(row["H_processed"], dtype=np.float64))
    cv2.polylines(canvas, [np.rint(true_corners).astype(np.int32) + [HPATCHES_WIDTH, panel_height]], True, (255, 0, 0), 2)
    if row["H_estimated"] is not None:
        estimated_corners = warp_points(corners, np.asarray(row["H_estimated"], dtype=np.float64))
        cv2.polylines(canvas, [np.rint(estimated_corners).astype(np.int32) + [HPATCHES_WIDTH, panel_height]], True, (0, 255, 255), 2)
    failure = "yes" if row["homography_failure"] else "no"
    metrics_text = (
        f"points A/B {len(features_a['points'])}/{len(features_b['points'])} | "
        f"MNN {len(index_a)} | correct {correct_matches} | "
        f"precision {_serializable(row.get('match_precision'))} | "
        f"H-AUC@5 {_serializable(row.get('h_auc_5'))} | "
        f"corner px {_serializable(row.get('corner_error'))} | failure {failure}"
    )
    cv2.putText(canvas, metrics_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT H", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "estimated H", (70, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "green correct / red incorrect MNN", (180, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA)
    output.mkdir(parents=True, exist_ok=True)
    filename = f"{method}_{row['sequence']}_{row['target_index']}_{row['condition']}.png"
    cv2.imwrite(str(output / filename), canvas)


def _write_visualization_indexes(output: Path, filenames: list[str]) -> dict[str, Any]:
    """Write filterable visualization pages without making a second evaluation pass."""
    visualization = output / "visualizations"
    visualization.mkdir(parents=True, exist_ok=True)
    unique_names = sorted(set(filenames))
    cards = []
    for filename in unique_names:
        parts = Path(filename).stem.split("_")
        method = parts[0] if parts else ""
        condition = next((name for name in CONDITIONS if name in parts), "")
        cards.append(
            f'<a class="card" data-method="{html.escape(method)}" data-condition="{html.escape(condition)}" '
            f'href="{html.escape(filename)}"><img src="{html.escape(filename)}" alt="{html.escape(Path(filename).stem)}">'
            f'<span>{html.escape(Path(filename).stem)}</span></a>'
        )
    cards_html = "".join(cards)
    controls = (
        '<label>method <select id="method"><option value="all">all</option>'
        '<option value="rawfeat">RawFeat</option><option value="superpoint">SuperPoint</option></select></label>'
        '<label>condition <select id="condition"><option value="all">all</option>'
        + "".join(f'<option value="{name}">{name}</option>' for name in CONDITIONS)
        + '</select></label>'
    )
    common = (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font:14px system-ui,sans-serif;margin:20px;background:#f8fafc;color:#172033}label{margin-right:14px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px;margin-top:16px}.card{background:#fff;border:1px solid #dbe3ee;padding:8px;color:#172033;text-decoration:none}.card img{width:100%;display:block}.card span{display:block;overflow-wrap:anywhere;margin-top:5px}.muted{color:#64748b}</style>'
    )
    index_html = common + f'</head><body><h1>HPatches visualizations</h1><p class="muted">{len(unique_names)} images; fixed samples use sequences i_ajuntament, i_autannes, v_abstract, v_adam and targets 2, 3.</p>{controls}<div class="grid">{cards_html}</div><script>function filter(){{const m=document.getElementById("method").value,c=document.getElementById("condition").value;document.querySelectorAll(".card").forEach(x=>x.style.display=(m==="all"||x.dataset.method===m)&&(c==="all"||x.dataset.condition===c)?"":"none")}}document.querySelectorAll("select").forEach(x=>x.onchange=filter)</script></body></html>'
    (visualization / "index.html").write_text(index_html)

    for condition in CONDITIONS:
        condition_cards = [name for name in unique_names if condition in Path(name).stem.split("_")]
        page = common + f'</head><body><h1>{condition} visualizations</h1><p><a href="index.html">all conditions</a></p><div class="grid">'
        page += "".join(f'<a class="card" href="{html.escape(name)}"><img src="{html.escape(name)}"><span>{html.escape(Path(name).stem)}</span></a>' for name in condition_cards)
        (visualization / f"condition_{condition}.html").write_text(page + "</div></body></html>")

    pair_keys = {}
    for name in unique_names:
        stem = Path(name).stem
        method = stem.split("_")[0]
        pair_key = stem.removeprefix(f"{method}_")
        pair_keys.setdefault(pair_key, {})[method] = name
    comparison_items = []
    for key, methods in sorted(pair_keys.items()):
        comparison_items.append(f'<h3>{html.escape(key)}</h3><div class="grid">' + "".join(
            f'<a class="card" href="{html.escape(name)}"><img src="{html.escape(name)}"><span>{method}</span></a>'
            for method, name in sorted(methods.items())
        ) + "</div>")
    (visualization / "rawfeat_superpoint.html").write_text(common + '</head><body><h1>RawFeat / SuperPoint comparison</h1><p><a href="index.html">all conditions</a></p>' + "".join(comparison_items) + "</body></html>")
    contact_sections = []
    for condition in CONDITIONS:
        contact_sections.append(f'<h2>{condition}</h2><div class="grid">' + "".join(
            f'<a class="card" href="{html.escape(name)}"><img src="{html.escape(name)}"><span>{html.escape(Path(name).stem)}</span></a>'
            for name in unique_names if condition in Path(name).stem.split("_")
        ) + "</div>")
    (visualization / "contact_sheet.html").write_text(common + '</head><body><h1>Six-condition overview</h1><p><a href="index.html">filterable index</a></p>' + "".join(contact_sections) + "</body></html>")
    return {
        "count": len(unique_names),
        "index": "visualizations/index.html",
        "condition_pages": [f"visualizations/condition_{condition}.html" for condition in CONDITIONS],
        "rawfeat_superpoint_page": "visualizations/rawfeat_superpoint.html",
        "contact_sheet_page": "visualizations/contact_sheet.html",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def _validate_timing_rows(rows: list[dict[str, Any]], expected_rows: int) -> None:
    """Reject timing data that cannot be traced to one image invocation."""
    required = {
        "method", "image_id", "condition", "batch_size", "single_image_inference",
        "network_forward_ms", "feature_extraction_ms", "total_inference_ms",
        "detected_points", "peak_memory_bytes",
    }
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} single-image timing rows, found {len(rows)}")
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("HPatches timing rows are missing required single-image fields")
    if any(row["batch_size"] != 1 or not row["single_image_inference"] for row in rows):
        raise RuntimeError("HPatches timing records do not correspond one-to-one with single-image inference")
    identities = {(row["method"], row["image_id"], row["condition"]) for row in rows}
    if len(identities) != len(rows):
        raise RuntimeError("HPatches timing rows contain duplicate image/method/condition identities")


DEFAULT_DEPLOYMENT_SMOKE_SEQUENCES = ("i_ajuntament", "i_autannes", "v_abstract", "v_adam")
DEFAULT_VISUALIZATION_SEQUENCES = DEFAULT_DEPLOYMENT_SMOKE_SEQUENCES
VISUALIZATION_TARGETS = (2, 3)


def _dense_difference(before: torch.Tensor, after: torch.Tensor) -> dict[str, float]:
    delta = (before - after).abs()
    return {"max_abs": float(delta.max().item()), "mean_abs": float(delta.mean().item())}


def _pair_mnn_comparison(
    before_a: dict[str, np.ndarray], before_b: dict[str, np.ndarray],
    after_a: dict[str, np.ndarray], after_b: dict[str, np.ndarray],
) -> dict[str, Any]:
    before_a_index, before_b_index = mutual_nearest(before_a["descriptors"], before_b["descriptors"])
    after_a_index, after_b_index = mutual_nearest(after_a["descriptors"], after_b["descriptors"])
    before_coordinates = {
        (*before_a["points"][a], *before_b["points"][b])
        for a, b in zip(before_a_index, before_b_index)
    }
    after_coordinates = {
        (*after_a["points"][a], *after_b["points"][b])
        for a, b in zip(after_a_index, after_b_index)
    }
    common_a = sorted(set(map(tuple, before_a["points"])) & set(map(tuple, after_a["points"])))
    common_b = sorted(set(map(tuple, before_b["points"])) & set(map(tuple, after_b["points"])))
    before_a_lookup = {tuple(point): index for index, point in enumerate(before_a["points"])}
    after_a_lookup = {tuple(point): index for index, point in enumerate(after_a["points"])}
    before_b_lookup = {tuple(point): index for index, point in enumerate(before_b["points"])}
    after_b_lookup = {tuple(point): index for index, point in enumerate(after_b["points"])}
    before_affinity = (
        before_a["descriptors"][[before_a_lookup[point] for point in common_a]]
        @ before_b["descriptors"][[before_b_lookup[point] for point in common_b]].T
        if common_a and common_b else np.empty((0, 0), dtype=np.float32)
    )
    after_affinity = (
        after_a["descriptors"][[after_a_lookup[point] for point in common_a]]
        @ after_b["descriptors"][[after_b_lookup[point] for point in common_b]].T
        if common_a and common_b else np.empty((0, 0), dtype=np.float32)
    )
    affinity_error = float(np.abs(before_affinity - after_affinity).max()) if before_affinity.size else 0.0
    tolerance = 2.0 * affinity_error + 8.0 * np.finfo(np.float32).eps
    for old, new in ((before_affinity, after_affinity), (before_affinity.T, after_affinity.T)):
        changed = np.where(old.argmax(1) != new.argmax(1))[0] if old.size else []
        if len(changed):
            gaps = old[changed].max(1) - old[changed, new[changed].argmax(1)]
            if float(gaps.max()) > tolerance:
                raise RuntimeError("fusion changed an MNN neighbour away from a numeric tie")
    return {
        "mnn_coordinate_set_changed": before_coordinates != after_coordinates,
        "mnn_count_before": len(before_coordinates),
        "mnn_count_after": len(after_coordinates),
        "descriptor_affinity_max_abs_error": affinity_error,
        "mnn_numeric_boundary_verified": True,
    }


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "h_auc_1", "h_auc_3", "h_auc_5", "repeatability", "localization_error",
        "match_precision", "corner_error", "detected_a", "detected_b", "matches",
        "correct_matches", "homography_failure",
    )
    result: dict[str, Any] = {}
    for key in keys:
        left, right = before.get(key), after.get(key)
        if isinstance(left, bool) or isinstance(right, bool):
            result[key] = bool(right) != bool(left)
        elif left is None or right is None or not np.isfinite(left) or not np.isfinite(right):
            result[key] = None
        else:
            result[key] = float(right - left)
    return result


@torch.no_grad()
def compare_deployment_smoke(
    manifest_path: str | Path,
    cache_root: str | Path,
    output: str | Path,
    *,
    training_model: torch.nn.Module,
    deployment_model: torch.nn.Module,
    checkpoint_path: str | Path,
    deployment_path: str | Path,
    sequence_names: Iterable[str] = DEFAULT_DEPLOYMENT_SMOKE_SEQUENCES,
    batch_images: int = 1,
    latency_warmup: int = 20,
    latency_repeats: int = 100,
    command: str | None = None,
) -> dict[str, Any]:
    """Compare unfused and reloaded deployment models on the fixed 20-pair smoke."""
    if not torch.cuda.is_available():
        raise RuntimeError("deployment smoke requires CUDA")
    if batch_images != 1:
        raise ValueError("deployment smoke requires batch_size=1; larger batches are forbidden")
    if latency_warmup < 0 or latency_repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    training_model.eval()
    deployment_model.eval()
    from .export import (
        _deployment_path,
        assert_deployment_structure,
        benchmark_model_latency,
        compare_feature_pair,
        load_export,
    )

    assert_deployment_structure(deployment_model)
    manifest, manifest_sha256 = read_manifest(manifest_path)
    cache_root = Path(cache_root)
    identity = json.loads((cache_root / "identity.json").read_text())
    if identity.get("manifest_sha256") != manifest_sha256 or identity.get("protocol") != HPATCHES_PROTOCOL:
        raise RuntimeError("HPatches cache identity differs from the selected manifest")
    selected_sequences = tuple(sequence_names)
    pairs = iter_manifest_pairs(manifest, selected_sequences)
    if len(selected_sequences) != 4 or len(pairs) != 20:
        raise ValueError("deployment smoke requires exactly four sequences and twenty base pairs")
    if (
        sum(sequence.startswith("i_") for sequence in selected_sequences) != 2
        or sum(sequence.startswith("v_") for sequence in selected_sequences) != 2
    ):
        raise ValueError("deployment smoke requires illumination and viewpoint sequences")
    deployment_file = _deployment_path(deployment_path).resolve()
    image_records = {image["id"]: image for image in iter_manifest_images(manifest, selected_sequences)}
    image_ids = sorted({pair["reference_image_id"] for pair in pairs} | {pair["target_image_id"] for pair in pairs})
    images = [image_records[image_id] for image_id in image_ids]
    device = next(deployment_model.parameters()).device
    if device.type != "cuda" or next(training_model.parameters()).device.type != "cuda":
        raise RuntimeError("deployment smoke must run on CUDA")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    feature_rows: list[dict[str, Any]] = []
    smoke_timing_rows: list[dict[str, Any]] = []
    latency_batches: dict[str, dict[int, list[dict[str, Any]]]] = {}
    pair_inputs: dict[str, list[dict[str, Any]]] = {}
    all_pair_rows: dict[str, list[dict[str, Any]]] = {"training": [], "deployed": []}

    for condition in CONDITIONS:
        raw_entries = {
            image["id"]: load_cache_entry(cache_path(cache_root, image["id"], condition))
            for image in images
        }
        batches = [images[index:index + batch_images] for index in range(0, len(images), batch_images)]
        before_map: dict[str, dict[str, np.ndarray]] = {}
        after_map: dict[str, dict[str, np.ndarray]] = {}
        for batch_index, batch in enumerate(batches):
            noisy = torch.stack([raw_entries[image["id"]]["noisy_bayer_dn"] for image in batch]).to(device)
            packed = pack_student(noisy)
            torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            before_output = training_model(packed)
            after_output = deployment_model(packed)
            torch.cuda.synchronize(device)
            batch_elapsed = 1000.0 * (time.perf_counter() - batch_started)
            smoke_timing_rows.append({
                "condition": condition, "batch_index": batch_index, "batch_size": len(batch),
                "image_ids": [image["id"] for image in batch], "elapsed_ms": batch_elapsed,
            })
            before_features = extract_features(before_output["logits"], before_output["descriptors"])
            after_features = extract_features(after_output["logits"], after_output["descriptors"])
            for index, (image, before_feature, after_feature) in enumerate(
                zip(batch, before_features, after_features)
            ):
                before_numpy = {name: value.detach().cpu().numpy() for name, value in before_feature.items()}
                after_numpy = {name: value.detach().cpu().numpy() for name, value in after_feature.items()}
                before_map[image["id"]] = before_numpy
                after_map[image["id"]] = after_numpy
                logits_delta = _dense_difference(before_output["logits"][index], after_output["logits"][index])
                descriptor_delta = _dense_difference(before_output["descriptors"][index], after_output["descriptors"][index])
                score_before = score_map(before_output["logits"][index:index + 1])[0].detach().cpu().numpy()
                score_after = score_map(after_output["logits"][index:index + 1])[0].detach().cpu().numpy()
                comparison = compare_feature_pair(before_numpy, after_numpy, score_before, score_after)
                feature_rows.append({
                    "image_id": image["id"], "sequence": image["sequence"], "condition": condition,
                    "logits": logits_delta, "descriptors": descriptor_delta,
                    "score_map_max_abs_error": comparison["score_map_max_abs_error"],
                    "score_map_mean_abs_error": comparison["score_map_mean_abs_error"],
                    "point_order_changed": comparison["point_order_changed"],
                    "point_set_changed": comparison["point_set_changed"],
                    "detected_before": comparison["detected_before"],
                    "detected_after": comparison["detected_after"],
                    "descriptor_max_abs_error": comparison["descriptor_max_abs_error"],
                    "descriptor_mean_abs_error": comparison["descriptor_mean_abs_error"],
                    "selected_score_max_abs_error": comparison["selected_score_max_abs_error"],
                    "selected_score_mean_abs_error": comparison["selected_score_mean_abs_error"],
                    "discrete_change_reasons": comparison["discrete_change_reasons"],
                    "numeric_boundaries_verified": comparison["numeric_boundaries_verified"],
                })
        latency_batches[condition] = {1: []}
        for image in images:
            bayer = raw_entries[image["id"]]["noisy_bayer_dn"][None].to(device)
            latency_batches[condition][1].append({
                "image_ids": [image["id"]],
                "bayer": bayer,
                "packed": pack_student(bayer),
            })
        pair_inputs[condition] = []
        for pair in pairs:
            pair_inputs[condition].append({
                "image_ids": [pair["reference_image_id"], pair["target_image_id"]],
                "feature_a": after_map[pair["reference_image_id"]],
                "feature_b": after_map[pair["target_image_id"]],
                "seed": HPATCHES_SEED,
            })
            before_row = _pair_row("training", pair, condition, before_map)
            after_row = _pair_row("deployed", pair, condition, after_map)
            mnn = _pair_mnn_comparison(
                before_map[pair["reference_image_id"]], before_map[pair["target_image_id"]],
                after_map[pair["reference_image_id"]], after_map[pair["target_image_id"]],
            )
            comparison = {**mnn, "metrics": _metric_delta(before_row, after_row)}
            after_row["fusion_comparison"] = comparison
            before_row["fusion_comparison"] = comparison
            all_pair_rows["training"].append(before_row)
            all_pair_rows["deployed"].append(after_row)

    restored = load_export(deployment_file, device)
    reload_max: dict[str, float] = {name: 0.0 for name in ("logits", "descriptors")}
    for condition in CONDITIONS:
        for batch in latency_batches[condition][1]:
            deployed_output = deployment_model(batch["packed"])
            restored_output = restored(batch["packed"])
            for name in reload_max:
                reload_max[name] = max(reload_max[name], float((deployed_output[name] - restored_output[name]).abs().max()))
                torch.testing.assert_close(deployed_output[name], restored_output[name], atol=0, rtol=0)
    latency = benchmark_model_latency(
        restored, latency_batches, pair_inputs, device,
        warmup=latency_warmup, repeats=latency_repeats,
    )
    sequence_rows = _aggregate_sequence_rows(all_pair_rows["training"] + all_pair_rows["deployed"])
    groups = ("all", "illumination", "viewpoint")
    summaries = {
        method: {
            group: {
                condition: _group_summary(sequence_rows, method, group, condition)
                for condition in CONDITIONS
            }
            for group in groups
        }
        for method in ("training", "deployed")
    }
    for method in summaries:
        for group in groups:
            for condition in CONDITIONS:
                summaries[method][group][condition].update(
                    _corner_distribution(all_pair_rows[method], method, group, condition)
                )
    main_scores = {
        method: float(np.mean([summaries[method]["all"][condition]["h_auc_5"] for condition in NOISY_CONDITIONS]))
        for method in ("training", "deployed")
    }
    metric_deltas = {
        group: {
            condition: {
                key: summaries["deployed"][group][condition].get(key) - summaries["training"][group][condition].get(key)
                if summaries["deployed"][group][condition].get(key) is not None
                and summaries["training"][group][condition].get(key) is not None else None
                for key in ("h_auc_1", "h_auc_3", "h_auc_5", "detected_a", "detected_b", "matches", "correct_matches")
            }
            for condition in CONDITIONS
        }
        for group in groups
    }
    feature_comparison_by_condition = {}
    for condition in CONDITIONS:
        rows = [row for row in feature_rows if row["condition"] == condition]
        feature_comparison_by_condition[condition] = {
            "images": len(rows),
            "logits_max_abs_error": max(row["logits"]["max_abs"] for row in rows),
            "logits_mean_abs_error": float(np.mean([row["logits"]["mean_abs"] for row in rows])),
            "descriptors_max_abs_error": max(row["descriptors"]["max_abs"] for row in rows),
            "descriptors_mean_abs_error": float(np.mean([row["descriptors"]["mean_abs"] for row in rows])),
            "score_map_max_abs_error": max(row["score_map_max_abs_error"] for row in rows),
            "score_map_mean_abs_error": float(np.mean([row["score_map_mean_abs_error"] for row in rows])),
            "selected_score_max_abs_error": max(row["selected_score_max_abs_error"] for row in rows),
            "selected_score_mean_abs_error": float(np.mean([row["selected_score_mean_abs_error"] for row in rows])),
            "point_order_changes": int(sum(row["point_order_changed"] for row in rows)),
            "point_set_changes": int(sum(row["point_set_changed"] for row in rows)),
            "discrete_change_reasons": sorted({
                reason for row in rows for reason in row["discrete_change_reasons"]
            }),
        }
    pair_comparisons = [row["fusion_comparison"] for row in all_pair_rows["deployed"]]
    reason_counts = {
        reason: int(sum(reason in row["discrete_change_reasons"] for row in feature_rows))
        for reason in ("score_order_near_tie", "nms_near_tie", "threshold_boundary", "topk_boundary")
    }
    discrete_checks = {
        "nms_changes": reason_counts["nms_near_tie"],
        "threshold_changes": reason_counts["threshold_boundary"],
        "topk_changes": reason_counts["topk_boundary"],
        "score_order_changes": reason_counts["score_order_near_tie"],
        "mnn_coordinate_set_changes": int(sum(row["mnn_coordinate_set_changed"] for row in pair_comparisons)),
        "all_changes_explained": True,
    }
    feature_summary = {
        "max_logits_abs_error": max(row["logits"]["max_abs"] for row in feature_rows),
        "mean_logits_abs_error": float(np.mean([row["logits"]["mean_abs"] for row in feature_rows])),
        "max_descriptor_abs_error": max(row["descriptors"]["max_abs"] for row in feature_rows),
        "mean_descriptor_abs_error": float(np.mean([row["descriptors"]["mean_abs"] for row in feature_rows])),
        "point_order_changes": int(sum(row["point_order_changed"] for row in feature_rows)),
        "point_set_changes": int(sum(row["point_set_changed"] for row in feature_rows)),
        "mnn_coordinate_set_changes": int(sum(row["mnn_coordinate_set_changed"] for row in pair_comparisons)),
        "max_descriptor_affinity_abs_error": max(
            row["descriptor_affinity_max_abs_error"] for row in pair_comparisons
        ),
        "discrete_checks": discrete_checks,
        "numeric_boundaries_verified": all(row["numeric_boundaries_verified"] for row in feature_rows),
        "by_condition": feature_comparison_by_condition,
    }
    deployment_metadata = json.loads((deployment_file.parent / "metadata.json").read_text())
    training_checkpoint_sha256 = file_digest(checkpoint_path)
    if deployment_metadata.get("source_checkpoint_sha256") != training_checkpoint_sha256:
        raise RuntimeError("deployment source checkpoint does not match the smoke checkpoint")
    run_manifest = {
        "protocol": HPATCHES_PROTOCOL, "manifest_sha256": manifest_sha256,
        "cache_identity_sha256": identity_digest(identity), "selected_sequences": list(selected_sequences),
        "selected_pairs": len(pairs), "conditions": list(CONDITIONS),
        "training_checkpoint": str(Path(checkpoint_path).resolve()),
        "training_checkpoint_sha256": training_checkpoint_sha256,
        "deployment_path": str(deployment_file),
        "deployment_sha256": file_digest(deployment_file),
        "deployment_metadata": deployment_metadata,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "command": command,
        "full_protocol": False, "single_image_inference": True, "batch_size": 1,
    }
    (output_path / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, allow_nan=False) + "\n")
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output_path / "cache_identity.json").write_text(json.dumps(identity, indent=2, allow_nan=False) + "\n")
    _write_jsonl(output_path / "training" / "per_pair.jsonl", all_pair_rows["training"])
    _write_jsonl(output_path / "deployed" / "per_pair.jsonl", all_pair_rows["deployed"])
    _write_jsonl(output_path / "per_pair_comparison.jsonl", [
        {"training": before, "deployed": after, "metric_delta": after["fusion_comparison"]["metrics"]}
        for before, after in zip(all_pair_rows["training"], all_pair_rows["deployed"])
    ])
    _write_jsonl(output_path / "feature_comparison.jsonl", feature_rows)
    _write_jsonl(output_path / "smoke_timing.jsonl", smoke_timing_rows)
    _write_jsonl(output_path / "sequence_summary.jsonl", sequence_rows)
    _write_jsonl(output_path / "latency_measurements.jsonl", latency["measurements"])
    latency_summary = {
        "protocol": HPATCHES_PROTOCOL, "deployment_path": str(deployment_file),
        "warmup": latency_warmup, "repeats": latency_repeats,
        "precision": "FP32", "input_sizes": {"packed": [4, 240, 320], "bayer": [480, 640]},
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda,
        "summaries": latency["summary"],
    }
    (output_path / "latency_summary.json").write_text(json.dumps(latency_summary, indent=2, allow_nan=False) + "\n")
    summary = {
        "protocol": HPATCHES_PROTOCOL, "manifest_sha256": manifest_sha256,
        "cache_identity_sha256": identity_digest(identity), "selected_sequences": list(selected_sequences),
        "conditions": list(CONDITIONS), "pair_count_per_condition": len(pairs),
        "full_protocol": False, "methods": ["training", "deployed"],
        "main_h_auc_5": main_scores,
        "main_h_auc_5_delta_percentage_points": 100 * (main_scores["deployed"] - main_scores["training"]),
        "summaries": summaries, "metric_deltas": metric_deltas,
        "feature_comparison": feature_summary, "reload_check": {
            "max_abs": reload_max, "exact": all(value == 0.0 for value in reload_max.values()),
        },
        "gray_branch_computed": False,
        "latency_summary": "latency_summary.json",
        "total_wall_clock_seconds": time.perf_counter() - started,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


@torch.no_grad()
def evaluate_hpatches(
    manifest_path: str | Path,
    cache_root: str | Path,
    output: str | Path,
    *,
    model: torch.nn.Module,
    superpoint: torch.nn.Module,
    checkpoint_path: str | Path,
    superpoint_checkpoint_path: str | Path,
    sequence_names: Iterable[str] | None = None,
    max_pairs: int | None = None,
    target_indices: Iterable[int] | None = None,
    batch_images: int = 1,
    command: str | None = None,
) -> dict[str, Any]:
    if batch_images != 1:
        raise ValueError("HPatches evaluation requires batch_size=1; larger batches are forbidden")
    started = time.perf_counter()
    manifest, digest = read_manifest(manifest_path)
    cache_root = Path(cache_root)
    identity_path = cache_root / "identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(f"HPatches cache identity is missing: {identity_path}")
    identity = json.loads(identity_path.read_text())
    if identity.get("manifest_sha256") != digest or identity.get("protocol") != HPATCHES_PROTOCOL:
        raise RuntimeError("HPatches cache identity differs from the selected manifest")
    pairs = iter_manifest_pairs(manifest, sequence_names, max_pairs, target_indices)
    if not pairs:
        raise ValueError("no HPatches pairs selected")
    image_ids = sorted({pair["reference_image_id"] for pair in pairs} | {pair["target_image_id"] for pair in pairs})
    image_records = {image["id"]: image for image in iter_manifest_images(manifest, sequence_names)}
    images = [image_records[image_id] for image_id in image_ids]
    for image in images:
        for condition in CONDITIONS:
            path = cache_path(cache_root, image["id"], condition)
            if not path.is_file():
                raise FileNotFoundError(f"missing HPatches image cache: {path}")
            load_cache_entry(path)
    device = next(model.parameters()).device
    if model.training or superpoint.training:
        raise ValueError("HPatches evaluation requires eval-mode rawfeat and superpoint")
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"evaluation output is not empty; use a new directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, Any]] = []
    rows_by_method: dict[str, list[dict[str, Any]]] = {"rawfeat": [], "superpoint": []}
    pair_lookup = {(pair["sequence"], pair["target_index"]): pair for pair in pairs}
    selected_sequences = tuple(sorted({pair["sequence"] for pair in pairs}))
    visual_sequences = tuple(name for name in DEFAULT_VISUALIZATION_SEQUENCES if name in selected_sequences)
    visual_pairs = {
        (sequence, target): pair_lookup[(sequence, target)]
        for sequence in visual_sequences
        for target in VISUALIZATION_TARGETS
        if (sequence, target) in pair_lookup
    }
    visualization_filenames: list[str] = []
    visualization_dir = output_path / "visualizations"
    for condition in CONDITIONS:
        rawfeat_features = _load_features(
            "rawfeat", images, condition, cache_root, device, model, None, batch_images, timing_rows
        )
        superpoint_features_map = _load_features(
            "superpoint", images, condition, cache_root, device, None, superpoint, batch_images, timing_rows
        )
        for pair in pairs:
            rawfeat_row = _pair_row("rawfeat", pair, condition, rawfeat_features)
            superpoint_row = _pair_row("superpoint", pair, condition, superpoint_features_map)
            rows_by_method["rawfeat"].append(rawfeat_row)
            rows_by_method["superpoint"].append(superpoint_row)
            visual_pair = visual_pairs.get((pair["sequence"], pair["target_index"]))
            if visual_pair is not None:
                _visualize_pair(
                    "rawfeat", rawfeat_row, rawfeat_features, cache_root, visualization_dir,
                    (visual_pair["reference_image_id"], visual_pair["target_image_id"]),
                )
                visualization_filenames.append(
                    f"rawfeat_{pair['sequence']}_{pair['target_index']}_{condition}.png"
                )
                _visualize_pair(
                    "superpoint", superpoint_row, superpoint_features_map, cache_root, visualization_dir,
                    (visual_pair["reference_image_id"], visual_pair["target_image_id"]),
                )
                visualization_filenames.append(
                    f"superpoint_{pair['sequence']}_{pair['target_index']}_{condition}.png"
                )
        del rawfeat_features, superpoint_features_map

    expected_timing_rows = len(images) * len(CONDITIONS) * 2
    _validate_timing_rows(timing_rows, expected_timing_rows)
    selected_target_indices = sorted({pair["target_index"] for pair in pairs})
    visualization_info = _write_visualization_indexes(output_path, visualization_filenames)
    all_rows = rows_by_method["rawfeat"] + rows_by_method["superpoint"]
    sequence_rows = _aggregate_sequence_rows(all_rows)
    groups = ("all", "illumination", "viewpoint")
    summaries = {
        method: {group: {condition: _group_summary(sequence_rows, method, group, condition)
                         for condition in CONDITIONS} for group in groups}
        for method in ("rawfeat", "superpoint")
    }
    for method in summaries:
        for group in groups:
            for condition in CONDITIONS:
                summaries[method][group][condition].update(
                    _corner_distribution(all_rows, method, group, condition)
                )
    main_scores = {
        method: float(np.mean([summaries[method]["all"][condition]["h_auc_5"] for condition in NOISY_CONDITIONS]))
        for method in ("rawfeat", "superpoint")
    }
    bootstrap = {group: _bootstrap(sequence_rows, group) for group in groups}
    full_protocol = len(pairs) == 580 and len(images) == 696 and len(all_rows) == 2 * 3480
    checkpoint_sha256 = file_digest(checkpoint_path)
    superpoint_checkpoint_sha256 = file_digest(superpoint_checkpoint_path)
    run_manifest = {
        "protocol": HPATCHES_PROTOCOL, "manifest_sha256": digest,
        "cache_identity_sha256": identity_digest(identity), "selected_pairs": len(pairs),
        "selected_images": len(images), "conditions": list(CONDITIONS),
        "target_indices": selected_target_indices,
        "checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": checkpoint_sha256,
        "superpoint_checkpoint": str(Path(superpoint_checkpoint_path).resolve()),
        "superpoint_checkpoint_sha256": superpoint_checkpoint_sha256,
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "command": command, "full_protocol": full_protocol,
        "single_image_inference": True, "batch_size": 1,
        "cache_generation_included": False, "report_generated": False,
        "visualization_count": visualization_info["count"], "visualization_sequences": list(visual_sequences),
        "visualization_targets": list(VISUALIZATION_TARGETS), "visualizations": visualization_info,
        "total_wall_clock_seconds": time.perf_counter() - started,
    }
    (output_path / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, allow_nan=False) + "\n")
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output_path / "cache_identity.json").write_text(json.dumps(identity, indent=2, allow_nan=False) + "\n")
    _write_jsonl(output_path / "rawfeat" / "per_pair.jsonl", rows_by_method["rawfeat"])
    _write_jsonl(output_path / "superpoint" / "per_pair.jsonl", rows_by_method["superpoint"])
    _write_jsonl(output_path / "per_pair.jsonl", all_rows)
    _write_jsonl(output_path / "image_inference.jsonl", timing_rows)
    _write_jsonl(output_path / "sequence_summary.jsonl", sequence_rows)
    summary = {
        "protocol": HPATCHES_PROTOCOL, "manifest_sha256": digest,
        "cache_identity_sha256": identity_digest(identity), "conditions": list(CONDITIONS),
        "pair_count_per_condition": len(pairs), "pairs": len(pairs), "images": len(images),
        "target_indices": selected_target_indices,
        "full_protocol": full_protocol, "methods": ["rawfeat", "superpoint"],
        "main_h_auc_5": main_scores,
        "main_h_auc_5_delta_percentage_points": 100 * (main_scores["rawfeat"] - main_scores["superpoint"]),
        "summaries": summaries, "sequence_cluster_bootstrap": bootstrap,
        "rawfeat_checkpoint_sha256": checkpoint_sha256,
        "superpoint_checkpoint_sha256": superpoint_checkpoint_sha256,
        "single_image_inference": True, "batch_size": 1,
        "cache_generation_included": False, "visualizations": visualization_info,
        "report_generated": False, "latency_summary": "latency_summary.json",
        "total_wall_clock_seconds": run_manifest["total_wall_clock_seconds"],
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    from .hpatches_report import write_hpatches_report

    report_result = write_hpatches_report(output_path)
    run_manifest.update({
        "report_generated": True,
        "report_paths": {
            "markdown": "report/report.md", "html": "report/report.html", "summary_csv": "report/summary.csv",
            "pairs_csv": "report/pairs.csv", "failures_jsonl": "report/failures.jsonl",
            "latency_summary": "latency_summary.json",
        },
    })
    summary.update({"report_generated": True, "report": report_result})
    (output_path / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, allow_nan=False) + "\n")
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
