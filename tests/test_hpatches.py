import numpy as np

from rawfeat.hpatches import (
    BOOTSTRAP_ITERATIONS,
    CONDITIONS,
    HPATCHES_HEIGHT,
    HPATCHES_WIDTH,
    NOISY_RATIOS,
    _bootstrap,
    _validate_timing_rows,
    _write_visualization_indexes,
    cache_path,
    condition_specs,
    identity_digest,
    iter_manifest_pairs,
    read_manifest,
    resize_transform,
    stable_seed,
    transform_homography,
    evaluate_hpatches,
)


def test_hpatches_conditions_are_symmetric_and_complete():
    specs = condition_specs()
    assert [spec["name"] for spec in specs] == list(CONDITIONS)
    assert specs[0]["clean"] and specs[0]["ratio"] is None
    for spec, ratio in zip(specs[1:], NOISY_RATIOS):
        assert spec["ratio_a"] == ratio == spec["ratio_b"]
        assert not spec["clean"]


def test_stable_seed_is_reproducible_and_ratio_specific():
    first = stable_seed(2027, 3, 2, 16)
    assert first == stable_seed(2027, 3, 2, 16)
    assert first != stable_seed(2027, 3, 2, 64)
    assert first != stable_seed(2027, 4, 2, 16)
    assert 0 <= first < 2**32


def test_resize_transform_and_homography_identity_contract():
    transform = resize_transform(800, 600)
    assert transform["output_width"] == HPATCHES_WIDTH
    assert transform["output_height"] == HPATCHES_HEIGHT
    identity = np.eye(3)
    np.testing.assert_allclose(transform_homography(transform, transform, identity), identity)
    point = np.array([[[123.25, 88.5]]], dtype=np.float64)
    transformed = np.asarray(transform["T"]) @ np.array([point[0, 0, 0], point[0, 0, 1], 1.0])
    transformed /= transformed[2]
    recovered = np.linalg.inv(np.asarray(transform["T"])) @ np.r_[transformed[:2], 1.0]
    recovered /= recovered[2]
    np.testing.assert_allclose(recovered[:2], point[0, 0], atol=1e-12)


def test_resize_homography_preserves_projected_points():
    transform_a = resize_transform(1280, 870)
    transform_b = resize_transform(845, 1126)
    homography = np.array([[1.02, 0.03, 12.0], [-0.01, 0.98, 7.0], [0.0001, -0.0002, 1.0]])
    processed = transform_homography(transform_a, transform_b, homography)
    source = np.array([[200.0, 100.0], [700.0, 400.0], [1100.0, 700.0]], dtype=np.float64)
    source_h = np.c_[source, np.ones(len(source))]
    raw_target = (homography @ source_h.T).T
    raw_target /= raw_target[:, 2:]
    target_from_transforms = (np.asarray(transform_b["T"]) @ np.c_[raw_target[:, :2], np.ones(len(source))].T).T
    target_from_transforms /= target_from_transforms[:, 2:]
    source_processed = (np.asarray(transform_a["T"]) @ source_h.T).T
    source_processed /= source_processed[:, 2:]
    target_from_h = (processed @ source_processed.T).T
    target_from_h /= target_from_h[:, 2:]
    np.testing.assert_allclose(target_from_h[:, :2], target_from_transforms[:, :2], atol=1e-10)


def test_frozen_manifest_counts_and_identity():
    manifest, digest = read_manifest("manifests/hpatches_synthetic_raw_v1.json")
    assert len(digest) == 64
    assert identity_digest({"b": 1, "a": 2}) == identity_digest({"a": 2, "b": 1})
    assert manifest["condition_pair_count"] == 3480
    assert all(pair["ratio_a"] == pair["ratio_b"] for pair in manifest["conditions"][1:])


def test_cache_path_keeps_image_and_condition_identity(tmp_path):
    path = cache_path(tmp_path, "i_scene/3", "ratio100")
    assert path == tmp_path / "images" / "i_scene" / "3_ratio100.pt"


def test_sequence_bootstrap_uses_five_noisy_conditions():
    rows = []
    for method, value in (("rawfeat", 0.7), ("superpoint", 0.5)):
        for condition in CONDITIONS:
            if condition == "clean":
                continue
            rows.append({"method": method, "sequence": "i_scene", "type": "illumination",
                         "condition": condition, "h_auc_5": value})
    sequence_rows = rows
    result = _bootstrap(sequence_rows, "all")
    assert result["iterations"] == BOOTSTRAP_ITERATIONS
    assert result["sequence_count"] == 1
    np.testing.assert_allclose(result["estimate"], 0.2)
    assert result["positive_sequences"] == 1


def test_target_filter_keeps_only_fixed_smoke_targets():
    manifest, _ = read_manifest("manifests/hpatches_synthetic_raw_v1.json")
    pairs = iter_manifest_pairs(
        manifest,
        ["i_ajuntament", "i_autannes", "v_abstract", "v_adam"],
        target_indices=(2, 3),
    )
    assert len(pairs) == 8
    assert {pair["target_index"] for pair in pairs} == {2, 3}


def test_evaluation_rejects_non_single_image_batch_before_work(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "batch_size=1"):
        evaluate_hpatches(
            "unused-manifest.json", tmp_path, tmp_path,
            model=None, superpoint=None, checkpoint_path="checkpoint.pt",
            superpoint_checkpoint_path="superpoint.pt", batch_images=2,
        )


def test_timing_rows_are_unique_single_image_records():
    row = {
        "method": "rawfeat", "image_id": "i_scene/1", "condition": "clean",
        "batch_size": 1, "single_image_inference": True,
        "network_forward_ms": 1.0, "feature_extraction_ms": 2.0,
        "total_inference_ms": 3.0, "detected_points": 10, "peak_memory_bytes": 20,
    }
    _validate_timing_rows([row], 1)
    duplicate = dict(row)
    with np.testing.assert_raises_regex(RuntimeError, "duplicate"):
        _validate_timing_rows([row, duplicate], 2)
    non_single = dict(row, batch_size=4)
    with np.testing.assert_raises_regex(RuntimeError, "single-image"):
        _validate_timing_rows([non_single], 1)


def test_visualization_indexes_cover_both_methods_and_all_conditions(tmp_path):
    names = [
        f"{method}_i_ajuntament_{target}_{condition}.png"
        for method in ("rawfeat", "superpoint")
        for target in (2, 3)
        for condition in CONDITIONS
    ]
    info = _write_visualization_indexes(tmp_path, names)
    assert info["count"] == 24
    assert all((tmp_path / page).is_file() for page in info["condition_pages"])
    assert (tmp_path / info["index"]).is_file()
