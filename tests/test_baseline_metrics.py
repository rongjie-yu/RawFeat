import numpy as np

from rawfeat.baseline import baseline_gray, linear_raw_uint16, meanad
from rawfeat.metrics import auc_at, mutual_nearest, pair_metrics
from rawfeat.sensor import BLACK_LEVEL


def test_meanad_shared_statistics_and_flat_image():
    rgb = np.array([[[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]]], np.float32)
    out = meanad(rgb)
    mean = rgb.mean(dtype=np.float64)
    deviation = np.abs(rgb.astype(np.float64) - mean).mean()
    np.testing.assert_allclose(out, np.clip((rgb - mean + 2 * deviation) / (4 * deviation + 1e-8), 0, 1))
    assert not meanad(np.ones((2, 2, 3), np.float32)).any()


def test_baseline_level_quantization():
    bayer = np.full((16, 16), BLACK_LEVEL + 500, np.float32)
    encoded = linear_raw_uint16(bayer)
    assert encoded.dtype == np.uint16
    expected = np.uint16((500 / 16383) * 65535)
    np.testing.assert_array_equal(encoded[4:-4, 4:-4], np.full((8, 8, 3), expected, np.uint16))
    assert baseline_gray(bayer).shape == (16, 16)


def test_homography_metrics_exact_matches():
    pts = np.array([[10, 10], [20, 10], [20, 20], [10, 20], [30, 30]], np.float32)
    desc = np.eye(5, dtype=np.float32)
    h = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]], np.float64)
    a = {"points": pts, "descriptors": desc}
    b = {"points": pts + [2, 3], "descriptors": desc}
    valid = np.ones((48, 48), dtype=bool)
    ia, ib = mutual_nearest(desc, desc)
    assert np.array_equal(ia, ib)
    metrics = pair_metrics(a, b, h, valid, valid)
    assert metrics["correct_matches"] == 5
    assert metrics["h_auc_5"] > 0.999
    assert metrics["match_precision"] == 1
    assert auc_at(np.array([0, 5, np.inf]), 5) == 1 / 3
