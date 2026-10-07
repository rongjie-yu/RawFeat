import pytest
import json

from rawfeat.validation import baseline_comparison, baseline_subset


def test_baseline_comparison_reports_every_bucket_and_metric():
    baseline = {bucket: {
        "h_auc_1": 0.1, "h_auc_3": 0.2, "h_auc_5": 0.3,
        "repeatability": 0.4, "match_precision": 0.5,
        "correct_matches": 10, "detected_a": 50, "detected_b": 60,
        "matches": 20, "homography_failures": 2,
        "corner_error": 3.0, "localization_error": 1.0,
    } for bucket in ("clean", "1", "4", "16", "64", "100")}
    student = {bucket: {**values, "h_auc_5": 0.35, "correct_matches": 12,
                        "localization_error": None if bucket == "100" else 0.8}
               for bucket, values in baseline.items()}
    differences = baseline_comparison(student, baseline)
    assert set(differences) == set(baseline)
    assert differences["4"]["h_auc_5_percentage_points"] == pytest.approx(5.0)
    assert differences["4"]["correct_matches"] == 2
    assert differences["4"]["localization_error"] == pytest.approx(-0.2)
    assert differences["100"]["localization_error"] is None


def test_partial_baseline_uses_same_images_and_keeps_failed_cases(tmp_path):
    rows = []
    for position, score in enumerate((0.2, 0.8)):
        for bucket in ("clean", "1", "4", "16", "64", "100"):
            rows.append({"image_position": position, "bucket": bucket,
                         "h_auc_1": score, "h_auc_3": score, "h_auc_5": score,
                         "repeatability": 0.5, "match_precision": 0.5,
                         "corner_error": None, "localization_error": None,
                         "correct_matches": 0, "detected_a": 10, "detected_b": 10, "matches": 0})
    (tmp_path / "per_pair.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    subset = baseline_subset(tmp_path / "summary.json", 1)
    assert subset["1"]["h_auc_5"] == 0.2
    assert subset["1"]["cases"] == 1
    assert subset["1"]["homography_failures"] == 1
    assert subset["1"]["corner_error"] is None
    assert baseline_subset(tmp_path / "summary.json", 2)["1"]["h_auc_5"] == pytest.approx(0.5)
