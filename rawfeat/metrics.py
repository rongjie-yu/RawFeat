"""Fixed COCO homography, repeatability and mutual-neighbor metrics."""

from __future__ import annotations

import cv2
import numpy as np


def mutual_nearest(descriptor_a: np.ndarray, descriptor_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(descriptor_a) == 0 or len(descriptor_b) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    similarities = descriptor_a @ descriptor_b.T
    a_to_b = similarities.argmax(axis=1)
    b_to_a = similarities.argmax(axis=0)
    a_index = np.arange(len(a_to_b))
    keep = b_to_a[a_to_b] == a_index
    return a_index[keep], a_to_b[keep]


def warp_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2).astype(np.float32)
    return cv2.perspectiveTransform(points.astype(np.float32)[None], homography.astype(np.float64))[0]


def auc_at(errors: np.ndarray, threshold: float) -> float:
    """Integral of the empirical corner-error recall curve over [0,T], divided by T."""
    return float(np.maximum(0, 1 - errors / threshold).mean())


def _visible(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    x, y = points[:, 0], points[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    result = np.zeros(len(points), dtype=bool)
    clipped_x = np.clip(np.rint(x).astype(int), 0, width - 1)
    clipped_y = np.clip(np.rint(y).astype(int), 0, height - 1)
    result[inside] = mask[clipped_y[inside], clipped_x[inside]]
    return result


def pair_metrics(
    feature_a: dict[str, np.ndarray],
    feature_b: dict[str, np.ndarray],
    homography: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    *,
    reprojection_threshold: float = 3.0,
    ransac_seed: int = 2027,
) -> dict[str, float | int]:
    points_a, points_b = feature_a["points"], feature_b["points"]
    descriptor_a, descriptor_b = feature_a["descriptors"], feature_b["descriptors"]
    index_a, index_b = mutual_nearest(descriptor_a, descriptor_b)
    matched_a, matched_b = points_a[index_a], points_b[index_b]
    error = np.linalg.norm(warp_points(matched_a, homography) - matched_b, axis=1)
    true_matches = int((error <= reprojection_threshold).sum())
    precision = float(true_matches / len(error)) if len(error) else 0.0

    estimated = None
    if len(matched_a) >= 4:
        cv2.setRNGSeed(ransac_seed)
        estimated, _ = cv2.findHomography(
            matched_a, matched_b, cv2.RANSAC, ransacReprojThreshold=reprojection_threshold,
            maxIters=10000, confidence=0.999,
        )
    height, width = valid_a.shape
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    corner_error = float("inf") if estimated is None else float(np.linalg.norm(
        warp_points(corners, estimated) - warp_points(corners, homography), axis=1).mean())

    a_warp = warp_points(points_a, homography)
    b_warp = warp_points(points_b, np.linalg.inv(homography))
    visible_a = _visible(a_warp, valid_b) & _visible(points_a, valid_a)
    visible_b = _visible(b_warp, valid_a) & _visible(points_b, valid_b)
    nearest_a = np.linalg.norm(a_warp[visible_a, None] - points_b[None], axis=2).min(axis=1) if len(points_b) and visible_a.any() else np.empty(0)
    nearest_b = np.linalg.norm(b_warp[visible_b, None] - points_a[None], axis=2).min(axis=1) if len(points_a) and visible_b.any() else np.empty(0)
    repeat_a = float((nearest_a <= reprojection_threshold).mean()) if len(nearest_a) else 0.0
    repeat_b = float((nearest_b <= reprojection_threshold).mean()) if len(nearest_b) else 0.0
    repeatability = (repeat_a + repeat_b) / 2
    localization = np.concatenate((nearest_a[nearest_a <= reprojection_threshold], nearest_b[nearest_b <= reprojection_threshold]))
    return {"corner_error": corner_error, "h_auc_1": max(0.0, 1 - corner_error / 1),
            "h_auc_3": max(0.0, 1 - corner_error / 3), "h_auc_5": max(0.0, 1 - corner_error / 5),
            "repeatability": repeatability, "localization_error": float(localization.mean()) if len(localization) else float("nan"),
            "match_precision": precision, "correct_matches": true_matches,
            "detected_a": len(points_a), "detected_b": len(points_b), "matches": len(error)}
