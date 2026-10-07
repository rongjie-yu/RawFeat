"""Balanced teacher and spatially uniform homography correspondence sampling."""

from __future__ import annotations

import cv2
import numpy as np


def _warp(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2).astype(np.float32)
    return cv2.perspectiveTransform(points.astype(np.float32)[None], homography.astype(np.float64))[0]


def sample_correspondences(
    teacher_a: np.ndarray,
    teacher_b: np.ndarray,
    homography_a_to_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    rng: np.random.Generator,
    max_points: int = 512,
    teacher_fraction: float = 0.75,
    min_spacing: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return corresponding float pixel positions, without rounding descriptor locations."""
    height, width = valid_a.shape
    inverse = np.linalg.inv(homography_a_to_b)
    quota = round(max_points * teacher_fraction)
    first_quota = quota // 2
    points_a = np.empty((max_points, 2), dtype=np.float32)
    points_b = np.empty((max_points, 2), dtype=np.float32)
    size = 0
    counts = {"teacher_a": 0, "teacher_b": 0, "uniform": 0}

    def accept(candidates_a: np.ndarray, candidates_b: np.ndarray, maximum: int, kind: str) -> None:
        nonlocal size
        for start in range(0, len(candidates_a), 1024):
            if counts[kind] >= maximum or size >= max_points:
                break
            chunk_a, chunk_b = candidates_a[start:start + 1024], candidates_b[start:start + 1024]
            inside = ((chunk_a >= 0) & (chunk_a < [width, height])).all(axis=1)
            inside &= ((chunk_b >= 0) & (chunk_b < [width, height])).all(axis=1)
            chunk_a, chunk_b = chunk_a[inside], chunk_b[inside]
            indices_a = np.clip(np.rint(chunk_a).astype(int), [0, 0], [width - 1, height - 1])
            indices_b = np.clip(np.rint(chunk_b).astype(int), [0, 0], [width - 1, height - 1])
            valid = valid_a[indices_a[:, 1], indices_a[:, 0]] & valid_b[indices_b[:, 1], indices_b[:, 0]]
            for a, b in zip(chunk_a[valid], chunk_b[valid]):
                if counts[kind] >= maximum or size >= max_points:
                    break
                if size:
                    if (np.sum((points_a[:size] - a) ** 2, axis=1) < min_spacing ** 2).any():
                        continue
                    if (np.sum((points_b[:size] - b) ** 2, axis=1) < min_spacing ** 2).any():
                        continue
                points_a[size], points_b[size] = a, b
                size += 1
                counts[kind] += 1

    a_candidates = teacher_a[rng.permutation(len(teacher_a))].astype(np.float32)
    b_candidates = teacher_b[rng.permutation(len(teacher_b))].astype(np.float32)
    accept(a_candidates, _warp(a_candidates, homography_a_to_b), first_quota, "teacher_a")
    accept(_warp(b_candidates, inverse), b_candidates, quota - first_quota, "teacher_b")

    # Uniform spatial sampling fills the explicit 25% share and teacher shortages.
    ys, xs = np.where(valid_a)
    candidates = np.stack((xs, ys), axis=1).astype(np.float32)
    candidates = candidates[rng.permutation(len(candidates))]
    accept(candidates, _warp(candidates, homography_a_to_b), max_points, "uniform")
    if size == 0:
        raise ValueError("no valid descriptor correspondences in this pair")
    return points_a[:size], points_b[:size], counts
