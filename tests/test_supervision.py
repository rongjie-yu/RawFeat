import numpy as np
import torch

from rawfeat.correspondences import sample_correspondences
from rawfeat.losses import detection_loss, gray_loss, matching_loss


def test_losses_and_gradients():
    student = torch.randn(2, 65, 2, 3, requires_grad=True)
    teacher = torch.randn_like(student)
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    valid[0, 0, 0] = False
    detection = detection_loss(student, teacher, valid)
    assert detection > 0
    clean = torch.full((2, 1, 16, 24), 0.2)
    gray = torch.zeros_like(clean, requires_grad=True)
    grayscale = gray_loss(gray, clean, torch.ones_like(clean))
    torch.testing.assert_close(grayscale, torch.tensor(0.04))
    desc_a = torch.randn(128, 2, 3, requires_grad=True)
    desc_b = torch.randn(128, 2, 3, requires_grad=True)
    points = torch.tensor([[4.0, 3.0], [12.0, 11.0], [20.0, 3.0]])
    matching = matching_loss(desc_a, desc_b, points, points, 16, 24)
    (detection + grayscale + matching).backward()
    assert student.grad.abs().sum() > 0
    assert gray.grad.abs().sum() > 0
    assert desc_a.grad.abs().sum() > 0


def test_correspondence_geometry_and_spacing():
    valid = np.ones((64, 80), dtype=bool)
    homography = np.array([[1, 0, 3.25], [0, 1, -2.5], [0, 0, 1]], dtype=np.float64)
    a, b, counts = sample_correspondences(
        np.array([[16, 16], [32, 32]], np.float32), np.array([[24, 24]], np.float32),
        homography, valid, valid, np.random.default_rng(42), max_points=16,
    )
    np.testing.assert_allclose(b - a, np.tile([3.25, -2.5], (len(a), 1)), atol=1e-5)
    assert counts["teacher_a"] > 0 and counts["teacher_b"] > 0
    for points in (a, b):
        distance = np.sqrt(((points[:, None] - points[None]) ** 2).sum(axis=2))
        np.fill_diagonal(distance, np.inf)
        assert distance.min() >= 8 - 1e-5


def test_gray_per_view_normalization_unequal_masks_and_targets():
    from rawfeat.losses import gray_loss
    p = torch.tensor([[[[.3, .7]]], [[[.4, .1]]]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[[[.1, .5]]], [[[.2, .8]]]], dtype=torch.float64)
    m = torch.tensor([[[[True, False]]], [[[True, True]]]])
    per_view = []
    for predicted, target, valid in zip(p, y, m):
        weight = (target+.01).pow(-2)*valid
        per_view.append((weight*(predicted-target).square()).sum()/weight.sum())
    loss = gray_loss(p, y, m)
    torch.testing.assert_close(loss, torch.stack(per_view).mean())
    loss.backward(); assert p.grad[0, 0, 0, 1] == 0
