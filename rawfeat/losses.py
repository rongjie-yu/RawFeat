"""The three losses specified for a pair of noisy Bayer views."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .features import sample_descriptors


def detection_terms(student: torch.Tensor, teacher: torch.Tensor, valid_cells: torch.Tensor) -> dict[str, torch.Tensor]:
    """Bernoulli occupancy and mass-normalized conditional position, per view."""
    ls, lt = F.log_softmax(student, dim=1), F.log_softmax(teacher.detach(), dim=1)
    lms, lmt = torch.logsumexp(ls[:, :64], dim=1), torch.logsumexp(lt[:, :64], dim=1)
    mt, td = lmt.exp(), lt[:, 64].exp()
    occupancy = mt * (lmt - lms) + td * (lt[:, 64] - ls[:, 64])
    position = (lt[:, :64].exp() * ((lt[:, :64] - lmt[:, None]) - (ls[:, :64] - lms[:, None]))).sum(dim=1)
    valid = valid_cells.to(student.dtype)
    count = valid.sum(dim=(-2, -1))
    if not torch.all(count > 0):
        raise ValueError("each view must contain valid detector cells")
    mass = (valid * mt).sum(dim=(-2, -1))
    position_sum = (valid * position).sum(dim=(-2, -1))
    return {"occupancy": (valid * occupancy).sum(dim=(-2, -1)) / count,
            "position_mass": position_sum / mass.clamp_min(1),
            "teacher_mass": mass, "valid_count": count, "position_scale": count / mass.clamp_min(1)}


def detection_loss(student: torch.Tensor, teacher: torch.Tensor, valid_cells: torch.Tensor) -> torch.Tensor:
    """The sole detector objective, with conditional position coefficient one."""
    terms = detection_terms(student, teacher, valid_cells)
    return (terms["occupancy"] + terms["position_mass"]).reshape(-1, 2).mean(dim=1).mean()


def gray_loss(predicted: torch.Tensor, target: torch.Tensor, valid_pixels: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    weights = valid_pixels.to(target.dtype) / (target + epsilon).square()
    if not torch.all(weights.sum(dim=(-2, -1, -3)) > 0):
        raise ValueError("each view must contain valid grayscale pixels")
    per_view = (weights * (predicted - target).square()).sum(dim=(-2, -1, -3)) / weights.sum(dim=(-2, -1, -3))
    return per_view.reshape(-1, 2).mean(dim=1).mean()


def matching_loss(
    descriptor_a: torch.Tensor,
    descriptor_b: torch.Tensor,
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    height: int,
    width: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    if len(points_a) != len(points_b) or len(points_a) == 0:
        raise ValueError("a pair requires at least one valid correspondence")
    a = sample_descriptors(descriptor_a, points_a, height, width)
    b = sample_descriptors(descriptor_b, points_b, height, width)
    similarities = a @ b.T / temperature
    row = F.log_softmax(similarities, dim=1).diagonal()
    column = F.log_softmax(similarities, dim=0).diagonal()
    return -(row + column).mean() / 2


def pair_losses(
    student: dict[str, torch.Tensor], teacher_logits: torch.Tensor, valid_cells: torch.Tensor,
    clean_gray: torch.Tensor, valid_pixels: torch.Tensor,
    correspondences: list[tuple[torch.Tensor, torch.Tensor]] | None, matching_weight: float,
) -> dict[str, torch.Tensor | None]:
    terms = detection_terms(student['logits'], teacher_logits, valid_cells)
    occupancy, position = terms['occupancy'].mean(), terms['position_mass'].mean()
    detector = occupancy + position
    grayscale = gray_loss(student['gray'], clean_gray, valid_pixels)
    matching = None
    if matching_weight > 0:
        if correspondences is None or student['logits'].shape[0] != 2 * len(correspondences):
            raise ValueError('student views and correspondence pairs disagree')
        height, width = clean_gray.shape[-2:]
        matching = torch.stack([
            matching_loss(student['descriptors'][2*i], student['descriptors'][2*i+1], a, b, height, width)
            for i, (a, b) in enumerate(correspondences)
        ]).mean()
    total = detector + 10 * grayscale
    if matching is not None:
        total = total + matching_weight * matching
    return {'det': detector, 'occupancy': occupancy, 'position': position,
            'gray': grayscale, 'match': matching, 'total': total,
            'teacher_mass': terms['teacher_mass'].mean(), 'position_scale': terms['position_scale'].mean()}
