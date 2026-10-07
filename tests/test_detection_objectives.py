import pytest
import torch
from torch.nn import functional as F

from rawfeat.losses import detection_loss, detection_terms


def test_mass_matches_independent_per_view_mass_formula():
    torch.manual_seed(18)
    student = torch.randn(4, 65, 2, 3, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    # Include a view with total foreground mass below one and unequal masks.
    with torch.no_grad():
        teacher[0, 64] += 15
    valid = torch.ones(4, 2, 3, dtype=torch.bool)
    valid[1, 0] = False
    pt, ps = teacher.detach().softmax(1), student.softmax(1)
    mt, ms = pt[:, :64].sum(1), ps[:, :64].sum(1)
    qt, qs = pt[:, :64] / mt[:, None], ps[:, :64] / ms[:, None]
    occupancy = mt * (mt.log() - ms.log()) + pt[:, 64] * (pt[:, 64].log() - ps[:, 64].log())
    position = mt * (qt * (qt.log() - qs.log())).sum(1)
    reference = ((occupancy * valid).sum((1, 2)) / valid.sum((1, 2))
                 + (position * valid).sum((1, 2)) / (mt * valid).sum((1, 2)).clamp_min(1)).mean()
    candidate = detection_loss(student, teacher, valid)
    torch.testing.assert_close(candidate, reference, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(candidate, student, retain_graph=True)[0]
    b = torch.autograd.grad(reference, student, retain_graph=True)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert torch.autograd.grad(candidate, teacher, allow_unused=True)[0] is None


def test_mass_extreme_logits_are_finite_and_reject_empty_view():
    student = torch.full((2, 65, 2, 2), -1000., requires_grad=True)
    teacher = torch.full_like(student, 1000.)
    with torch.no_grad():
        student[:, 64] = 1000.
        teacher[:, 64] = -1000.
    valid = torch.ones(2, 2, 2, dtype=torch.bool)
    loss = detection_loss(student, teacher, valid)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    valid[1] = False
    with pytest.raises(ValueError, match="each view"):
        detection_loss(student, teacher, valid)
