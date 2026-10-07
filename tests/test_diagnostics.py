import torch
from rawfeat.diagnostics import detector_view, recall_metrics, gray_metrics


def test_empty_diagnostic_sets_are_null_with_counts():
    values = recall_metrics(torch.empty(0, 2), torch.empty(0, 2))
    assert values['recall_1px'] is None and values['reference_count'] == 0
    values = recall_metrics(torch.ones(2, 2), torch.empty(0, 2))
    assert values['recall_1px'] == 0 and values['localization_3px'] is None
    logits = torch.zeros(65, 2, 3)
    values = detector_view(logits, logits, torch.ones(2, 3, dtype=torch.bool), torch.empty(0, 2))
    assert values['phase_top1'] is None and values['teacher_nms_points'] == 0
    assert values['foreground_mae_background'] is None


def test_gray_edges_mask_and_error():
    gray = torch.zeros(1, 8, 8); gray[:, :, 4:] = .2
    valid = torch.ones_like(gray, dtype=torch.bool)
    metrics = gray_metrics(gray, gray, valid, torch.zeros(4, 4, 4))
    assert metrics['gray_weighted_mse'] == 0 and metrics['edge_gradient_mse'] == 0
    assert metrics['edge_gradient_cosine'] > .999 and metrics['edge_count'] == 8
