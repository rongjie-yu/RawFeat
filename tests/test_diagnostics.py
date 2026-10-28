import copy

import pytest
import torch
from rawfeat.diagnostics import detector_view, recall_metrics, gray_metrics, gradient_probe


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


@pytest.fixture
def probe_inputs(monkeypatch):
    import rawfeat.diagnostics as diagnostics
    import rawfeat.training as training

    class ProbeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gray = torch.nn.Linear(1, 1, bias=False)
            self.register_buffer('calls', torch.tensor(0))

        def forward(self, packed, compute_descriptors=True):
            self.calls.add_(1)
            return self.gray(packed)

    def losses(out, *args):
        det, gray, match = out.mean(), out.square().mean(), (out-1).square().mean()
        return dict(det=det, gray=gray, match=match, total=det+10*gray+match)

    monkeypatch.setattr(training, 'set_training_mode', lambda model, config, step: model.train())
    monkeypatch.setattr(diagnostics, 'pair_losses', losses)
    monkeypatch.setattr(diagnostics, 'update_schedule', lambda step, config: {'matching_weight': 1.})
    model = ProbeModel()
    optimizer = torch.optim.AdamW(model.parameters())
    for p in model.parameters():
        optimizer.state[p]['step'] = torch.tensor(7.)
        p.grad = torch.ones_like(p)
    batches = [dict(packed=torch.tensor([[.2+i*.1]]), teacher_logits=None, valid_cells=None,
                    clean_gray=None, valid_pixels=None, matches=None) for i in range(4)]
    return model, batches, {'schedule': {}}, optimizer


def test_finite_probe_mismatch_warns_and_preserves_live_training_state(monkeypatch, probe_inputs):
    model, batches, config, optimizer = probe_inputs
    model_before = copy.deepcopy(model.state_dict())
    adam_before = copy.deepcopy(optimizer.state_dict())
    gradients_before = [p.grad.clone() for p in model.parameters()]
    rng_before = torch.get_rng_state()
    original_grad = torch.autograd.grad
    calls = 0

    def perturbed_grad(*args, **kwargs):
        nonlocal calls
        calls += 1
        values = original_grad(*args, **kwargs)
        return tuple(v*1.0001 if v is not None and calls % 4 == 0 else v for v in values)

    monkeypatch.setattr(torch.autograd, 'grad', perturbed_grad)
    with pytest.warns(RuntimeWarning, match='finite gradients recorded, training continues'):
        result = gradient_probe(model, batches, config, 999, optimizer)
    assert result['gradient_sum_relative_l2'] > result['gradient_sum_tolerance']
    assert result['gradient_sum_absolute_l2'] > 0 and not result['gradient_sum_within_tolerance']
    assert result['optimizer_updates'] == 0 and set(result['adam_steps'].values()) == {7}
    assert all(torch.equal(v, model.state_dict()[name]) for name, v in model_before.items())
    assert optimizer.state_dict() == adam_before
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), gradients_before))
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_zero_probe_gradients_have_zero_discrepancy(monkeypatch, probe_inputs):
    import rawfeat.diagnostics as diagnostics
    monkeypatch.setattr(diagnostics, 'pair_losses',
                        lambda out, *args: {name: out.sum()*0 for name in ('det', 'gray', 'match', 'total')})
    model, batches, config, optimizer = probe_inputs
    result = gradient_probe(model, batches, config, 999, optimizer)
    assert result['gradient_sum_relative_l2'] == result['gradient_sum_absolute_l2'] == 0
    assert result['gradient_sum_within_tolerance'] and result['global_norm'] == 0


@pytest.mark.parametrize('task_index', range(4))
def test_probe_rejects_nonfinite_gradients_in_every_task(monkeypatch, probe_inputs, task_index):
    original_grad = torch.autograd.grad
    calls = 0

    def nonfinite_grad(*args, **kwargs):
        nonlocal calls
        index = calls % 4
        calls += 1
        values = original_grad(*args, **kwargs)
        return tuple(v*float('nan') if v is not None and index == task_index else v for v in values)

    monkeypatch.setattr(torch.autograd, 'grad', nonfinite_grad)
    model, batches, config, optimizer = probe_inputs
    with pytest.raises(FloatingPointError, match='nonfinite task gradient'):
        gradient_probe(model, batches, config, 999, optimizer)
