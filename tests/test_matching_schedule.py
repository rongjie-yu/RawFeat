"""Synthetic mathematical/state checks; no online training or GPU optimizer updates."""
import copy
from pathlib import Path

import pytest
import torch

from rawfeat.config import load_config
from rawfeat.losses import pair_losses
from rawfeat.model import RawFeatureExtractor
from rawfeat.schedules import learning_rate, matching_coefficient, phase, ratio_limit
from rawfeat.training import (checkpoint_payload, load_checkpoint, make_optimizer, save_checkpoint,
                              set_training_mode, capture_rng, restore_rng, training_request)

CONFIG = load_config('configs/smoke.yaml')


@pytest.mark.parametrize('name,bounds', [('formal', (50000, 55000, 100000, 2000, 5000, 40000)),
                                       ('smoke', (300, 400, 600, 20, 30, 240))])
def test_absolute_next_update_boundaries(name, bounds):
    c = load_config(f'configs/{name}.yaml')['schedule']; a, b, total, warm, noise_start, noise_end = bounds
    assert phase(a-1, c) == 'A' and matching_coefficient(a-1, c) == 0
    assert phase(a, c) == 'B' and matching_coefficient(a, c) == pytest.approx(1/(b-a))
    assert phase(b-1, c) == 'B' and matching_coefficient(b-1, c) == 1
    assert phase(b, c) == 'C' and matching_coefficient(total-1, c) == 1
    assert learning_rate(0, c) == 3e-6
    assert learning_rate(warm-1, c) == pytest.approx(3e-4)
    assert learning_rate(a-1, c) == 3e-4
    assert learning_rate(a, c) < 3e-4
    assert learning_rate(total-1, c) == 3e-6
    assert ratio_limit(noise_start-1, c) == 1
    assert ratio_limit(noise_end-1, c) == pytest.approx(100)
    assert ratio_limit(a-1, c) == pytest.approx(100)


def synthetic_batch():
    return torch.randn(8, 4, 16, 16), torch.randn(8, 65, 4, 4), torch.ones(8, 4, 4, dtype=torch.bool), torch.rand(8, 1, 32, 32), torch.ones(8, 1, 32, 32, dtype=torch.bool)


def test_pause_bn_adam_decay_and_gradient_routes(monkeypatch):
    import rawfeat.losses as loss_module
    torch.set_num_threads(2)
    model = RawFeatureExtractor(); opt = make_optimizer(model, CONFIG)
    assert sum(len(g['params']) for g in opt.param_groups) == len(list(model.parameters()))
    before = copy.deepcopy(model.descriptor.state_dict())
    set_training_mode(model, CONFIG, 299)
    def forbidden(*args, **kwargs):
        raise AssertionError('paused descriptor or matching executed')
    monkeypatch.setattr(model.descriptor, 'forward', forbidden)
    monkeypatch.setattr(loss_module, 'matching_loss', forbidden)
    x, teacher, cells, gray, pixels = synthetic_batch()
    losses = pair_losses(model(x, compute_descriptors=False), teacher, cells, gray, pixels, None, 0.)
    assert losses['match'] is None
    for task in ('det', 'gray'):
        for index, stage in enumerate((model.stage1, model.stage2, model.stage3)):
            gs = torch.autograd.grad(losses[task], list(stage.parameters()), retain_graph=True, allow_unused=True)
            norm = sum(float(g.norm()) for g in gs if g is not None)
            assert norm == 0 if task == 'gray' and index == 2 else norm > 0
    losses['total'].backward(); opt.step()
    assert all(p.grad is None and not opt.state[p] for p in model.descriptor.parameters())
    assert all(torch.equal(v, model.descriptor.state_dict()[k]) for k, v in before.items())
    assert all(int(opt.state[p]['step']) == 1 for n, p in model.named_parameters() if not n.startswith('descriptor.'))
    assert model.stage1[0].conv3[1].num_batches_tracked == 1
    set_training_mode(model, CONFIG, 300)
    assert model.descriptor.training and all(p.requires_grad for p in model.descriptor.parameters())


@pytest.mark.parametrize('step', [0, 299, 300, 301, 399, 400, 600])
def test_complete_restore_identity_rng_cursor_and_parameter_steps(tmp_path, step):
    model = RawFeatureExtractor(); opt = make_optimizer(model, CONFIG)
    identity = {'training': 'synthetic test', 'config': CONFIG}
    for name, p in model.named_parameters():
        count = max(0, step-300) if name.startswith('descriptor.') else step
        if count:
            opt.state[p] = {'step': torch.tensor(float(count)), 'exp_avg': torch.ones_like(p)*.01, 'exp_avg_sq': torch.ones_like(p)*.001}
    state = checkpoint_payload(model, opt, CONFIG, step, identity, {'det': None, 'geometry': None}, None, 7)
    save_checkpoint(tmp_path/'state.pt', state)
    # A reliable snapshot cannot share mutable live BN/optimizer tensors.
    with torch.no_grad(): model.stage1[0].conv3[1].running_mean.add_(1.)
    resumed = RawFeatureExtractor(); new_opt = make_optimizer(resumed, CONFIG)
    loaded = load_checkpoint(tmp_path/'state.pt', resumed, new_opt, CONFIG, identity)
    assert loaded['step'] == step and loaded['sample_cursor'] == step*16
    assert torch.equal(torch.get_rng_state(), state['rng']['torch_cpu'])
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in state['student'].items())
    set_training_mode(resumed, CONFIG, step)
    assert resumed.descriptor.training == (step >= 300)
    state['scheduler']['matching_weight'] = -1
    save_checkpoint(tmp_path/'bad.pt', state)
    with pytest.raises(RuntimeError, match='cursor or next schedule'):
        load_checkpoint(tmp_path/'bad.pt', resumed, new_opt, CONFIG, identity)
    with pytest.raises(RuntimeError, match='identity differs'):
        load_checkpoint(tmp_path/'state.pt', resumed, new_opt, CONFIG, {'training': 'changed'})


def test_stateless_sampling_and_rng_replay():
    paths = [Path('0001.jpg'), Path('0002.jpg')]
    request = training_request(paths, 42, 4800, 300, CONFIG)
    state = capture_rng(); a = torch.rand(8, device='cuda:0'); restore_rng(state); b = torch.rand(8, device='cuda:0')
    assert torch.equal(a, b)
    assert training_request(paths, 42, 4800, 300, CONFIG) == request
    paused = training_request(paths, 42, 4800, 299, CONFIG)
    assert paused == request  # noise distribution already fixed before matching entry


def test_restore_rejects_wrong_per_parameter_adam_count(tmp_path):
    model = RawFeatureExtractor(); opt = make_optimizer(model, CONFIG)
    for name, p in model.named_parameters():
        if not name.startswith('descriptor.'):
            opt.state[p] = {'step': torch.tensor(300.), 'exp_avg': torch.zeros_like(p), 'exp_avg_sq': torch.zeros_like(p)}
    identity = {'training': 'synthetic'}
    state = checkpoint_payload(model, opt, CONFIG, 300, identity, {'det': None, 'geometry': None}, None, 0)
    # All groups must remain present but A must never have advanced descriptor Adam.
    target = next(model.descriptor.parameters())
    opt.state[target] = {'step': torch.tensor(1.), 'exp_avg': torch.zeros_like(target), 'exp_avg_sq': torch.zeros_like(target)}
    state['optimizer'] = copy.deepcopy(opt.state_dict())
    save_checkpoint(tmp_path/'bad.pt', state)
    resumed = RawFeatureExtractor(); optimizer = make_optimizer(resumed, CONFIG)
    with pytest.raises(RuntimeError, match='per-parameter Adam step'):
        load_checkpoint(tmp_path/'bad.pt', resumed, optimizer, CONFIG, identity)
