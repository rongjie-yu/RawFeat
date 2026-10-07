"""GPU mapping migration must preserve every other run and recovery contract."""
import copy
import json

import pytest
import torch

from rawfeat.cli import _parser
from rawfeat.identity import check_resume_identity
from rawfeat.model import RawFeatureExtractor
from rawfeat.training import checkpoint_payload, load_checkpoint, load_config, make_optimizer, save_checkpoint, train


@pytest.fixture
def identities(tmp_path, monkeypatch):
    import rawfeat.identity as identity_module
    saved = {'cuda_visible_devices': '1', 'gpu': 'NVIDIA GeForce RTX 5090',
             'config': {'output': '/run', 'schedule': {'updates': 100000}},
             'code_sha256': {'rawfeat/identity.py': 'old', 'rawfeat/losses.py': 'same'},
             'dependencies': {'torch': 'same'}, 'assets_sha256': {'teacher': 'same'},
             'train_paths_sha256': 'same', 'precision': 'FP32', 'tf32_matmul': False,
             'fixed_protocol': {'manifest_sha256': 'same'}}
    current = copy.deepcopy(saved)
    current['cuda_visible_devices'] = '3'
    current['code_sha256']['rawfeat/identity.py'] = 'new'
    revision = tmp_path/'revision.json'
    revision.write_text(json.dumps({'source_code_sha256': saved['code_sha256'],
                                    'target_code_sha256': current['code_sha256']}))
    monkeypatch.setattr(identity_module, 'GPU_RESUME_REVISION', revision)
    return saved, current


def test_same_code_gpu_move_requires_explicit_flag(identities):
    saved, current = identities
    current['code_sha256'] = saved['code_sha256'].copy()
    with pytest.raises(RuntimeError, match='cuda_visible_devices'):
        check_resume_identity(saved, current)
    assert check_resume_identity(saved, current, allow_gpu_change=True) == ['cuda_visible_devices']


def test_exact_source_revision_and_gpu_move_are_audited(identities):
    saved, current = identities
    original = copy.deepcopy(saved)
    with pytest.raises(RuntimeError, match='code_sha256'):
        check_resume_identity(saved, current)
    assert check_resume_identity(saved, current, allow_gpu_change=True) == [
        'code_sha256.rawfeat/identity.py', 'cuda_visible_devices']
    assert check_resume_identity(current, current) == []
    assert saved == original
    with pytest.raises(RuntimeError, match='code_sha256'):
        check_resume_identity(current, saved, allow_gpu_change=True)


@pytest.mark.parametrize('path,value', [
    ('gpu', 'different GPU'), ('dependencies.torch', 'different version'),
    ('assets_sha256.teacher', 'different weights'), ('train_paths_sha256', 'different sources'),
    ('precision', 'FP16'), ('tf32_matmul', True), ('config.output', '/different'),
    ('config.schedule.updates', 200000), ('fixed_protocol.manifest_sha256', 'different protocol'),
    ('code_sha256.rawfeat/losses.py', 'different math'), ('code_sha256.rawfeat/identity.py', 'unknown revision'),
])
def test_gpu_flag_rejects_other_identity_changes(identities, path, value):
    saved, current = identities
    target = current
    parts = path.split('.')
    # Source paths themselves contain a dot; only the first separator is structural here.
    if path.startswith('code_sha256.'):
        target = current['code_sha256']
        key = path.removeprefix('code_sha256.')
    else:
        for key in parts[:-1]:
            target = target[key]
        key = parts[-1]
    target[key] = value
    with pytest.raises(RuntimeError, match='identity differs'):
        check_resume_identity(saved, current, allow_gpu_change=True)


def test_gpu_flag_rejects_missing_identity_fields(identities):
    saved, current = identities
    del current['precision']
    with pytest.raises(RuntimeError, match='precision'):
        check_resume_identity(saved, current, allow_gpu_change=True)


def test_checkpoint_gpu_migration_preserves_full_state(tmp_path, monkeypatch, identities):
    import rawfeat.training as training
    saved, current = identities
    config = load_config('configs/smoke.yaml')
    model = RawFeatureExtractor()
    optimizer = make_optimizer(model, config)
    step = 299
    for name, parameter in model.named_parameters():
        if not name.startswith('descriptor.'):
            optimizer.state[parameter] = {'step': torch.tensor(float(step)),
                                          'exp_avg': torch.ones_like(parameter)*.01,
                                          'exp_avg_sq': torch.ones_like(parameter)*.001}
    state = checkpoint_payload(model, optimizer, config, step, saved, {'det': .2, 'geometry': None}, None, 3)
    state['rng']['torch_cuda'] = [torch.zeros(10, dtype=torch.uint8)]
    checkpoint = tmp_path/'state.pt'
    save_checkpoint(checkpoint, state)
    original = checkpoint.read_bytes()
    restored_rng = []
    monkeypatch.setattr(training, 'restore_rng', restored_rng.append)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    restored = RawFeatureExtractor()
    restored_optimizer = make_optimizer(restored, config)
    loaded = load_checkpoint(checkpoint, restored, restored_optimizer, config, current, allow_gpu_change=True)
    assert loaded['step'] == step and loaded['sample_cursor'] == 16*step
    assert loaded['scheduler'] == state['scheduler'] and loaded['best'] == state['best']
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in state['student'].items())
    for name, parameter in restored.named_parameters():
        if name.startswith('descriptor.'):
            assert not restored_optimizer.state[parameter]
        else:
            assert int(restored_optimizer.state[parameter]['step']) == step
            assert torch.equal(restored_optimizer.state[parameter]['exp_avg'], torch.ones_like(parameter)*.01)
            assert torch.equal(restored_optimizer.state[parameter]['exp_avg_sq'], torch.ones_like(parameter)*.001)
    assert torch.equal(restored_rng[0]['torch_cuda'][0], state['rng']['torch_cuda'][0])
    assert checkpoint.read_bytes() == original
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    with pytest.raises(RuntimeError, match='one saved and one visible'):
        load_checkpoint(checkpoint, restored, restored_optimizer, config, current, allow_gpu_change=True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    state['rng']['torch_cuda'].append(state['rng']['torch_cuda'][0])
    save_checkpoint(checkpoint, state)
    with pytest.raises(RuntimeError, match='one saved and one visible'):
        load_checkpoint(checkpoint, restored, restored_optimizer, config, current, allow_gpu_change=True)


def test_gpu_migration_option_requires_resume():
    args = _parser().parse_args(['train', '--config', 'configs/formal.yaml', '--resume', 'latest.pt', '--allow-gpu-change'])
    assert args.allow_gpu_change
    with pytest.raises(ValueError, match='requires --resume'):
        train('configs/formal.yaml', allow_gpu_change=True)


def test_reviewed_revision_hashes_match_installed_source():
    from rawfeat.identity import GPU_RESUME_REVISION, ROOT
    from rawfeat.validation import file_digest
    revision = json.loads(GPU_RESUME_REVISION.read_text())
    assert revision['target_code_sha256'] == {str(path.relative_to(ROOT)): file_digest(path)
                                             for path in sorted((ROOT/'rawfeat').glob('*.py'))}
    changed = {path for path in revision['source_code_sha256']
               if revision['source_code_sha256'][path] != revision['target_code_sha256'][path]}
    assert changed == {'rawfeat/cli.py', 'rawfeat/identity.py', 'rawfeat/training.py'}
