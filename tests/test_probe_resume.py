"""Only the exact reviewed diagnostic revision may restore the failed V2 run."""
import copy
import json

import pytest

from rawfeat.cli import _parser
from rawfeat.identity import check_resume_identity
from rawfeat.training import train


@pytest.fixture
def probe_revision(tmp_path, monkeypatch):
    import rawfeat.identity as identity
    saved = {'code_sha256': {'rawfeat/diagnostics.py': 'before', 'rawfeat/losses.py': 'same'},
             'cuda_visible_devices': '3', 'gpu': 'NVIDIA GeForce RTX 5090',
             'config': {'schedule': {'updates': 30000}}, 'dependencies': {'torch': 'same'},
             'assets_sha256': {'teacher': 'same'}, 'train_paths_sha256': 'same'}
    current = copy.deepcopy(saved)
    current['code_sha256']['rawfeat/diagnostics.py'] = 'after'
    registry = tmp_path/'probe_revision.json'
    registry.write_text(json.dumps({'source_code_sha256': saved['code_sha256'],
                                    'target_code_sha256': current['code_sha256']}))
    monkeypatch.setattr(identity, 'PROBE_RESUME_REVISION', registry)
    return saved, current


def test_probe_revision_requires_its_explicit_flag(probe_revision):
    saved, current = probe_revision
    original = copy.deepcopy(saved)
    for flags in ({}, {'allow_gpu_change': True}):
        with pytest.raises(RuntimeError, match='code_sha256'):
            check_resume_identity(saved, current, **flags)
    assert check_resume_identity(saved, current, allow_probe_fix=True) == ['code_sha256.rawfeat/diagnostics.py']
    assert check_resume_identity(current, current) == []
    assert saved == original
    with pytest.raises(RuntimeError, match='code_sha256'):
        check_resume_identity(current, saved, allow_probe_fix=True)


def test_probe_flag_does_not_authorize_gpu_move(probe_revision):
    saved, current = probe_revision
    current['cuda_visible_devices'] = '0'
    with pytest.raises(RuntimeError, match='cuda_visible_devices'):
        check_resume_identity(saved, current, allow_probe_fix=True)
    assert check_resume_identity(saved, current, allow_probe_fix=True, allow_gpu_change=True) == [
        'code_sha256.rawfeat/diagnostics.py', 'cuda_visible_devices']


@pytest.mark.parametrize('field,key,value', [
    ('code_sha256', 'rawfeat/diagnostics.py', 'unknown revision'),
    ('code_sha256', 'rawfeat/losses.py', 'different math'),
    ('dependencies', 'torch', 'different version'),
    ('assets_sha256', 'teacher', 'different weights'),
    ('config', 'schedule', {'updates': 60000}),
    (None, 'train_paths_sha256', 'different data'),
    (None, 'gpu', 'different GPU model'),
])
def test_probe_flag_rejects_unreviewed_changes(probe_revision, field, key, value):
    saved, current = probe_revision
    (current if field is None else current[field])[key] = value
    with pytest.raises(RuntimeError, match='identity differs'):
        check_resume_identity(saved, current, allow_probe_fix=True, allow_gpu_change=True)


def test_probe_flag_requires_resume():
    args = _parser().parse_args(['train', '--config', 'configs/formal_v2.yaml',
                                 '--resume', 'latest.pt', '--allow-probe-fix'])
    assert args.allow_probe_fix
    with pytest.raises(ValueError, match='--allow-probe-fix requires --resume'):
        train('configs/formal_v2.yaml', allow_probe_fix=True)


def test_reviewed_probe_registry_is_immutable_and_changes_only_diagnostics_and_resume():
    from rawfeat.identity import PROBE_RESUME_REVISION
    from rawfeat.validation import file_digest
    assert file_digest(PROBE_RESUME_REVISION) == 'de7a17d625d83178544f89006f47ccf52dc21af922793ab12b3196655d5d20c0'
    revision = json.loads(PROBE_RESUME_REVISION.read_text())
    assert revision['source_code_sha256'].keys() == revision['target_code_sha256'].keys()
    changed = {name for name, digest in revision['source_code_sha256'].items()
               if revision['target_code_sha256'][name] != digest}
    assert changed == {'rawfeat/cli.py', 'rawfeat/diagnostics.py', 'rawfeat/identity.py', 'rawfeat/training.py'}
