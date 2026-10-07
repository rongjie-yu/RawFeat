from pathlib import Path
import pytest
import yaml

from rawfeat.training import _link_checkpoint, check_training_output, load_config


def test_best_and_latest_pointers_replace_atomically(tmp_path: Path):
    first = tmp_path / "checkpoint_step_002000.pt"
    second = tmp_path / "checkpoint_step_004000.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    _link_checkpoint(tmp_path, "latest.pt", first)
    _link_checkpoint(tmp_path, "best.pt", first)
    _link_checkpoint(tmp_path, "latest.pt", second)
    assert (tmp_path / "latest.pt").read_bytes() == b"second"
    assert (tmp_path / "best.pt").read_bytes() == b"first"
    _link_checkpoint(tmp_path, "best.pt", second)
    assert (tmp_path / "best.pt").read_bytes() == b"second"


def test_new_training_cannot_overwrite_existing_history(tmp_path):
    check_training_output(tmp_path, None)
    checkpoint = tmp_path / "checkpoint_step_000050.pt"
    checkpoint.write_bytes(b"retained")
    with pytest.raises(RuntimeError, match="already contains a run"):
        check_training_output(tmp_path, None)
    check_training_output(tmp_path, checkpoint)
    assert checkpoint.read_bytes() == b"retained"


def test_formal_training_rejects_partial_model_selection(tmp_path):
    config = yaml.safe_load(Path("configs/formal.yaml").read_text())
    config["validation_images"] = 16
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="all 256"):
        load_config(path)


def test_legacy_schema_rejected(tmp_path):
    config = yaml.safe_load(Path('configs/formal.yaml').read_text())
    config['detection_objective'] = 'kl'
    path = tmp_path/'old.yaml'; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='schema'): load_config(path)


def test_resume_requires_same_run(tmp_path):
    output = tmp_path/'run'; output.mkdir()
    other = tmp_path/'other'; other.mkdir()
    with pytest.raises(RuntimeError, match='same run'): check_training_output(output, other/'latest.pt')


def test_log_recovery_archives_successful_updates_not_in_checkpoint(tmp_path):
    import json
    from rawfeat.training import reconcile_logs
    log = tmp_path/'scalars.jsonl'
    log.write_text(''.join(json.dumps({'step': k})+'\n' for k in range(1, 6)))
    reconcile_logs(tmp_path, 3)
    assert [json.loads(r)['step'] for r in log.read_text().splitlines()] == [1, 2, 3]
    assert len(list(tmp_path.glob('scalars_before_recovery_*.jsonl'))) == 1
    log.write_text('{"step":1}\n{"step":3}\n')
    with pytest.raises(RuntimeError, match='missing/duplicate'): reconcile_logs(tmp_path, 3)


def test_best_selection_excludes_random_A_and_partial_geometry():
    from rawfeat.training import update_best
    best = {'det': None, 'geometry': None}
    full = {'full_protocol': True, 'teacher_recall_1px': .1}
    geometry = {'full_protocol': True, 'score': .9}
    assert update_best(best, 'A', full, geometry) == (True, False)
    assert best == {'det': .1, 'geometry': None}
    assert update_best(best, 'B', full, geometry) == (False, False)
    assert update_best(best, 'C', full, {'full_protocol': False, 'score': .99}) == (False, False)
    assert update_best(best, 'C', full, geometry) == (False, True)
    assert best['geometry'] == .9


def test_checkpoint_retention_preserves_boundaries_and_best(tmp_path):
    from rawfeat.training import retain_checkpoints
    for step in (100, 200, 300, 400, 500, 600, 700, 800):
        (tmp_path/f'checkpoint_step_{step:06d}.pt').write_bytes(b'test')
    _link_checkpoint(tmp_path, 'best_det.pt', tmp_path/'checkpoint_step_000100.pt')
    _link_checkpoint(tmp_path, 'latest.pt', tmp_path/'checkpoint_step_000800.pt')
    retain_checkpoints(tmp_path, {300, 400, 600})
    assert (tmp_path/'checkpoint_step_000100.pt').exists()
    assert all((tmp_path/f'checkpoint_step_{step:06d}.pt').exists() for step in (300, 400, 600, 800))

    assert not (tmp_path/'checkpoint_step_000200.pt').exists()
