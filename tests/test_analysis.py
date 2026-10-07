"""Isolated analysis fixtures, never student training records."""
import json
import matplotlib.figure
from rawfeat.config import load_config
from rawfeat.analysis import analyze_run


def test_early_formal_curves_keep_actual_horizon_and_separate_gray(tmp_path, monkeypatch):
    config = load_config('configs/formal.yaml')
    (tmp_path/'resolved_config.json').write_text(json.dumps(config))
    (tmp_path/'run_manifest.json').write_text(json.dumps({'identity_sha256': 'analysis-fixture'}))
    rows = [{'step': step, 'phase': 'A', 'loss_total': 1., 'loss_gray': .1/step,
             'update_seconds': .01, 'max_memory_bytes': 0} for step in range(1, 26)]
    (tmp_path/'scalars.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    axes = []
    def inspect_figure(figure, *args, **kwargs):
        axes.extend((ax.get_title(), ax.get_xlim()) for ax in figure.axes)
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', inspect_figure)
    report = analyze_run(tmp_path)
    assert report['completed_updates'] == 25 and not report['complete']
    assert all(limits == (0., 25.) for _, limits in axes)
    assert any(title == 'raw gray' for title, _ in axes)
    assert any(title == 'raw matching (A unavailable)' for title, _ in axes)
