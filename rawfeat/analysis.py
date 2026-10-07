"""Rebuild PNG/PDF curves and a final report from append-only raw run records."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .identity import write_json


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def analyze_run(run: str | Path) -> dict:
    run = Path(run)
    config = json.loads((run/'resolved_config.json').read_text())
    rows = read_jsonl(run/'scalars.jsonl')
    if not rows:
        return {'completed_updates': 0}
    if [r['step'] for r in rows] != list(range(1, rows[-1]['step']+1)):
        raise RuntimeError('analysis needs contiguous unique update records')
    output = run/'analysis'; output.mkdir(exist_ok=True)
    boundaries = [config['schedule']['detection_updates'], config['schedule']['detection_updates']+config['schedule']['matching_ramp']]
    diagnostics = [json.loads(p.read_text()) for p in sorted((run/'diagnostics').glob('step_*/summary.json'))]
    diagnostics = [r for r in diagnostics if r['step'] <= rows[-1]['step']]
    geometry = [json.loads(p.read_text()) for p in sorted((run/'validation').glob('step_*/selection.json'))]
    geometry = [r for r in geometry if r['step'] <= rows[-1]['step']]

    def figure(name, series):
        fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.7*len(series)), squeeze=False)
        for ax, (title, curves) in zip(axes[:, 0], series):
            for label, points, smooth in curves:
                x = [p[0] for p in points]; y = np.array([p[1] if p[1] is not None else np.nan for p in points], dtype=float)
                if smooth and len(y) >= 20:
                    line, = ax.plot(x, y, alpha=.25, linewidth=.6, label=label+' raw')
                    # Smoothing stays inside each phase and never replaces the raw records.
                    smoothed_labeled = False
                    for start, stop in zip([0, *boundaries], [*boundaries, config['schedule']['updates']]):
                        mask = (np.asarray(x) > start) & (np.asarray(x) <= stop) & np.isfinite(y)
                        values = y[mask]
                        if len(values) >= 20:
                            ax.plot(np.asarray(x)[mask][19:], np.convolve(values, np.ones(20)/20, mode='valid'),
                                    color=line.get_color(), label=label+' mean20' if not smoothed_labeled else None)
                            smoothed_labeled = True
                else:
                    ax.plot(x, y, '.-', markersize=3, label=label+' raw')
            for boundary in boundaries:
                ax.axvline(boundary, color='black', linestyle='--', alpha=.5)
            ax.set_xlim(0, max(1, rows[-1]['step']))
            ax.set_title(title); ax.set_xlabel('completed optimizer update'); ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc='best')
        fig.tight_layout(); fig.savefig(output/(name+'.png'), dpi=140); fig.savefig(output/(name+'.pdf')); plt.close(fig)

    def scalars(names):
        return [(name, [(r['step'], r.get(name)) for r in rows], True) for name in names]

    figure('losses', [('detector', scalars(['loss_det', 'loss_occupancy', 'loss_position'])),
                     ('raw matching (A unavailable)', scalars(['loss_match'])),
                     ('raw gray', scalars(['loss_gray'])),
                     ('weighted contributions and total', scalars(['loss_det', 'loss_match_weighted', 'loss_gray_weighted', 'loss_total']))])
    figure('schedule', [('learning rate', scalars(['lr'])), ('task coefficients', scalars(['matching_weight', 'gray_weight'])),
                       ('noise cap and sampled ratios', scalars(['ratio_limit', 'ratio_mean', 'ratio_max'])),
                       ('gradient clipping', scalars(['gradient_norm_before_clip', 'clip_factor'])),
                       ('time seconds', scalars(['update_seconds']))])
    if diagnostics:
        def diagnostic_curves(keys):
            return [(bucket+'/'+key, [(r['step'], r['buckets'][bucket]['mean'].get(key)) for r in diagnostics], False)
                    for key in keys for bucket in ('clean', '1', '4', '16', '64', '100')]
        figure('detection_geometry', [('teacher recall1px', diagnostic_curves(['formal_recall_1px'])),
                                    ('teacher recall3px', diagnostic_curves(['formal_recall_3px'])),
                                    ('fixedK teacher1px', diagnostic_curves(['top400_recall_1px', 'top1024_recall_1px'])),
                                    ('background occupancy error', diagnostic_curves(['foreground_mae_background'])),
                                    ('conditional phase Top1', diagnostic_curves(['phase_top1'])),
                                    ('automatic H-AUC@1/3/5 (A unavailable)',
                                     [(bucket+'/'+key, [(r['step'], r['geometry']['buckets'][bucket][key] if r['geometry'] else None) for r in geometry], False)
                                      for bucket in ('1', '4', '16', '64', '100') for key in ('h_auc_1', 'h_auc_3', 'h_auc_5')])])
        figure('gray', [('weighted gray error', diagnostic_curves(['gray_weighted_mse'])),
                        ('ordinary gray error', diagnostic_curves(['gray_ordinary_mse'])),
                        ('edge gradient error', diagnostic_curves(['edge_gradient_mse'])),
                        ('edge direction', diagnostic_curves(['edge_gradient_cosine'])),
                        ('clean-point retention (changing reference)', diagnostic_curves(['clean_point_retention_1px']))])
    phases = {}
    for phase in ('A', 'B', 'C'):
        phase_rows = [r for r in rows if r['phase'] == phase]
        if phase_rows:
            phases[phase] = {'updates': len(phase_rows), 'mean_update_seconds': float(np.mean([r['update_seconds'] for r in phase_rows])),
                             'median_update_seconds': float(np.median([r['update_seconds'] for r in phase_rows])),
                             'first_loss': phase_rows[0]['loss_total'], 'last_loss': phase_rows[-1]['loss_total'],
                             'max_memory_bytes': max(r['max_memory_bytes'] for r in phase_rows)}
    overhead = read_jsonl(run/'overhead.jsonl')
    validation_costs = [r for batch in overhead for r in batch['validation']]
    saves = [r for batch in overhead for r in batch['checkpoint']]
    estimate = sum(phases[p]['mean_update_seconds']*count for p, count in [('A', 50000), ('B', 5000), ('C', 45000)]) if len(phases) == 3 else None
    comparisons = {}
    for label, filename in [('last', 'latest.pt'), ('best_det', 'best_det.pt'), ('best_geometry', 'best_geometry.pt')]:
        target = run/filename
        if target.exists() and target.resolve().stem.startswith('checkpoint_step_'):
            selected_step = int(target.resolve().stem.split('_')[-1])
            selected = next((g for g in geometry if g['step'] == selected_step), None)
            if selected is not None:
                comparisons[label] = {'step': selected_step, 'teacher_recall_1px': selected['detection']['teacher_recall_1px'],
                                      'geometry': selected['geometry'], 'detection_buckets': selected['detection']['buckets']}
    journal = read_jsonl(run/'optimizer_updates.jsonl')
    report = {'run': str(run.resolve()), 'configured_updates': config['schedule']['updates'], 'completed_updates': rows[-1]['step'],
              'complete': rows[-1]['step'] == config['schedule']['updates'], 'actual_optimizer_updates': len(journal),
              'best_last_comparison': comparisons, 'phase_boundaries': boundaries, 'phases': phases,
              'formal_training_seconds_estimate': estimate, 'validation_seconds': validation_costs, 'checkpoint_seconds': saves,
              'latest_selection': geometry[-1] if geometry else None,
              'best_checkpoints': {name: str((run/name).resolve()) if (run/name).exists() else None for name in ('best_det.pt', 'best_geometry.pt', 'latest.pt')},
              'notes': 'Estimate scales smoke update time only. Shared GPU load, longer convergence, full256 validation, gradient probes, disk/plot costs add uncertainty. Raw and mean20 curves are separate.'}
    write_json(output/'summary.json', report)
    lines = [f'# RawFeat staged run: {rows[-1]["step"]}/{config["schedule"]["updates"]}',
             f'Identity: `{json.loads((run/"run_manifest.json").read_text())["identity_sha256"]}`',
             '', 'Raw data: scalars.jsonl, diagnostics/step_*/per_view.jsonl and per_pair.jsonl, validation/step_*/per_pair.jsonl.',
             'A has no matching/geometry loss; missing values remain null. Best selection requires the full256 protocol, with geometry selected only in C.',
             '', '|Phase|Updates|Mean seconds/update|First total|Last total|', '|---|---:|---:|---:|---:|']
    lines += [f'|{p}|{v["updates"]}|{v["mean_update_seconds"]:.3f}|{v["first_loss"]:.5f}|{v["last_loss"]:.5f}|' for p, v in phases.items()]
    if geometry:
        selected = geometry[-1]
        lines += ['', f'Last diagnostic: phase {selected["phase"]}, sources {selected["detection"]["sources"]}; teacher1px={selected["detection"]["teacher_recall_1px"]}.']
        if selected['geometry']:
            g = selected['geometry']
            lines += [f'Last automatic geometry: {g["cases"]} pairs; full_protocol={g["full_protocol"]}; H-AUC@5={g["score"]:.6f}; paired baseline={g.get("baseline_score")}.']
            lines += ['', '|Bucket|H-AUC@1|H-AUC@3|H-AUC@5|Failures|', '|---|---:|---:|---:|---:|']
            lines += [f'|{bucket}|{v["h_auc_1"]:.6f}|{v["h_auc_3"]:.6f}|{v["h_auc_5"]:.6f}|{v["homography_failures"]}|' for bucket, v in g['buckets'].items()]
    lines += ['', 'Best/last comparison is stored with per-noise detector/geometry and the paired baseline in summary.json. Missing full-protocol best checkpoints remain null, including smoke subset selection.',
              f'100000-update compute estimate: {estimate} seconds; validation/saving costs are listed in summary.json. This is an estimate, not a timing guarantee.',
              'Use all per-step and per-noise records to interpret gray/geometry; a short smoke score does not establish the stage allocation or gray10 as optimal.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return {k: report[k] for k in ('run', 'completed_updates', 'complete', 'formal_training_seconds_estimate')}
