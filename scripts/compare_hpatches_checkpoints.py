"""Recompute paired HPatches checkpoint scores from complete sequence summaries."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(root: Path, reference_name: str, checkpoint_names: list[str] | None = None) -> dict:
    legacy_conditions = ('ratio1', 'ratio4', 'ratio16', 'ratio64', 'ratio100')
    extra_conditions = ('ratio150', 'ratio225')
    conditions = ('clean', *legacy_conditions, *extra_conditions)
    names = checkpoint_names or sorted(p.parent.parent.name for p in root.glob('*/legacy/summary.json'))
    models = {}
    for name in names:
        directory = root/name
        summaries = {family: json.loads((directory/family/'summary.json').read_text())
                     for family in ('legacy', 'extra')}
        legacy, extra = summaries['legacy'], summaries['extra']
        if not all(s['full_protocol'] and s['pair_count_per_condition'] == 580 for s in summaries.values()):
            raise ValueError(f'{name}: complete 580-pair HPatches results required')
        if (legacy['rawfeat_checkpoint_sha256'] != extra['checkpoint_sha256']
                or legacy['cache_identity_sha256'] != extra['legacy_cache_identity_sha256']
                or legacy['manifest_sha256'] != extra['manifest_sha256']):
            raise ValueError(f'{name}: legacy and extra result identities differ')
        rows = []
        for family, filename in [('legacy', 'sequence_summary.jsonl'), ('extra', 'per_sequence.jsonl')]:
            rows.extend(json.loads(line) for line in (directory/family/filename).read_text().splitlines())
        raw = {(r['sequence'], r['condition']): r for r in rows if r['method'] == 'rawfeat'}
        if (len(raw) != 116*8 or {c for _, c in raw} != set(conditions)
                or any(r['pair_count'] != 5 or r['h_auc_5'] is None
                       or not np.isfinite(r['h_auc_5']) for r in raw.values())):
            raise ValueError(f'{name}: incomplete or nonfinite sequence coverage')
        models[name] = dict(
            checkpoint_sha256=legacy['rawfeat_checkpoint_sha256'],
            cache_identity_sha256=legacy['cache_identity_sha256'], manifest_sha256=legacy['manifest_sha256'],
            full_protocol=True, pairs_per_condition=580,
            legacy_five_score=legacy['main_h_auc_5']['rawfeat'],
            baseline_five_score=legacy['main_h_auc_5']['superpoint'],
            by_condition={c: summaries['extra' if c in extra_conditions else 'legacy']['summaries']['rawfeat']['all'][c]['h_auc_5']
                          for c in conditions}, sequence_rows=raw)
    if reference_name not in models:
        raise ValueError(f'reference checkpoint result missing: {reference_name}')
    reference = models[reference_name]
    comparisons = {}
    for name, model in models.items():
        if (any(model[k] != reference[k] for k in ('manifest_sha256', 'cache_identity_sha256'))
                or model['sequence_rows'].keys() != reference['sequence_rows'].keys()):
            raise ValueError(f'{name}: checkpoint inputs differ from reference')
        groups = {}
        for group in ('all', 'illumination', 'viewpoint'):
            sequences = sorted({s for (s, _), row in reference['sequence_rows'].items()
                                if group == 'all' or row['type'] == group})
            groups[group] = {}
            for label, selected in [('legacy_five', legacy_conditions), ('extra_two', extra_conditions)]:
                delta = np.array([np.mean([model['sequence_rows'][s, c]['h_auc_5']-
                                          reference['sequence_rows'][s, c]['h_auc_5'] for c in selected]) for s in sequences])
                bootstrap = np.random.default_rng(20261028).choice(delta, size=(10000, len(delta)), replace=True).mean(1)
                groups[group][label] = dict(delta_percentage_points=float(delta.mean()*100),
                                           ci95_percentage_points=(np.quantile(bootstrap, [.025, .975])*100).tolist(),
                                           sequences=len(sequences))
        comparisons[name] = groups
    return dict(protocol='rawfeat.hpatches.checkpoint_comparison.v2', reference=reference_name,
                note='Original five-noise and150/225 scores are separate. Paired sequence bootstrap:10000 resamples, seed20261028, no multiplicity correction. These test results do not update training selection.',
                checkpoints={name: {k:v for k,v in model.items() if k != 'sequence_rows'} for name, model in models.items()},
                paired_vs_v1=comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='directory containing checkpoint legacy/extra results')
    parser.add_argument('--reference', default='step_000000', help='reference result directory name')
    parser.add_argument('--checkpoints', nargs='+', help='result directory names; defaults to every legacy result')
    parser.add_argument('--output', type=Path, help='output directory; defaults to --root')
    args = parser.parse_args()
    result = compare(args.root, args.reference, args.checkpoints)
    output = args.output or args.root
    output.mkdir(parents=True, exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# HPatches checkpoint comparison', '', result['note'], '',
             '| Checkpoint | Original five-noise mean | clean | ratio150 | ratio225 | Five-noise delta vs reference (pp) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, model in result['checkpoints'].items():
        c = model['by_condition']
        delta = result['paired_vs_v1'][name]['all']['legacy_five']['delta_percentage_points']
        lines.append(f"| {name} | {model['legacy_five_score']*100:.3f}% | {c['clean']*100:.3f}% | {c['ratio150']*100:.3f}% | {c['ratio225']*100:.3f}% | {delta:+.3f} |")
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
