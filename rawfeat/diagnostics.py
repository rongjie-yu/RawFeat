"""Current fixed-noise detector/gray/oracle/automatic-point diagnostics and task gradients."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .features import extract_features, sample_descriptors
from .identity import write_json
from .losses import detection_terms, pair_losses
from .metrics import pair_metrics
from .schedules import update_schedule
from .validation import RATIOS, cache_path, identity_digest, read_cache_identity, read_manifest


def mean_or_none(value: torch.Tensor) -> float | None:
    return float(value.float().mean()) if value.numel() else None


def average(rows: list[dict]) -> dict:
    keys = set().union(*(r.keys() for r in rows))
    result = {}
    for key in keys:
        values = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        if values or any(key in r and r[key] is None for r in rows):
            result[key] = float(np.mean(values)) if values else None
    return result


def recall_metrics(reference: torch.Tensor, points: torch.Tensor) -> dict:
    distance = (torch.cdist(reference, points).min(1).values if len(reference) and len(points)
                else reference.new_full((len(reference),), float('inf')))
    return {'reference_count': len(reference), 'selected_count': len(points),
            'recall_1px': mean_or_none(distance <= 1), 'recall_3px': mean_or_none(distance <= 3),
            'recalled_1px': int((distance <= 1).sum()), 'recalled_3px': int((distance <= 3).sum()),
            'localization_3px': mean_or_none(distance[distance <= 3])}


def detector_view(logits, teacher, valid, points) -> dict:
    terms = detection_terms(logits[None], teacher[None], valid[None])
    ls, lt = logits.log_softmax(0), teacher.log_softmax(0)
    ms, mt = ls[:64].logsumexp(0).exp(), lt[:64].logsumexp(0).exp()
    qs, qt = logits[:64].softmax(0), teacher[:64].softmax(0)
    salient = valid & (lt[:64].exp().max(0).values >= .005)
    background = valid & ~salient
    entropy = -(qs * qs.clamp_min(1e-30).log()).sum(0)
    result = {'valid_cells': int(valid.sum()), 'salient_cells': int(salient.sum()), 'background_cells': int(background.sum()),
              'occupancy': float(terms['occupancy'][0]), 'position': float(terms['position_mass'][0]),
              'teacher_mass': float(terms['teacher_mass'][0]), 'position_scale': float(terms['position_scale'][0]),
              'teacher_mass_mean': mean_or_none(mt[valid]), 'student_mass_mean': mean_or_none(ms[valid]),
              'conditional_entropy': mean_or_none(entropy[valid]),
              'foreground_mae': mean_or_none((ms-mt).abs()[valid]),
              'foreground_mae_salient': mean_or_none((ms-mt).abs()[salient]),
              'foreground_mae_background': mean_or_none((ms-mt).abs()[background]),
              'background_student_mass': mean_or_none(ms[background]),
              'background_student_peak': mean_or_none(ls[:64].exp().max(0).values[background]),
              'salient_phase_agreement': mean_or_none((qs.argmax(0) == qt.argmax(0))[salient]),
              'teacher_nms_points': len(points)}
    x, y = points.long().T
    usable = valid[y//8, x//8]
    result['teacher_nms_in_valid_cell'] = int(usable.sum())
    x, y = x[usable], y[usable]
    phase = y%8*8 + x%8
    values = qs[:, y//8, x//8].T
    ranks = 1 + (values > values.gather(1, phase[:, None])).sum(1)
    predicted = values.argmax(1)
    dx, dy = predicted%8 - phase%8, predicted//8 - phase//8
    result.update(phase_top1=mean_or_none(ranks == 1), phase_top5=mean_or_none(ranks <= 5),
                  phase_dx=mean_or_none(dx), phase_dy=mean_or_none(dy),
                  phase_error=mean_or_none(torch.stack((dx, dy), 1).float().norm(dim=1)),
                  teacher_phase_histogram=torch.bincount(phase, minlength=64).tolist(),
                  student_phase_histogram=torch.bincount(predicted, minlength=64).tolist())
    result['teacher_mass_quantiles'] = [float(v) for v in torch.quantile(mt[valid], mt.new_tensor([0., .1, .5, .9, 1.]))]
    return result


def gray_metrics(predicted, target, valid, packed) -> dict:
    y, p = target[valid], predicted[valid]
    weight = (y+.01).pow(-2)
    pg, tg = [], []
    for dim in (-1, -2):
        mask = valid.narrow(dim, 0, valid.shape[dim]-1) & valid.narrow(dim, 1, valid.shape[dim]-1)
        pg.append(predicted.diff(dim=dim)[mask]); tg.append(target.diff(dim=dim)[mask])
    a, b = torch.cat(pg), torch.cat(tg)
    edge = b.abs() >= .02
    proxy = .299*packed[0] + .587*(packed[1]+packed[2])/2 + .114*packed[3]
    proxy = F.interpolate(proxy[None, None], size=target.shape[-2:], mode='bilinear', align_corners=False)[0]
    return {'gray_weighted_mse': float((weight*(p-y).square()).sum()/weight.sum()),
            'gray_ordinary_mse': mean_or_none((p-y).square()),
            'edge_gradient_mse': mean_or_none((a[edge]-b[edge]).square()),
            'edge_gradient_cosine': float(F.cosine_similarity(a[edge], b[edge], dim=0)) if edge.any() else None,
            'edge_gradient_energy_ratio': float(a[edge].norm()/b[edge].norm()) if edge.any() else None,
            'edge_count': int(edge.sum()), 'proxy_ordinary_mse': mean_or_none((proxy[valid]-y).square()),
            'proxy_weighted_mse': float((weight*(proxy[valid]-y).square()).sum()/weight.sum())}


@torch.no_grad()
def diagnose_model(model, teacher, config, output: Path, step: int, *, images: int = 16, geometry: bool = True) -> dict:
    from .training import point_pairs
    if model.training or not 1 <= images <= 256:
        raise ValueError('diagnostics require an eval copy and 1..256 fixed sources')
    manifest, digest = read_manifest(config['validation_manifest'])
    identity = read_cache_identity(config['validation_cache'], digest)
    device = next(model.parameters()).device
    clean_points, clean_targets = {}, {}
    rows, pair_rows, buckets = [], [], {}
    for bucket in ('clean', *map(str, RATIOS)):
        selected_rows, selected_pairs = [], []
        for index, image in enumerate(manifest['images'][:images]):
            pair = torch.load(cache_path(config['validation_cache'], index, bucket), map_location='cpu', weights_only=True)
            pair = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in pair.items()}
            logits_t, grid_t = teacher(pair['srgb_gray'])
            valid = pair['valid_pixels'][:, 0]
            references = extract_features(logits_t, grid_t, valid=valid)
            out = model(pair['packed'], compute_descriptors=geometry)
            grid = out['descriptors'] if geometry else out['logits'].new_zeros((2, 1, *out['logits'].shape[-2:]))
            selections = {name: extract_features(out['logits'], grid, threshold=threshold, max_points=k, valid=valid)
                          for name, threshold, k in (('formal', .005, 1024), ('top400', -float('inf'), 400), ('top1024', -float('inf'), 1024))}
            if bucket == 'clean':
                clean_points[index] = [f['points'].clone() for f in selections['formal']]
                clean_targets[index] = pair['clean_gray'].clone()
            if not torch.equal(clean_targets[index], pair['clean_gray']):
                raise RuntimeError('noise buckets do not share the same clean supervision')
            oracle = {'oracle_available': geometry, 'oracle_retrieval_top1': None, 'oracle_correspondences': None,
                      'oracle_diagonal_cosine': None, 'oracle_offdiagonal_cosine': None}
            if geometry:
                (pa, pb), _ = point_pairs(references, pair, 10527000+index, device)
                a = sample_descriptors(grid[0], pa, 480, 640); b = sample_descriptors(grid[1], pb, 480, 640)
                sim = a@b.T; diagonal = torch.arange(len(a), device=device)
                oracle.update(oracle_retrieval_top1=float(((sim.argmax(1) == diagonal).float().mean() + (sim.argmax(0) == diagonal).float().mean())/2),
                              oracle_correspondences=len(a), oracle_diagonal_cosine=mean_or_none(sim.diagonal()),
                              oracle_offdiagonal_cosine=mean_or_none(sim[~torch.eye(len(a), dtype=torch.bool, device=device)]))
                for name, features in selections.items():
                    numpy_features = [{k: v.cpu().numpy() for k, v in f.items()} for f in features]
                    metrics = pair_metrics(*numpy_features, pair['homography'].cpu().numpy(), valid[0].cpu().numpy(), valid[1].cpu().numpy())
                    metrics = {k: v if np.isfinite(v) else None for k, v in metrics.items()}
                    selected_pairs.append({'image_position': index, 'source': image['image'], 'bucket': bucket,
                                           'selection': name, 'geometry_available': True, **metrics})
            for view in (0, 1):
                metrics = detector_view(out['logits'][view], logits_t[view], pair['valid_cells'][view], references[view]['points'])
                metrics.update(gray_metrics(out['gray'][view], pair['clean_gray'][view], pair['valid_pixels'][view], pair['packed'][view]))
                stability = recall_metrics(clean_points[index][view], selections['formal'][view]['points'])
                metrics.update(clean_point_retention_1px=stability['recall_1px'], clean_point_retention_3px=stability['recall_3px'],
                               clean_reference_count=stability['reference_count'], **oracle)
                for name, features in selections.items():
                    recalls = recall_metrics(references[view]['points'], features[view]['points'])
                    metrics.update({name+'_'+k: v for k, v in recalls.items()})
                selected_rows.append({'image_position': index, 'source': image['image'], 'view': view,
                                      'bucket': bucket, 'actual_ratio': pair['ratios'][view], 'clean': bucket == 'clean', **metrics})
        buckets[bucket] = {'views': len(selected_rows), 'sources': images, 'mean': average(selected_rows),
                           'selections': {name: average([r for r in selected_pairs if r['selection'] == name])
                                          for name in ('formal', 'top400', 'top1024')}}
        rows.extend(selected_rows); pair_rows.extend(selected_pairs)
    actual = {str(ratio): {'views': len(values), 'mean': average(values)} for ratio in RATIOS
              if (values := [r for r in rows if not r['clean'] and r['actual_ratio'] == ratio])}
    recall = [buckets[str(r)]['mean']['formal_recall_1px'] for r in RATIOS]
    report = {'step': step, 'phase': update_schedule(max(0, step-1), config['schedule'])['phase'],
              'optimizer_updates': 0, 'full_protocol': images == 256, 'sources': images, 'cases': images*6,
              'cache_identity_sha256': identity_digest(identity), 'geometry_available': geometry,
              'teacher_recall_1px': float(np.mean(recall)) if all(v is not None for v in recall) else None,
              'buckets': buckets, 'actual_ratios': actual,
              'notes': {'recall': 'teacher NMS reference, per-view average; H-AUC@1 is a separate automatic homography metric',
                        'phase': 'teacher NMS points in all-valid cells; coverage counts reported',
                        'oracle': 'known teacher/uniform positions, not student automatic geometry',
                        'stability': 'clean student reference changes with checkpoint; retention alone does not prove robustness',
                        'proxy': 'bilinear packed Bayer gray proxy includes interpolation error',
                        'subset': 'first fixed sources; compressed subset has both-side noise, full256 includes balanced one-side cases'}}
    output.mkdir(parents=True, exist_ok=True)
    (output/'per_view.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    (output/'per_pair.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in pair_rows))
    write_json(output/'summary.json', report)
    return report


def flatten(gradients, parameters) -> torch.Tensor:
    return torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for g, p in zip(gradients, parameters)])


def cosine(a, b) -> float | None:
    return float(F.cosine_similarity(a, b, dim=0)) if a.norm() > 0 and b.norm() > 0 else None


def gradient_probe(model, batches: list[dict], config: dict, step: int, optimizer) -> dict:
    """Four-micro weighted task gradients on a disposable model; no optimizer step."""
    from .training import set_training_mode
    tested = copy.deepcopy(model)
    set_training_mode(tested, config, step)
    params = list(tested.parameters()); active_indices = [i for i, p in enumerate(params) if p.requires_grad]
    active = [params[i] for i in active_indices]
    weight = update_schedule(step, config['schedule'])['matching_weight']
    vectors, losses_mean = {}, {}
    for batch in batches:
        out = tested(batch['packed'], compute_descriptors=weight > 0)
        losses = pair_losses(out, batch['teacher_logits'], batch['valid_cells'], batch['clean_gray'], batch['valid_pixels'], batch['matches'], weight)
        objectives = {'det': losses['det'], 'gray': 10*losses['gray'],
                      'match': weight*losses['match'] if losses['match'] is not None else None, 'total': losses['total']}
        for task, objective in objectives.items():
            if objective is None:
                continue
            gradients = torch.autograd.grad(objective/4, active, retain_graph=True, allow_unused=True)
            full = [None]*len(params)
            for i, g in zip(active_indices, gradients): full[i] = g
            vector = flatten(full, params)
            vectors[task] = vectors.get(task, torch.zeros_like(vector)) + vector
            losses_mean[task] = losses_mean.get(task, 0.) + float(objective.detach())/4
    total = vectors['det'] + vectors['gray']
    if 'match' in vectors: total = total + vectors['match']
    relative = float((total-vectors['total']).norm()/vectors['total'].norm())
    if not torch.isfinite(total).all() or relative > 1e-5:
        raise RuntimeError('accumulated task gradients do not reconstruct total')
    groups, offset = {}, 0
    for name, p in tested.named_parameters():
        if name.startswith('stage'):
            label = 's'+name[5]
        elif name.startswith('gray'):
            label = 'gray'
        else:
            label = name.split('.')[0]
        groups.setdefault(label, []).append((offset, offset+p.numel()))
        offset += p.numel()
    stats = {}
    for group, ranges in groups.items():
        selected = {task: torch.cat([v[a:b] for a, b in ranges]) for task, v in vectors.items()}
        stats[group] = {'weighted_gradient_norms': {task: float(v.norm()) for task, v in selected.items()},
                        'cos_det_match': cosine(selected['det'], selected['match']) if 'match' in selected else None,
                        'cos_det_gray': cosine(selected['det'], selected['gray'])}
        if 'match' not in vectors: stats[group]['weighted_gradient_norms']['match'] = None
    adam = {name: int(optimizer.state[p]['step']) if optimizer.state[p] else 0 for name, p in model.named_parameters()}
    norm = float(vectors['total'].norm())
    return {'optimizer_updates': 0, 'next_update': step+1, 'microbatches': len(batches),
            'matching_available': weight > 0, 'matching_weight': weight, 'gray_weight': 10.,
            'weighted_losses': {**losses_mean, 'match': losses_mean.get('match')}, 'groups': stats,
            'global_norm': norm, 'clip_factor': min(1., 5/(norm+1e-6)), 'gradient_sum_relative_l2': relative,
            'adam_steps': adam, 'probe_context': 'same complete accumulated inputs on a training copy after committed update; no live BN/RNG changes'}
