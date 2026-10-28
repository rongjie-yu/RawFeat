"""V2 fixed COCO + held-out MegaDepth validation; legacy caches stay read-only."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .data_v2 import RATIOS, VAL_PROTOCOL, extra_cases
from .features import extract_features
from .metrics import pair_metrics
from .validation import (cache_identity as legacy_identity, cache_path, case_request, file_digest,
                         identity_digest, manifest_digest, read_cache_identity as read_legacy_cache,
                         read_manifest as read_legacy_manifest, summarize_buckets)


def read_manifest(config):
    manifest = json.loads(Path(config['validation_manifest']).read_text())
    sources = json.loads(Path(config['source_manifest']).read_text())
    legacy, digest = read_legacy_manifest(config['legacy_validation_manifest'])
    smoke = config['mode'] == 'smoke_v2'
    nc, nm = (2, 2) if smoke else (256, 200)
    if (manifest['protocol'] != VAL_PROTOCOL or manifest['seed'] != 2027
            or manifest['mode'] != ('smoke' if smoke else 'formal')
            or manifest['roots'] != sources['roots']
            or manifest['source_manifest_sha256'] != manifest_digest(sources)
            or manifest['legacy_manifest_sha256'] != digest
            or manifest['image_count'] != nc+nm or manifest['case_count'] != 8*(nc+nm)
            or len(manifest['images']) != nc+nm):
        raise ValueError('V2 validation manifest identity/counts differ')
    md_rows = sources['megadepth_validation']
    if len(md_rows) != nm or len(set(sources['validation_scenes'])) != (1 if smoke else 10):
        raise ValueError('V2 validation scene/source counts differ')
    for index, image in enumerate(manifest['images']):
        if index < nc:
            expected = legacy['images'][index]
            if (image['dataset'] != 'coco' or any(image[k] != v for k, v in expected.items() if k != 'cases')
                    or image['cases'] != expected['cases'] + extra_cases(index, 256, 0x434F)[-2:]):
                raise ValueError('V2 must preserve every legacy COCO case and supervision seed')
        else:
            position = index-nc
            row = md_rows[position]
            if (image['dataset'] != 'megadepth' or image['scene'] != row['scene'] or image['image'] != row['image']
                    or image['cases'] != extra_cases(position, nm, 0x4D44)):
                raise ValueError('V2 MegaDepth validation assignment differs')
        if [c['bucket'] for c in image['cases']] != ['clean', *map(str, RATIOS)]:
            raise ValueError('V2 requires clean and all seven noisy buckets')
    return manifest, manifest_digest(manifest)


def validation_indices(manifest, images):
    coco = [i for i, r in enumerate(manifest['images']) if r['dataset'] == 'coco']
    md = [i for i, r in enumerate(manifest['images']) if r['dataset'] == 'megadepth']
    if images >= len(coco):
        return list(range(len(manifest['images'])))
    # Round robin scenes so every small diagnostic samples the available geography.
    scenes = sorted({manifest['images'][i]['scene'] for i in md})
    groups = [[i for i in md if manifest['images'][i]['scene'] == s] for s in scenes]
    balanced = [g[j] for j in range(max(map(len, groups))) for g in groups if j < len(g)]
    return coco[:images] + balanced[:max(1, round(images*1.5))]


def cache_identity(config, digest):
    identity = legacy_identity(digest, config['invisp_repo'], config['eld_calibration'])
    return {**identity, 'protocol': VAL_PROTOCOL, 'recipe': 'rawfeat.synthetic.cache.v2',
            'v2_code_sha256': {name: file_digest(Path(__file__).with_name(name))
                               for name in ('data_v2.py', 'validation_v2.py')}}


def read_cache_identity(config, digest):
    identity = json.loads((Path(config['validation_cache'])/'identity.json').read_text())
    if identity != cache_identity(config, digest):
        raise RuntimeError('V2 cache identity differs; rebuild in a new cache directory')
    return identity


@torch.no_grad()
def cache_validation(config, generator):
    manifest, digest = read_manifest(config)
    root = Path(config['validation_cache'])
    root.mkdir(parents=True, exist_ok=True)
    identity = cache_identity(config, digest)
    if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text()) != identity:
        raise RuntimeError('V2 cache identity differs; choose a new output directory')
    _, legacy_digest = read_legacy_manifest(config['legacy_validation_manifest'])
    old_root = Path(config['legacy_validation_cache'])
    old_identity = read_legacy_cache(old_root, legacy_digest)
    if old_identity != legacy_identity(legacy_digest, config['invisp_repo'], config['eld_calibration']):
        raise RuntimeError('legacy cache assets differ')
    (root/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    completed = 0
    for position, image in enumerate(manifest['images']):
        paths = [cache_path(root, position, c['bucket']) for c in image['cases']]
        for case, dest in zip(image['cases'], paths):
            old = cache_path(old_root, position, case['bucket'])
            if image['dataset'] == 'coco' and case['bucket'] not in ('150', '225') and not dest.exists():
                # Same output filesystem: hardlinks reuse immutable V1 tensors without copying.
                os.link(old, dest)
        if not all(p.exists() for p in paths):
            source_root = manifest['roots'][image['dataset']]
            base = generator.generate_base(case_request(image, image['cases'][0], source_root), fixed_crop=True)
            for case, path in zip(image['cases'], paths):
                if path.exists():
                    continue
                pair = generator.apply_noise(base, case_request(image, case, source_root))
                stored = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in pair.items() if k != 'clean_bayer_dn'}
                temporary = path.with_suffix('.tmp')
                torch.save(stored, temporary)
                temporary.replace(path)
        completed += len(paths)
        print(json.dumps({'cached_images': position+1, 'total_images': len(manifest['images'])}), flush=True)
    return dict(cached_cases=completed, cache=str(root), identity_sha256=identity_digest(identity))


def preflight(config, pool):
    manifest, digest = read_manifest(config)
    identity = read_cache_identity(config, digest)
    missing = [str(cache_path(config['validation_cache'], i, c['bucket']))
               for i, row in enumerate(manifest['images']) for c in row['cases']
               if not cache_path(config['validation_cache'], i, c['bucket']).is_file()]
    if missing:
        raise FileNotFoundError(f'V2 cache missing {len(missing)} cases, first: {missing[0]}')
    if not Path(config['initial_checkpoint']).is_file():
        raise FileNotFoundError(config['initial_checkpoint'])
    free, total = torch.cuda.mem_get_info(torch.device(config['device']))
    return dict(ready=True, cached_cases=manifest['case_count'], manifest_sha256=digest,
                cache_identity_sha256=identity_digest(identity), train_sources=len(pool['metadata']),
                train_paths_sha256=pool['manifest_sha256'], source_counts={'coco': len(pool['coco']),
                'megadepth': sum(map(len, pool['megadepth'].values()))},
                free_memory_bytes=free, total_memory_bytes=total, gpu=torch.cuda.get_device_name(),
                torch=torch.__version__, cuda=torch.version.cuda)


def summarize(rows):
    domains = {}
    for dataset in ('coco', 'megadepth'):
        domain_rows = [r for r in rows if r['dataset'] == dataset]
        scenes = sorted({r['scene'] for r in domain_rows})
        per_scene = {s: summarize_buckets({b: [r for r in domain_rows if r['scene'] == s and r['bucket'] == b]
                                          for b in ('clean', *map(str, RATIOS))}) for s in scenes}
        buckets = {}
        for bucket in ('clean', *map(str, RATIOS)):
            buckets[bucket] = {k: float(np.mean([per_scene[s][bucket][k] for s in scenes
                                               if per_scene[s][bucket][k] is not None]))
                               if any(per_scene[s][bucket][k] is not None for s in scenes) else None
                               for k in per_scene[scenes[0]][bucket]}
            for key in ('cases', 'homography_failures'):
                buckets[bucket][key] = sum(per_scene[s][bucket][key] for s in scenes)
        domains[dataset] = dict(buckets=buckets, scene_count=len(scenes),
                                score=float(np.mean([buckets[str(r)]['h_auc_5'] for r in RATIOS])))
    combined = {b: {k: (.4*domains['coco']['buckets'][b][k] + .6*domains['megadepth']['buckets'][b][k])
                       if domains['coco']['buckets'][b][k] is not None and domains['megadepth']['buckets'][b][k] is not None else None
                    for k in domains['coco']['buckets'][b]} for b in ('clean', *map(str, RATIOS))}
    for bucket in combined:
        for key in ('cases', 'homography_failures'):
            combined[bucket][key] = sum(domains[d]['buckets'][bucket][key] for d in domains)
    return dict(score=.4*domains['coco']['score']+.6*domains['megadepth']['score'], buckets=combined,
                domains=domains, legacy_coco_score=float(np.mean([domains['coco']['buckets'][str(r)]['h_auc_5']
                                                                  for r in (1, 4, 16, 64, 100)])),
                coco_clean_score=domains['coco']['buckets']['clean']['h_auc_5'])


@torch.no_grad()
def evaluate(config, model, output, *, images=256):
    if model.training:
        raise ValueError('V2 validation requires eval mode')
    manifest, digest = read_manifest(config)
    identity = read_cache_identity(config, digest)
    indices = validation_indices(manifest, images)
    rows, device = [], next(model.parameters()).device
    for index in indices:
        image = manifest['images'][index]
        for case in image['cases']:
            pair = torch.load(cache_path(config['validation_cache'], index, case['bucket']), map_location='cpu', weights_only=True)
            valid = pair['valid_pixels'][:, 0].to(device)
            out = model(pair['packed'].to(device))
            features = [{k: v.cpu().numpy() for k, v in f.items()}
                        for f in extract_features(out['logits'], out['descriptors'], valid=valid)]
            metrics = pair_metrics(*features, pair['homography'].numpy(), *pair['valid_pixels'][:, 0].numpy())
            rows.append(dict(image_position=index, source=image['image'], dataset=image['dataset'], scene=image['scene'],
                             bucket=case['bucket'], ratio_a=case['ratio_a'], ratio_b=case['ratio_b'],
                             **{k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}))
    summary = dict(protocol=VAL_PROTOCOL, manifest_sha256=digest, cache_identity_sha256=identity_digest(identity),
                   full_protocol=config['mode'] == 'formal_v2' and len(indices) == len(manifest['images']),
                   cases=len(rows), sources=len(indices), **summarize(rows))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/'per_pair.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def update_best(best, geometry):
    eligible = (geometry['legacy_coco_score'] >= best['reference_coco_score']-.01
                and geometry['coco_clean_score'] >= best['reference_coco_clean']-.01)
    geometry['selection_eligible'] = eligible
    improved = geometry['full_protocol'] and eligible and geometry['score'] > best['geometry']
    if improved:
        best['geometry'] = geometry['score']
    return improved
