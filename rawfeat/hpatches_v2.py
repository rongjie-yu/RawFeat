"""150/225 HPatches extension using immutable V1 clean-image caches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import hpatches as hp
from .identity import check_schema
from .model import RawFeatureExtractor
from .noise import CanonELD
from .teacher import load_teacher


@torch.no_grad()
def evaluate_extra(config, checkpoint, manifest_path, cache_root, output, *, sequence_names=None, max_pairs=None):
    if not torch.cuda.is_available():
        raise RuntimeError('HPatches V2 extension requires CUDA')
    manifest, digest = hp.read_manifest(manifest_path)
    cache_root, output = Path(cache_root), Path(output)
    identity = json.loads((cache_root/'identity.json').read_text())
    if identity != hp.cache_identity(digest, manifest['dataset_integrity_sha256'], config['invisp_repo'], config['eld_calibration']):
        raise RuntimeError('HPatches clean cache identity differs')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'HPatches extension output is not empty: {output}')
    device = torch.device(config['device'])
    torch.cuda.set_device(device)
    stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    check_schema(stored)
    model = RawFeatureExtractor().to(device).eval()
    model.load_state_dict(stored['student'], strict=True)
    teacher = load_teacher(config['superpoint_repo'], device)
    eld = CanonELD(config['eld_calibration'])
    pairs = hp.iter_manifest_pairs(manifest, sequence_names)
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError('max_pairs must be positive')
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError('HPatches extension requires at least one selected pair')
    required = {p[k] for p in pairs for k in ('reference_image_id', 'target_image_id')}
    images = [i for i in hp.iter_manifest_images(manifest, sequence_names) if i['id'] in required]
    positions = {s['name']: s['position'] for s in manifest['sequences']}
    rows, timings = [], []
    for ratio in (150, 225):
        condition = f'ratio{ratio}'
        features = {method: {} for method in ('rawfeat', 'superpoint')}
        for image in images:
            clean = hp.load_cache_entry(hp.cache_path(cache_root, image['id'], 'clean'))['noisy_bayer_dn'].to(device)
            scene = image['id'].rsplit('/', 1)[0]
            seed = int(np.random.SeedSequence([2027, positions[scene], image['index'], ratio, 0x4850]).generate_state(1, dtype=np.uint32)[0])
            noisy = eld.apply(clean, ratio, seed).cpu()
            for method, network in (('rawfeat', model), ('superpoint', teacher)):
                feature, timing = hp._single_image_inference(method, network, noisy, image['id'], condition, device)
                features[method][image['id']] = feature
                timings.append(timing)
        for method in features:
            rows.extend(hp._pair_row(method, pair, condition, features[method]) for pair in pairs)
    sequence_rows = hp._aggregate_sequence_rows(rows)
    summaries = {method: {group: {condition: hp._group_summary(sequence_rows, method, group, condition)
                                  for condition in ('ratio150', 'ratio225')}
                          for group in ('all', 'illumination', 'viewpoint')}
                 for method in ('rawfeat', 'superpoint')}
    differences = []
    for scene in sorted({r['sequence'] for r in sequence_rows}):
        scores = {(r['method'], r['condition']): r['h_auc_5'] for r in sequence_rows if r['sequence'] == scene}
        differences.append(np.mean([scores['rawfeat', c]-scores['superpoint', c] for c in ('ratio150', 'ratio225')]))
    values = np.asarray(differences)
    bootstrap = np.random.default_rng(20261007).choice(values, size=(10000, len(values)), replace=True).mean(1)
    result = dict(protocol='rawfeat.hpatches.extension.v2', full_protocol=len(pairs) == 580,
                  pair_count_per_condition=len(pairs), conditions=['ratio150', 'ratio225'], summaries=summaries,
                  mean_two_high_noise_delta=float(values.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                  manifest_sha256=digest, legacy_cache_identity_sha256=hp.identity_digest(identity),
                  checkpoint_sha256=hp.file_digest(checkpoint), superpoint_sha256=hp.file_digest(teacher.rawfeat_checkpoint_path),
                  extension_code_sha256=hp.file_digest(__file__),
                  note='150/225 extension only. Keep the legacy five-noise HPatches score separate.')
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (('per_pair', rows), ('per_sequence', sequence_rows), ('timings', timings)):
        (output/f'{name}.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in data))
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
