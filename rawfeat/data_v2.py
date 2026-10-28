"""Scene splits, immutable source manifests and cursor-based 6:4 sampling."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .data import image_paths
from .validation import file_digest, manifest_digest, read_manifest as read_coco_manifest

SOURCE_PROTOCOL = 'rawfeat.sources.v2'
VAL_PROTOCOL = 'rawfeat.synthetic.validation.v2'
RATIOS = (1, 4, 16, 64, 100, 150, 225)


def write_manifest(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError(f'manifest already exists with different contents: {path}')
    path.write_text(json.dumps(value, indent=2) + '\n')


def extra_cases(position, count, seed):
    rng = np.random.default_rng(np.random.SeedSequence([2027, position, seed]))
    cases = []
    for bucket in ('clean', *map(str, RATIOS)):
        r = 1. if bucket == 'clean' else float(bucket)
        ratios = (r, r) if position < count/2 else (1., r) if position < count*3/4 else (r, 1.)
        a, b = rng.integers(0, 2**31-1, size=2).tolist()
        cases.append(dict(bucket=bucket, ratio_a=ratios[0], ratio_b=ratios[1],
                          noise_a_seed=a, noise_b_seed=b, clean=bucket == 'clean'))
    return cases


def prepare_sources(coco_root, megadepth_root, output, *, smoke=False,
                    split_path='manifests/megadepth_scene_split_v2.json',
                    coco_manifest='manifests/coco_val2017_seed2027.json'):
    coco_root, md_root = Path(coco_root).resolve(), Path(megadepth_root).resolve()
    splits = json.loads(Path(split_path).read_text())
    allowed = set(splits['train'])
    if allowed & (set(splits['valid']) | set(splits['test'])):
        raise ValueError('MegaDepth train and reserved scenes overlap')
    scenes, seen, rejected = {}, set(), []
    for path in sorted(md_root.rglob('*.jpg')):
        if path.parent.name != 'imgs' or not path.parent.parent.name.startswith('dense'):
            continue
        scene = path.parent.parent.parent.name
        if scene not in allowed:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append(str(path.relative_to(md_root)))
            continue
        digest = file_digest(path)
        if digest in seen:
            continue
        seen.add(digest)
        stat = path.stat()
        scenes.setdefault(scene, []).append(dict(image=str(path.relative_to(md_root)), scene=scene,
                                                width=image.shape[1], height=image.shape[0], sha256=digest,
                                                size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    nscenes, per_scene = (1, 2) if smoke else (10, 20)
    eligible = sorted(s for s, rows in scenes.items() if len(rows) >= per_scene)
    if len(eligible) <= nscenes:
        raise ValueError(f'need at least {nscenes+1} clean scenes with {per_scene} valid JPGs; got {len(eligible)}')
    rng = np.random.default_rng(42)
    val_scenes = sorted(rng.choice(eligible, nscenes, replace=False).tolist())
    val = []
    for scene in val_scenes:
        indices = rng.choice(len(scenes[scene]), per_scene, replace=False)
        val.extend(scenes[scene][int(i)] for i in indices)
    train = [row for s in sorted(scenes) if s not in val_scenes for row in scenes[s]]
    coco = image_paths(coco_root, 'train2017')
    if len(coco) != 118287:
        raise ValueError('V2 requires all 118287 COCO train2017 sources')
    sources = dict(protocol=SOURCE_PROTOCOL, mode='smoke' if smoke else 'formal', seed=42,
                   roots=dict(coco=str(coco_root), megadepth=str(md_root)),
                   split_sha256=file_digest(split_path), split_source=splits['source'],
                   coco_train=[dict(image=str(p.relative_to(coco_root)), size_bytes=p.stat().st_size,
                                    mtime_ns=p.stat().st_mtime_ns) for p in coco],
                   megadepth_train=train, megadepth_validation=val, validation_scenes=val_scenes,
                   rejected_images=rejected)
    output = Path(output)
    legacy, legacy_digest = read_coco_manifest(coco_manifest)
    images = []
    for position, row in enumerate(legacy['images'][:2 if smoke else 256]):
        extension = extra_cases(position, 256, 0x434F)
        images.append({**row, 'dataset': 'coco', 'scene': 'coco',
                       'cases': row['cases'] + extension[-2:]})
    for position, row in enumerate(val):
        seeds = np.random.default_rng(np.random.SeedSequence([2027, position, 0x4D44])).integers(0, 2**31-1, 5).tolist()
        images.append(dict(image=row['image'], dataset='megadepth', scene=row['scene'],
                           **dict(zip(('crop_seed', 'appearance_a_seed', 'appearance_b_seed', 'geometry_seed', 'base_seed'), seeds)),
                           cases=extra_cases(position, len(val), 0x4D44)))
    manifest = dict(protocol=VAL_PROTOCOL, seed=2027, mode=sources['mode'], roots=sources['roots'],
                    source_manifest_sha256=manifest_digest(sources), legacy_manifest_sha256=legacy_digest,
                    image_count=len(images), case_count=8*len(images), images=images)
    write_manifest(output, sources)
    write_manifest(output.with_name('validation.json'), manifest)
    return dict(sources=str(output), validation=str(output.with_name('validation.json')),
                coco_train=len(coco), megadepth_train=len(train), megadepth_validation=len(val),
                training_scenes=len(scenes)-len(val_scenes), validation_scenes=val_scenes,
                rejected_images=len(rejected))


def source_pool(config):
    path = Path(config['source_manifest'])
    sources = json.loads(path.read_text())
    if sources['protocol'] != SOURCE_PROTOCOL or sources['mode'] != ('smoke' if config['mode'] == 'smoke_v2' else 'formal'):
        raise ValueError('V2 source manifest protocol/mode differs from configuration')
    roots = {k: Path(v) for k, v in sources['roots'].items()}
    if roots['coco'].resolve() != Path(config['coco_root']).resolve():
        raise ValueError('COCO root differs from source manifest')
    coco = [roots['coco']/r['image'] for r in sources['coco_train']]
    for dataset, rows in (('coco', sources['coco_train']),
                          ('megadepth', sources['megadepth_train']+sources['megadepth_validation'])):
        for row in rows:
            p = roots[dataset]/row['image']
            stat = p.stat()
            if (stat.st_size, stat.st_mtime_ns) != (row['size_bytes'], row['mtime_ns']):
                raise RuntimeError(f'source changed since manifest preparation: {p}')
    md, metadata = {}, {}
    for p in coco:
        metadata[str(p)] = dict(dataset='coco', scene='coco')
    val_scenes = set(sources['validation_scenes'])
    split_path = Path(__file__).resolve().parents[1]/'manifests/megadepth_scene_split_v2.json'
    splits = json.loads(split_path.read_text())
    if sources['split_sha256'] != file_digest(split_path) or not val_scenes <= set(splits['train']):
        raise ValueError('V2 MegaDepth split identity differs')
    val_hashes = {r['sha256'] for r in sources['megadepth_validation']}
    for row in sources['megadepth_train']:
        if row['scene'] in val_scenes or row['scene'] not in splits['train'] or row['sha256'] in val_hashes:
            raise ValueError('MegaDepth training and validation scenes overlap')
        p = roots['megadepth']/row['image']
        md.setdefault(row['scene'], []).append(p)
        metadata[str(p)] = dict(dataset='megadepth', scene=row['scene'])
    if len(coco) != 118287 or len(set(coco)) != len(coco) or not md or any(not Path(p).is_file() for p in metadata):
        raise ValueError('V2 source files missing or COCO/MegaDepth source counts invalid')
    return dict(coco=coco, megadepth=dict(sorted(md.items())), metadata=metadata,
                manifest_sha256=manifest_digest(sources), sources=sources)


def source_dataset(seed, sample_number):
    block, slot = divmod(sample_number, 80)
    update, within = divmod(slot, 16)
    micro, local = divmod(within, 4)
    rng = np.random.default_rng(np.random.SeedSequence([seed, block, 0x4D4958]))
    counts = rng.permutation([10, 9, 10, 9, 10])
    rng = np.random.default_rng(np.random.SeedSequence([seed, block, update, 0x4D4958]))
    per_micro = rng.permutation([3, 3, 2, 2] if counts[update] == 10 else [3, 2, 2, 2])
    rng = np.random.default_rng(np.random.SeedSequence([seed, block, update, micro, 0x4D4958]))
    tags = rng.permutation(['megadepth']*int(per_micro[micro]) + ['coco']*(4-int(per_micro[micro])))
    return str(tags[local])


def sample_source(pool, seed, sample_number):
    dataset = source_dataset(seed, sample_number)
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_number, 0x535243]))
    if dataset == 'coco':
        images = pool['coco']
    else:
        scenes = list(pool['megadepth'])
        images = pool['megadepth'][scenes[int(rng.integers(len(scenes)))]]
    return str(images[int(rng.integers(len(images)))])
