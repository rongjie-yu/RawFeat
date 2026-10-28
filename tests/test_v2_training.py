"""V2 data, schedule, initialization and selection contracts."""
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from rawfeat.config import load_config
from rawfeat.data_v2 import SOURCE_PROTOCOL, sample_source, source_dataset, source_pool
from rawfeat.validation import file_digest
from rawfeat.model import RawFeatureExtractor
from rawfeat.schedules import learning_rate, matching_coefficient, phase, ratio_limit, sample_ratio
from rawfeat.training import initialize_v2, make_optimizer, checkpoint_payload, save_checkpoint, load_checkpoint, training_request
from rawfeat.validation_v2 import summarize, update_best


def test_v2_joint_curriculum_and_learning_rate_boundaries():
    c = load_config('configs/formal_v2.yaml')['schedule']
    for step in (0, 999, 5499, 9999, 29999):
        assert phase(step, c) == 'C' and matching_coefficient(step, c) == 1
    for step, cap in ((0, 100), (999, 100), (5499, 150), (9999, 225), (29999, 225)):
        assert ratio_limit(step, c) == pytest.approx(cap)
        assert sample_ratio(step, .5, c) == pytest.approx(np.sqrt(cap))
    assert learning_rate(0, c) == pytest.approx(1e-5)
    assert learning_rate(999, c) == pytest.approx(1e-4)
    assert learning_rate(29999, c) == pytest.approx(1e-5)
    assert learning_rate(1000, c) < learning_rate(999, c)


def test_source_quota_every_block_microbatch_and_cursor():
    for block in range(8):
        tags = [source_dataset(42, i) for i in range(block*80, (block+1)*80)]
        assert Counter(tags) == {'megadepth': 48, 'coco': 32}
        assert all(set(tags[i:i+4]) == {'coco', 'megadepth'} for i in range(0, 80, 4))
        assert all(tags[i:i+16].count('megadepth') in (9, 10) for i in range(0, 80, 16))
    pool = {'coco': [Path('coco/a')], 'megadepth': {'small': [Path('md/small/a')],
            'large': [Path(f'md/large/{i}') for i in range(100)]}}
    draws = [sample_source(pool, 42, i) for i in range(4000)]
    small = sum(p.startswith('md/small/') for p in draws)
    large = sum(p.startswith('md/large/') for p in draws)
    assert .45 < small/(small+large) < .55  # Equal scene probability despite 100x image count.
    c = load_config('configs/smoke_v2.yaml')
    request = training_request(pool, 42, 73, 4, c)
    np.random.seed(999); np.random.random(100)
    assert training_request(pool, 42, 73, 5, c) == request  # Both updates use the final225 range.


def test_optimizer_groups_cover_all_parameters_and_resume(tmp_path):
    config = load_config('configs/smoke_v2.yaml')
    model = RawFeatureExtractor()
    opt = make_optimizer(model, config)
    named = dict(model.named_parameters())
    all_ids = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(all_ids) == len(set(all_ids)) == len(named)
    for group in opt.param_groups:
        for name, p in named.items():
            if any(p is q for q in group['params']):
                assert group['lr_scale'] == (.3 if name.startswith('stage') else 1.)
    # Exercise strict state restoration without a fake GPU optimizer run.
    identity = {'test': 'v2'}
    for p in model.parameters():
        opt.state[p] = dict(step=torch.tensor(5.), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.ones_like(p))
    saved = checkpoint_payload(model, opt, config, 5, identity, {'det': None, 'geometry': 0.}, None, 0)
    save_checkpoint(tmp_path/'state.pt', saved)
    target = RawFeatureExtractor(); target_opt = make_optimizer(target, config)
    restored = load_checkpoint(tmp_path/'state.pt', target, target_opt, config, identity)
    assert restored['sample_cursor'] == 80
    assert all(int(target_opt.state[p]['step']) == 5 for p in target.parameters())


def test_weight_initialization_preserves_bn_and_does_not_import_optimizer(tmp_path):
    config = copy.deepcopy(load_config('configs/smoke_v2.yaml'))
    original = RawFeatureExtractor()
    state = checkpoint_payload(original, make_optimizer(original, config), config, 96000,
                               {'test': 'initial'}, {'det': None, 'geometry': None}, None, 0)
    path = tmp_path/'initial.pt'; save_checkpoint(path, state)
    config['initial_checkpoint'] = str(path)
    model = RawFeatureExtractor(); initialize_v2(model, config)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.state_dict().items())
    assert not make_optimizer(model, config).state


def test_scene_weighted_score_selection_and_low_noise_guard():
    rows = []
    # Deliberately unequal scene sizes: their contributions must remain equal.
    for dataset, scene, count, quality in [('coco', 'coco', 3, .7), ('megadepth', 'a', 1, .2), ('megadepth', 'b', 9, .8)]:
        for bucket in ('clean', '1', '4', '16', '64', '100', '150', '225'):
            for _ in range(count):
                rows.append(dict(dataset=dataset, scene=scene, bucket=bucket, h_auc_1=quality,
                                 h_auc_3=quality, h_auc_5=quality, repeatability=quality, match_precision=quality,
                                 corner_error=1., localization_error=1., correct_matches=10, detected_a=20,
                                 detected_b=20, matches=10))
    result = summarize(rows)
    assert result['score'] == pytest.approx(.4*.7+.6*.5)
    best = dict(geometry=.5, reference_coco_score=.7, reference_coco_clean=.7)
    result['full_protocol'] = True
    assert update_best(best, result)
    assert best['geometry'] == pytest.approx(.58)
    result.update(score=.9, legacy_coco_score=.68)
    assert not update_best(best, result)
    result.update(legacy_coco_score=.7, full_protocol=False)
    assert not update_best(best, result)


def test_changed_source_rejected_before_training(tmp_path):
    image = tmp_path/'a.jpg'; image.write_bytes(b'original')
    config = load_config('configs/smoke_v2.yaml')
    config.update(coco_root=str(tmp_path), source_manifest=str(tmp_path/'sources.json'))
    sources = dict(protocol=SOURCE_PROTOCOL, mode='smoke', roots={'coco': str(tmp_path), 'megadepth': str(tmp_path)},
                   coco_train=[dict(image='a.jpg', size_bytes=8, mtime_ns=image.stat().st_mtime_ns)],
                   megadepth_train=[], megadepth_validation=[])
    Path(config['source_manifest']).write_text(json.dumps(sources))
    image.write_bytes(b'changed source')
    with pytest.raises(RuntimeError, match='source changed'):
        source_pool(config)


def test_scene_leakage_rejected_before_training(tmp_path):
    image = tmp_path/'a.jpg'; image.write_bytes(b'unchanged')
    split_path = Path('manifests/megadepth_scene_split_v2.json')
    scene = json.loads(split_path.read_text())['train'][0]
    row = dict(image='a.jpg', scene=scene, size_bytes=image.stat().st_size,
               mtime_ns=image.stat().st_mtime_ns, sha256=file_digest(image))
    config = load_config('configs/smoke_v2.yaml')
    config.update(coco_root=str(tmp_path), source_manifest=str(tmp_path/'sources.json'))
    sources = dict(protocol=SOURCE_PROTOCOL, mode='smoke', roots={'coco': str(tmp_path), 'megadepth': str(tmp_path)},
                   coco_train=[row], megadepth_train=[row], megadepth_validation=[row],
                   validation_scenes=[scene], split_sha256=file_digest(split_path))
    Path(config['source_manifest']).write_text(json.dumps(sources))
    with pytest.raises(ValueError, match='overlap'):
        source_pool(config)
