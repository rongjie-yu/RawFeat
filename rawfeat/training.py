"""One FP32 staged trainer, absolute data cursor, and complete transactional recovery."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import load_config
from .correspondences import sample_correspondences
from .data import PairGenerator, PairRequest, image_paths
from .features import extract_features
from .identity import (SCHEMA, check_resume_identity, check_schema, identity_differences,
                       run_identity, write_json)
from .losses import pair_losses
from .model import RawFeatureExtractor
from .noise import CanonELD
from .runtime import configure_fp32
from .schedules import matching_coefficient, sample_ratio, update_schedule
from .sensor import load_invisp
from .teacher import load_teacher
from .validation import (BASELINE_PREPROCESSING, baseline_code_digest, cache_identity, cache_path,
                         evaluate, file_digest, identity_digest, read_cache_identity, read_manifest)


def training_paths(config: dict) -> list[Path]:
    paths = image_paths(config['coco_root'], 'train2017')
    if len(paths) != 118287:
        raise RuntimeError('training requires all 118287 COCO train2017 sources')
    return paths


def paths_hash(paths: list[Path]) -> str:
    return hashlib.sha256('\n'.join(path.name for path in paths).encode()).hexdigest()


def formal_preflight(config: dict) -> dict:
    """Read-only asset/data/GPU check, also used by the compressed smoke run."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run nvidia-smi and check the selected Python environment')
    device = torch.device(config['device'])
    torch.cuda.set_device(device)
    configure_fp32()
    manifest, digest = read_manifest(config['validation_manifest'])
    cached = read_cache_identity(config['validation_cache'], digest)
    if cached != cache_identity(digest, config['invisp_repo'], config['eld_calibration']):
        raise RuntimeError('assets differ from the fixed validation cache')
    missing = sum(not cache_path(config['validation_cache'], index, case['bucket']).is_file()
                  for index, image in enumerate(manifest['images']) for case in image['cases'])
    if missing:
        raise FileNotFoundError(f'fixed validation cache missing {missing} of 1536 cases')
    baseline = json.loads(Path(config['baseline_summary']).read_text())
    if (not baseline['full_protocol'] or baseline['manifest_sha256'] != digest
            or baseline['preprocessing'] != BASELINE_PREPROCESSING
            or baseline.get('cache_identity_sha256') != identity_digest(cached)
            or baseline.get('baseline_code_sha256') != baseline_code_digest()
            or baseline.get('superpoint_sha256') != file_digest(Path(config['superpoint_repo'])/'superpoint_v1.pth')):
        raise RuntimeError('full MeanAD+SuperPoint baseline identity differs')
    paths = training_paths(config)
    if {p.name for p in paths} & {Path(image['image']).name for image in manifest['images']}:
        raise RuntimeError('training and fixed validation sources overlap')
    free, total = torch.cuda.mem_get_info(device)
    return {'ready': True, 'cached_cases': 1536, 'manifest_sha256': digest,
            'cache_identity_sha256': identity_digest(cached), 'baseline_score': baseline['score'],
            'train_sources': len(paths), 'train_paths_sha256': paths_hash(paths),
            'gpu': torch.cuda.get_device_name(device), 'free_memory_bytes': free, 'total_memory_bytes': total,
            'torch': torch.__version__, 'cuda': torch.version.cuda}


def make_optimizer(model: RawFeatureExtractor, config: dict) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            (decay if isinstance(module, torch.nn.Conv2d) and name == 'weight' else no_decay).append(parameter)
    assert len(decay + no_decay) == len(list(model.parameters()))
    assert {id(p) for p in decay + no_decay} == {id(p) for p in model.parameters()}
    return torch.optim.AdamW([{'params': decay, 'weight_decay': config['weight_decay']},
                              {'params': no_decay, 'weight_decay': 0.}],
                             lr=config['schedule']['lr_initial'], betas=(.9, .999), eps=1e-8)


def set_training_mode(model: RawFeatureExtractor, config: dict, step: int = 0) -> None:
    model.train()
    active = matching_coefficient(step, config['schedule']) > 0
    model.descriptor.train(active)
    model.descriptor.requires_grad_(active)
    if not active:
        for p in model.descriptor.parameters():
            p.grad = None


def training_request(paths: list[Path], seed: int, sample_number: int, step: int, config: dict) -> tuple[PairRequest, int]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_number]))
    image = str(paths[int(rng.integers(len(paths)))])
    values = rng.integers(0, 2**31 - 1, size=7).tolist()
    ratio_a = sample_ratio(step, float(rng.random()), config['schedule'])
    ratio_b = sample_ratio(step, float(rng.random()), config['schedule'])
    return PairRequest(image, *values[:6], ratio_a, ratio_b), values[6]


def capture_rng() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])


def checkpoint_payload(model, optimizer, config, step, identity, best, validation, clipped_count) -> dict:
    # A small GPU state copy is retained after each successful step. It excludes inputs,
    # survives a partial BN update/OOM, and avoids per-update disk checkpoints.
    return {'schema': SCHEMA, 'student': copy.deepcopy(model.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict()), 'config': config, 'step': step,
            'phase': update_schedule(max(0, step-1), config['schedule'])['phase'],
            'sample_cursor': step * 16, 'scheduler': update_schedule(step, config['schedule']),
            'rng': capture_rng(), 'identity': identity, 'identity_sha256': identity_digest(identity),
            'best': best.copy(), 'validation': validation, 'clipped_count': clipped_count}


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, config, identity, *, allow_gpu_change=False) -> dict:
    state = torch.load(path, map_location='cpu', weights_only=False)
    check_schema(state)
    if state['config'] != config:
        raise RuntimeError('checkpoint configuration identity differs')
    check_resume_identity(state['identity'], identity, allow_gpu_change=allow_gpu_change)
    if allow_gpu_change and (len(state['rng']['torch_cuda']) != 1 or torch.cuda.device_count() != 1):
        raise RuntimeError('GPU migration requires one saved and one visible CUDA device')
    step = state['step']
    if not isinstance(step, int) or not 0 <= step <= config['schedule']['updates']:
        raise RuntimeError('checkpoint step outside configured budget')
    if state['sample_cursor'] != step * 16 or state['scheduler'] != update_schedule(step, config['schedule']):
        raise RuntimeError('checkpoint cursor or next schedule disagrees with completed updates')
    model.load_state_dict(state['student'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    for name, p in model.named_parameters():
        expected = max(0, step-config['schedule']['detection_updates']) if name.startswith('descriptor.') else step
        actual = int(optimizer.state[p]['step']) if optimizer.state[p] else 0
        if actual != expected:
            raise RuntimeError(f'per-parameter Adam step differs: {name}: {actual} != {expected}')
    restore_rng(state['rng'])
    return state


def point_pairs(teacher_features: list[dict], pair: dict, seed: int, device: torch.device):
    valid = pair['valid_pixels'][:, 0].cpu().numpy()
    a, b, counts = sample_correspondences(teacher_features[0]['points'].cpu().numpy(),
                                        teacher_features[1]['points'].cpu().numpy(),
                                        pair['homography'].cpu().numpy(), valid[0], valid[1],
                                        np.random.default_rng(seed))
    return (torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)), counts


def make_microbatch(generator, teacher, paths, config, step, cursor, device) -> tuple[dict, dict]:
    active = matching_coefficient(step, config['schedule']) > 0
    pairs, logits_list, matches, counts, ratios = [], [], [], [], []
    for slot in range(config['batch_pairs']):
        request, seed = training_request(paths, config['seed'], cursor+slot, step, config)
        pair = generator.generate(request)
        with torch.no_grad():
            logits, grid = teacher(pair['srgb_gray'])
            if active:
                features = extract_features(logits, grid, valid=pair['valid_pixels'][:, 0])
                points, count = point_pairs(features, pair, seed, device)
                matches.append(points)
                counts.append(count)
        pairs.append(pair)
        logits_list.append(logits)
        ratios.extend(pair['ratios'])
    batch = {name: torch.cat([p[name] for p in pairs]) for name in ('packed', 'clean_gray', 'valid_cells', 'valid_pixels')}
    batch.update(teacher_logits=torch.cat(logits_list), matches=matches if active else None)
    measured = {'ratios': ratios, 'correspondences': sum(len(a) for a, _ in matches) if active else None,
                'teacher_points': sum(c['teacher_a'] + c['teacher_b'] for c in counts) if active else None,
                'uniform_points': sum(c['uniform'] for c in counts) if active else None,
                'valid_cells': int(batch['valid_cells'].sum()), 'valid_pixels': int(batch['valid_pixels'].sum())}
    return batch, measured


def microbatch_digest(batch: dict) -> str:
    digest = hashlib.sha256()
    tensors = [batch[k] for k in ('packed', 'teacher_logits', 'clean_gray', 'valid_cells', 'valid_pixels')]
    if batch['matches'] is not None:
        tensors.extend(p for pair in batch['matches'] for p in pair)
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_digest(values) -> str:
    digest = hashlib.sha256()
    for name, tensor in values:
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_audit(model, optimizer, state) -> dict:
    adam = {name: int(optimizer.state[p]['step']) if optimizer.state[p] else 0 for name, p in model.named_parameters()}
    optimizer_tensors = [(name+'/'+key, value) for name, p in model.named_parameters()
                         for key, value in optimizer.state[p].items() if isinstance(value, torch.Tensor)]
    rng = capture_rng()
    return {'step': state['step'], 'sample_cursor': state['sample_cursor'], 'scheduler': state['scheduler'],
            'identity_sha256': state['identity_sha256'], 'adam_steps': adam,
            'student_sha256': tensor_digest(model.state_dict().items()),
            'descriptor_sha256': tensor_digest(model.descriptor.state_dict().items()),
            'optimizer_tensors_sha256': tensor_digest(optimizer_tensors),
            'torch_rng_sha256': tensor_digest([('cpu', rng['torch_cpu']), *[(f'cuda{i}', v) for i, v in enumerate(rng['torch_cuda'])]]),
            'python_numpy_rng_sha256': hashlib.sha256(repr((rng['python'], rng['numpy'])).encode()).hexdigest()}


def _link_checkpoint(directory: Path, name: str, checkpoint: Path) -> None:
    temporary = directory/f'.{name}.tmp'
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint.name)
    os.replace(temporary, directory/name)


def check_training_output(output: Path, resume) -> None:
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('training output already contains a run; choose a new directory or --resume')
    elif Path(resume).resolve().parent != output.resolve():
        raise RuntimeError('resume must use the same run/config/output directory')


def reconcile_logs(output: Path, step: int) -> None:
    """Separate updates beyond the recovered state; keep the original record as evidence."""
    path = output/'scalars.jsonl'
    if not path.exists():
        if step:
            raise RuntimeError('resume run has no scalar history')
        return
    lines = path.read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    kept = [r for r in rows if r['step'] <= step]
    if [r['step'] for r in kept] != list(range(1, step+1)):
        raise RuntimeError('scalar history missing/duplicate completed updates')
    if len(kept) != len(rows):
        archive = output/f'scalars_before_recovery_{time.time_ns()}.jsonl'
        archive.write_text('\n'.join(lines)+'\n')
        path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in kept))


def update_best(best: dict, phase: str, detection: dict, geometry: dict | None) -> tuple[bool, bool]:
    new_det = (phase == 'A' and detection['full_protocol'] and detection['teacher_recall_1px'] is not None
               and (best['det'] is None or detection['teacher_recall_1px'] > best['det']))
    new_geometry = (phase == 'C' and geometry is not None and geometry['full_protocol']
                    and (best['geometry'] is None or geometry['score'] > best['geometry']))
    if new_det: best['det'] = detection['teacher_recall_1px']
    if new_geometry: best['geometry'] = geometry['score']
    return new_det, new_geometry


def retain_checkpoints(output: Path, boundaries: set[int]) -> None:
    protected = {p.resolve() for name in ('latest.pt', 'best_det.pt', 'best_geometry.pt') if (p := output/name).exists()}
    ordinary = [p for p in sorted(output.glob('checkpoint_step_*.pt'))
                if int(p.stem.split('_')[-1]) not in boundaries]
    for p in ordinary[:-3]:
        if p.resolve() not in protected:
            p.unlink()


def train(config_path, *, resume=None, max_updates=None, output_root=None, allow_gpu_change=False) -> dict:
    from .analysis import analyze_run
    from .diagnostics import diagnose_model, gradient_probe
    if allow_gpu_change and resume is None:
        raise ValueError('--allow-gpu-change requires --resume')
    config = load_config(config_path)
    if output_root is not None:
        config['output'] = str(output_root)
    output = Path(config['output'])
    check_training_output(output, resume)
    stop = config['schedule']['updates'] if max_updates is None else max_updates
    if not 1 <= stop <= config['schedule']['updates']:
        raise ValueError('max-updates is an absolute stop inside the unchanged total budget')
    preflight = formal_preflight(config)
    device = torch.device(config['device'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    paths = training_paths(config)
    # Free memory changes between resumes; it belongs to preflight evidence, not identity.
    fixed = {k: v for k, v in preflight.items() if k not in ('free_memory_bytes', 'total_memory_bytes')}
    identity = run_identity(config, paths_hash(paths), fixed)
    student = RawFeatureExtractor().to(device)
    optimizer = make_optimizer(student, config)
    teacher = load_teacher(config['superpoint_repo'], device)
    generator = PairGenerator(load_invisp(config['invisp_repo'], device), CanonELD(config['eld_calibration']), device)
    step, clipped_count, best, validation = 0, 0, {'det': None, 'geometry': None}, None
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        state = load_checkpoint(resume, student, optimizer, config, identity, allow_gpu_change=allow_gpu_change)
        step, clipped_count, best, validation = state['step'], state['clipped_count'], state['best'], state['validation']
        manifest = json.loads((output/'run_manifest.json').read_text())
        if manifest['identity_sha256'] != identity_digest(manifest['identity']):
            raise RuntimeError('run manifest identity digest differs')
        manifest_changes = check_resume_identity(manifest['identity'], identity, allow_gpu_change=allow_gpu_change)
        reconcile_logs(output, step)
        write_json(output/f'resume_identity_{step:06d}_{time.time_ns()}.json', {
            'step': step, 'checkpoint': str(Path(resume).resolve()), 'allow_gpu_change': allow_gpu_change,
            'checkpoint_identity': state['identity'], 'checkpoint_identity_sha256': state['identity_sha256'],
            'current_identity': identity, 'current_identity_sha256': identity_digest(identity),
            'checkpoint_changes': identity_differences(state['identity'], identity),
            'manifest_identity_sha256': manifest['identity_sha256'], 'manifest_changes': manifest_changes,
            'resumed_utc': datetime.now(timezone.utc).isoformat()})
    else:
        write_json(output/'run_manifest.json', {'identity': identity, 'identity_sha256': identity_digest(identity),
                   'command': sys.argv, 'started_utc': datetime.now(timezone.utc).isoformat(), 'preflight': preflight})
        write_json(output/'resolved_config.json', config)
        (output/'training_images.txt').write_text(''.join(p.name+'\n' for p in paths))
    if stop < step:
        raise ValueError('absolute stop precedes restored update count')
    set_training_mode(student, config, step)
    reliable = checkpoint_payload(student, optimizer, config, step, identity, best, validation, clipped_count)
    initial_descriptor = copy.deepcopy(student.descriptor.state_dict()) if step <= config['schedule']['detection_updates'] else None
    boundaries = {config['schedule']['detection_updates'], config['schedule']['detection_updates']+config['schedule']['matching_ramp'], config['schedule']['updates']}
    writer = SummaryWriter(output/'tensorboard', purge_step=step+1 if resume else None)
    scalar_log = (output/'scalars.jsonl').open('a')
    cursor = step*16
    checkpoint_seconds, validation_seconds = [], []
    successful_step = step
    update_journal = (output/'optimizer_updates.jsonl').open('a')
    write_json(output/f'startup_{step:06d}_{time.time_ns()}.json', state_audit(student, optimizer, reliable))
    try:
        for step_index in range(step, stop):
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            schedule = update_schedule(step_index, config['schedule'])
            completed = step_index+1
            set_training_mode(student, config, step_index)
            for group in optimizer.param_groups:
                group['lr'] = schedule['lr']
            optimizer.zero_grad(set_to_none=True)
            totals = {name: None if name == 'match' and schedule['matching_weight'] == 0 else 0.
                      for name in ('det', 'occupancy', 'position', 'gray', 'match', 'total', 'teacher_mass', 'position_scale')}
            counts = {k: None if k in ('correspondences', 'teacher_points', 'uniform_points') and schedule['matching_weight'] == 0 else 0
                      for k in ('correspondences', 'teacher_points', 'uniform_points', 'valid_cells', 'valid_pixels')}
            ratios, input_digests = [], []
            audit = completed == 1 or completed in boundaries or step_index == step or step_index in boundaries
            probe = completed == 1 or completed in boundaries or completed % config['probe_interval'] == 0
            batches = []
            for micro in range(4):
                cursor = step_index*16 + micro*4
                batch, observed = make_microbatch(generator, teacher, paths, config, step_index, cursor, device)
                if audit:
                    input_digests.append(microbatch_digest(batch))
                if probe:
                    batches.append(batch)
                out = student(batch['packed'], compute_descriptors=schedule['matching_weight'] > 0)
                losses = pair_losses(out, batch['teacher_logits'], batch['valid_cells'], batch['clean_gray'],
                                     batch['valid_pixels'], batch['matches'], schedule['matching_weight'])
                if not bool(torch.isfinite(losses['total'])):
                    raise FloatingPointError(f'nonfinite loss at next update {completed}, cursor {cursor}')
                (losses['total']/4).backward()
                for name in totals:
                    if losses[name] is not None:
                        totals[name] += float(losses[name].detach())/4
                ratios.extend(observed['ratios'])
                for name in counts:
                    if observed[name] is not None:
                        counts[name] += observed[name]
            norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 5., error_if_nonfinite=True))
            clipped_count += int(norm > 5)
            if schedule['phase'] == 'A':
                assert all(p.grad is None and not optimizer.state[p] for p in student.descriptor.parameters())
                assert all(torch.equal(v, initial_descriptor[k]) for k, v in student.descriptor.state_dict().items())
            optimizer.step()
            successful_step = completed
            update_journal.write(json.dumps({'step': completed, 'resumed_from': step, 'sample_cursor': completed*16})+'\n')
            update_journal.flush()
            # Commit the completed state before any diagnostics or I/O can fail.
            reliable = checkpoint_payload(student, optimizer, config, completed, identity, best, validation, clipped_count)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-started
            row = {'step': completed, **schedule, 'det_weight': 1., 'match_available': totals['match'] is not None,
                   'sample_cursor': completed*16, 'ratio_min': min(ratios), 'ratio_mean': float(np.mean(ratios)), 'ratio_max': max(ratios),
                   'ratio_counts': {label: sum(low <= r < high for r in ratios) for label, low, high in
                                    (('[1,4)', 1., 4.), ('[4,16)', 4., 16.), ('[16,64)', 16., 64.), ('[64,100]', 64., 100.000001))},
                   **{f'loss_{k}': v for k, v in totals.items() if k not in ('teacher_mass', 'position_scale')},
                   'teacher_mass': totals['teacher_mass'], 'position_scale': totals['position_scale'],
                   'loss_gray_weighted': 10*totals['gray'],
                   'loss_match_weighted': None if totals['match'] is None else schedule['matching_weight']*totals['match'],
                   'gradient_norm_before_clip': norm, 'clip_factor': min(1., 5/(norm+1e-6)),
                   'clipped': norm > 5, 'clipped_count': clipped_count, 'clip_frequency': clipped_count/completed,
                   'update_seconds': seconds, 'pairs_per_second': 16/seconds,
                   'max_memory_bytes': torch.cuda.max_memory_allocated(device), **counts}
            if input_digests:
                row['input_sha256'] = hashlib.sha256(''.join(input_digests).encode()).hexdigest()
            scalar_log.write(json.dumps(row, allow_nan=False)+'\n'); scalar_log.flush()
            if completed == 1 or completed % config['log_interval'] == 0 or completed == stop or completed in boundaries:
                for key, value in row.items():
                    if key != 'step' and isinstance(value, (int, float)):
                        writer.add_scalar(key, value, completed)
                for label, value in row['ratio_counts'].items():
                    writer.add_scalar('ratio_counts/'+label, value, completed)
                writer.add_text('phase', schedule['phase'], completed)
                print(json.dumps(row, allow_nan=False), flush=True)
                writer.flush()
            if probe:
                rng = capture_rng()
                try:
                    probe_report = gradient_probe(student, batches, config, step_index, optimizer)
                    write_json(output/'probes'/f'step_{completed:06d}.json', probe_report)
                finally:
                    restore_rng(rng)
                batches.clear()
            should_validate = completed in boundaries or completed % config['validation_interval'] == 0
            should_diagnose = should_validate or completed % config['diagnostic_interval'] == 0
            if should_diagnose:
                rng = capture_rng(); start_validation = time.perf_counter()
                try:
                    tested = copy.deepcopy(student).eval()
                    images = config['validation_images'] if should_validate else config['diagnostic_images']
                    detection = diagnose_model(tested, teacher, config, output/'diagnostics'/f'step_{completed:06d}',
                                               completed, images=images, geometry=schedule['phase'] != 'A')
                    geometry = None
                    if should_validate and schedule['phase'] != 'A':
                        geometry = evaluate(config['validation_manifest'], config['validation_cache'],
                                            output/'validation'/f'step_{completed:06d}', model=tested,
                                            baseline_summary=config['baseline_summary'], max_images=config['validation_images'])
                    validation = {'step': completed, 'phase': schedule['phase'], 'detection': detection, 'geometry': geometry}
                    write_json(output/'validation'/f'step_{completed:06d}'/'selection.json', validation)
                    new_det, new_geometry = update_best(best, schedule['phase'], detection, geometry)
                    for label, value in [('teacher_recall_1px', detection['teacher_recall_1px']),
                                         ('geometry_score', geometry['score'] if geometry else None)]:
                        if value is not None: writer.add_scalar('validation/'+label, value, completed)
                    del tested
                finally:
                    restore_rng(rng)
                validation_seconds.append({'step': completed, 'seconds': time.perf_counter()-start_validation})
                reliable = checkpoint_payload(student, optimizer, config, completed, identity, best, validation, clipped_count)
                analyze_run(output)
            if completed in boundaries or completed % config['checkpoint_interval'] == 0 or completed == stop:
                checkpoint = output/f'checkpoint_step_{completed:06d}.pt'
                start_save = time.perf_counter(); save_checkpoint(checkpoint, reliable)
                _link_checkpoint(output, 'latest.pt', checkpoint)
                if should_diagnose and new_det: _link_checkpoint(output, 'best_det.pt', checkpoint)
                if should_diagnose and new_geometry: _link_checkpoint(output, 'best_geometry.pt', checkpoint)
                if completed in boundaries:
                    write_json(output/f'boundary_{completed:06d}.json', state_audit(student, optimizer, reliable))
                retain_checkpoints(output, boundaries)
                checkpoint_seconds.append({'step': completed, 'seconds': time.perf_counter()-start_save})
    except BaseException as error:
        # The live model can have partially changed BN/parameters; only committed state is resumable.
        recovery = output/'recovery.pt'
        save_checkpoint(recovery, reliable)
        _link_checkpoint(output, 'latest.pt', recovery)
        write_json(output/f'failure_{time.time_ns()}.json', {'error': repr(error), 'committed_updates': reliable['step'],
                   'failed_sample_cursor': cursor, 'recovery': str(recovery),
                   'successful_optimizer_step': successful_step, 'successful_updates_not_in_recovery': successful_step-reliable['step'],
                   'requests': [asdict(training_request(paths, config['seed'], cursor+slot, cursor//16, config)[0]) for slot in range(4)]})
        raise
    finally:
        scalar_log.close(); update_journal.close(); writer.close()
        overhead_path = output/'overhead.jsonl'
        with overhead_path.open('a') as log:
            log.write(json.dumps({'resumed_from': step, 'committed_updates': reliable['step'],
                                 'validation': validation_seconds, 'checkpoint': checkpoint_seconds})+'\n')
    analyze_run(output)
    return {'completed_updates': reliable['step'], 'new_updates': reliable['step']-step, 'output': str(output), 'best': best}
