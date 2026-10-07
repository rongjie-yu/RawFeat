"""Strict run identity with an explicit, audited same-model GPU mapping migration."""
from __future__ import annotations

import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import torch

from .validation import file_digest, identity_digest

SCHEMA = 'rawfeat.staged.v1'
ROOT = Path(__file__).resolve().parents[1]
GPU_RESUME_REVISION = ROOT / 'rawfeat/gpu_resume_revision.json'


def run_identity(config: dict, train_hash: str, preflight: dict) -> dict:
    code = {str(p.relative_to(ROOT)): file_digest(p) for p in sorted((ROOT / 'rawfeat').glob('*.py'))}
    dependencies = {name: version(name) for name in ('torch', 'numpy', 'opencv-python-headless', 'albumentations',
                    'kornia', 'PyYAML', 'tensorboard', 'scipy', 'colour-demosaicing', 'thop', 'matplotlib')}
    assets = {name: file_digest(path) for name, path in {
        'teacher': Path(config['superpoint_repo']) / 'superpoint_v1.pth',
        'invisp': Path(config['invisp_repo']) / 'pretrained/canon.pth',
        'calibration': Path(config['eld_calibration']),
        'teacher_source': Path(config['superpoint_repo'])/'demo_superpoint.py',
        'invisp_model_source': Path(config['invisp_repo'])/'model/model.py',
        'invisp_modules_source': Path(config['invisp_repo'])/'model/modules.py',
        'baseline': Path(config['baseline_summary']),
    }.items()}
    return {'schema': SCHEMA, 'config': config, 'code_sha256': code, 'dependencies': dependencies,
            'environment_files': {name: file_digest(ROOT/name) for name in ('requirements.txt', 'environment.yml')},
            'python': sys.version, 'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'precision': 'FP32',
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'train_paths_sha256': train_hash, 'assets_sha256': assets, 'fixed_protocol': preflight}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_schema(state: dict) -> None:
    if state.get('schema') != SCHEMA:
        raise RuntimeError('checkpoint must use the current staged schema; historical states are evidence only')
    if state.get('identity_sha256') != identity_digest(state['identity']):
        raise RuntimeError('checkpoint identity digest does not match its contents')


def identity_differences(saved: dict, current: dict, prefix: str = '') -> list[str]:
    differences = []
    for key in sorted(saved.keys() | current.keys()):
        path = f'{prefix}.{key}' if prefix else key
        if key not in saved or key not in current:
            differences.append(path)
        elif isinstance(saved[key], dict) and isinstance(current[key], dict):
            differences.extend(identity_differences(saved[key], current[key], path))
        elif saved[key] != current[key]:
            differences.append(path)
    return differences


def check_resume_identity(saved: dict, current: dict, *, allow_gpu_change: bool = False) -> list[str]:
    differences = identity_differences(saved, current)
    allowed = set()
    if allow_gpu_change:
        allowed.add('cuda_visible_devices')
        if saved.get('code_sha256') != current.get('code_sha256'):
            revision = json.loads(GPU_RESUME_REVISION.read_text())
            if (saved.get('code_sha256') == revision['source_code_sha256']
                    and current.get('code_sha256') == revision['target_code_sha256']):
                allowed.update(path for path in differences if path.startswith('code_sha256.'))
    rejected = set(differences) - allowed
    if rejected:
        raise RuntimeError('checkpoint identity differs: ' + ', '.join(sorted(rejected)))
    return differences
