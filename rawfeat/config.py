"""The current configuration schema. Historical experiment configurations are rejected."""
from pathlib import Path
import yaml

TOP_FIELDS = {'mode', 'device', 'seed', 'coco_root', 'invisp_repo', 'superpoint_repo', 'eld_calibration',
              'validation_manifest', 'validation_cache', 'baseline_summary', 'output', 'batch_pairs',
              'accumulation', 'weight_decay', 'gradient_clip', 'log_interval', 'validation_interval',
              'validation_images', 'diagnostic_interval', 'diagnostic_images', 'checkpoint_interval',
              'probe_interval', 'schedule'}
SCHEDULE_FIELDS = {'updates', 'detection_updates', 'matching_ramp', 'lr_initial', 'lr_peak', 'lr_min',
                   'lr_warmup', 'ratio_start', 'ratio_end'}


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != TOP_FIELDS or set(config['schedule']) != SCHEDULE_FIELDS:
        raise ValueError('configuration must use the current staged schema; unknown/missing fields')
    if config['mode'] not in ('formal', 'smoke'):
        raise ValueError('mode must be formal or smoke')
    schedule = config['schedule']
    expected = ((100000, 50000, 5000, 2000, 5000, 40000) if config['mode'] == 'formal'
                else (600, 300, 100, 20, 30, 240))
    keys = ('updates', 'detection_updates', 'matching_ramp', 'lr_warmup', 'ratio_start', 'ratio_end')
    if tuple(schedule[k] for k in keys) != expected:
        raise ValueError('stage, LR and noise boundaries differ from the selected formal/smoke budget')
    if (config['batch_pairs'], config['accumulation'], config['seed'], config['gradient_clip'], config['weight_decay']) != (4, 4, 42, 5., 1e-4):
        raise ValueError('seed42, four pairs x four accumulation, clip5 and AdamW decay1e-4 are required')
    if (schedule['lr_initial'], schedule['lr_peak'], schedule['lr_min']) != (3e-6, 3e-4, 3e-6):
        raise ValueError('LR must remain 3e-6 -> 3e-4 -> 3e-6')
    if config['validation_images'] != (256 if config['mode'] == 'formal' else 32):
        raise ValueError('formal selection requires all 256 fixed sources; smoke uses 32 sources')
    if config['diagnostic_images'] != 16:
        raise ValueError('fixed diagnostics require 16 sources')
    for key in ('log_interval', 'validation_interval', 'diagnostic_interval', 'checkpoint_interval', 'probe_interval'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive interval')
    if config['mode'] == 'formal' and tuple(config[k] for k in ('log_interval', 'validation_interval', 'diagnostic_interval', 'checkpoint_interval', 'probe_interval')) != (20, 2000, 1000, 1000, 5000):
        raise ValueError('formal recording intervals differ from the design')
    return config
