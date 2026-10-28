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


def is_v2(config: dict) -> bool:
    return config.get('mode') in ('formal_v2', 'smoke_v2')


def load_v2(config: dict) -> dict:
    extra = {'source_manifest', 'initial_checkpoint', 'legacy_validation_manifest', 'legacy_validation_cache'}
    if set(config) != TOP_FIELDS | extra or not isinstance(config.get('schedule'), dict):
        raise ValueError('V2 configuration schema differs')
    schedule = config['schedule']
    if set(schedule) != SCHEDULE_FIELDS | {'ratio_initial', 'ratio_max', 'backbone_lr_multiplier'}:
        raise ValueError('V2 configuration schema differs')
    formal = config['mode'] == 'formal_v2'
    keys = ('updates', 'detection_updates', 'matching_ramp', 'lr_warmup', 'ratio_start', 'ratio_end')
    if tuple(schedule[k] for k in keys) != ((30000, 0, 0, 1000, 1000, 10000) if formal else (20, 0, 0, 2, 2, 5)):
        raise ValueError('V2 schedule boundaries differ from the selected formal/smoke budget')
    if tuple(schedule[k] for k in ('lr_initial', 'lr_peak', 'lr_min', 'ratio_initial', 'ratio_max', 'backbone_lr_multiplier')) != (1e-5, 1e-4, 1e-5, 100, 225, .3):
        raise ValueError('V2 requires approved learning rates and log-uniform ratio100 -> 225')
    if tuple(config[k] for k in ('batch_pairs', 'accumulation', 'seed', 'gradient_clip', 'weight_decay')) != (4, 4, 42, 5., 1e-4):
        raise ValueError('V2 requires four pairs x four accumulation, seed42, clip5 and decay1e-4')
    intervals = ('log_interval', 'validation_interval', 'diagnostic_interval', 'checkpoint_interval', 'probe_interval')
    if tuple(config[k] for k in intervals) != ((20, 2000, 1000, 1000, 5000) if formal else (1, 10, 10, 5, 20)):
        raise ValueError('V2 recording intervals differ')
    if (config['validation_images'], config['diagnostic_images']) != ((256, 16) if formal else (2, 2)):
        raise ValueError('V2 validation/diagnostic source counts differ')
    return config


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if isinstance(config, dict) and is_v2(config):
        return load_v2(config)
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
