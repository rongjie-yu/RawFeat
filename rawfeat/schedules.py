"""One schedule; arguments count completed updates, so the next update is k=step+1."""
from __future__ import annotations

import math


def phase(step: int, config: dict) -> str:
    k = step + 1
    if k <= config['detection_updates']:
        return 'A'
    if k <= config['detection_updates'] + config['matching_ramp']:
        return 'B'
    return 'C'


def matching_coefficient(step: int, config: dict) -> float:
    return max(0.0, min(1.0, (step + 1 - config['detection_updates']) / config['matching_ramp']))


def learning_rate(step: int, config: dict) -> float:
    k = step + 1
    if k <= config['lr_warmup']:
        return config['lr_initial'] + (config['lr_peak'] - config['lr_initial']) * (k - 1) / (config['lr_warmup'] - 1)
    if k <= config['detection_updates']:
        return config['lr_peak']
    progress = min(1.0, (k - config['detection_updates']) / (config['updates'] - config['detection_updates']))
    return config['lr_min'] + (config['lr_peak'] - config['lr_min']) * (1 + math.cos(math.pi * progress)) / 2


def ratio_limit(step: int, config: dict) -> float:
    progress = max(0.0, min(1.0, (step + 1 - config['ratio_start']) / (config['ratio_end'] - config['ratio_start'])))
    return math.exp((1 - math.cos(math.pi * progress)) / 2 * math.log(100))


def sample_ratio(step: int, uniform_sample: float, config: dict) -> float:
    return math.exp(uniform_sample * math.log(ratio_limit(step, config)))


def update_schedule(step: int, config: dict) -> dict:
    return {'phase': phase(step, config), 'lr': learning_rate(step, config),
            'matching_weight': matching_coefficient(step, config), 'gray_weight': 10.0,
            'ratio_limit': ratio_limit(step, config)}
