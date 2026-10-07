"""Calibrated Canon ELD shot, Tukey read, row and quantization noise."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

class CanonELD:
    def __init__(self, calibration: str | Path):
        self.calibration = Path(calibration)
        profile = np.load(self.calibration, allow_pickle=True).item()
        self.k_min = float(profile["Kmin"])
        self.k_max = float(profile["Kmax"])
        self.g_shape = np.asarray(profile["G_shape"], dtype=np.float64)
        self.g_scale = profile["Profile-1"]["G_scale"]
        self.row_scale = profile["Profile-1"]["R_scale"]
        histogram, edges = np.histogram(self.g_shape, bins=10, range=(-0.25, 0.25))
        self.shape_probabilities = histogram / histogram.sum()
        self.shape_edges = edges

    @staticmethod
    def _scale(parameters: dict, log_k: float, rng: np.random.Generator) -> float:
        return math.exp(float(parameters["slope"]) * log_k + float(parameters["bias"])
                        + float(parameters["sigma"]) * rng.standard_normal())

    def apply(self, clean_bayer_dn: torch.Tensor, ratio: float, seed: int) -> torch.Tensor:
        """Apply the prior projects' full-DN exposure sequence; do not clip the output."""
        if clean_bayer_dn.ndim != 2 or ratio < 1:
            raise ValueError("ELD expects one full-resolution Bayer frame and ratio >= 1")
        rng = np.random.default_rng(seed)
        # Poisson can consume more than PyTorch CUDA's reserved Philox offset.
        # Separate streams prevent overlap with the other independent ELD terms.
        generator = torch.Generator(device=clean_bayer_dn.device).manual_seed(seed)
        component_seeds = np.random.SeedSequence([seed, 1]).generate_state(3, dtype=np.uint64)
        read_generator, row_generator, quant_generator = [
            torch.Generator(device=clean_bayer_dn.device).manual_seed(int(value)) for value in component_seeds
        ]
        log_k = rng.uniform(math.log(self.k_min), math.log(self.k_max))
        k = math.exp(log_k)
        shape_index = rng.choice(len(self.shape_probabilities), p=self.shape_probabilities)
        tukey_shape = rng.uniform(self.shape_edges[shape_index], self.shape_edges[shape_index + 1])
        g_scale = self._scale(self.g_scale, log_k, rng)
        row_scale = self._scale(self.row_scale, log_k, rng)
        low = clean_bayer_dn / ratio
        shot = torch.poisson(low.clamp_min(0) / k, generator=generator) * k
        uniform = torch.rand(low.shape, device=low.device, dtype=low.dtype, generator=read_generator).clamp(1e-7, 1 - 1e-7)
        if abs(tukey_shape) < 1e-5:
            read = torch.log(uniform) - torch.log1p(-uniform)
        else:
            read = (uniform.pow(tukey_shape) - (1 - uniform).pow(tukey_shape)) / tukey_shape
        row = torch.randn((low.shape[0], 1), device=low.device, dtype=low.dtype, generator=row_generator) * row_scale
        # One ADC count is one DN here; level normalization happens in pack_student.
        quant = torch.rand(low.shape, device=low.device, dtype=low.dtype, generator=quant_generator) - 0.5
        return (shot + read * g_scale + row + quant) * ratio


def no_noise(clean_bayer_dn: torch.Tensor) -> torch.Tensor:
    return clean_bayer_dn.clone()
