import numpy as np
import torch

from rawfeat.noise import CanonELD
from rawfeat.sensor import BLACK_LEVEL, SIGNAL_RANGE, WHITE_BALANCE, clean_gray, invisp_to_sensor_rgb, pack_student, sample_bayer


def test_sensor_bridge_phase_and_clean_gray():
    inverse = torch.full((1, 3, 4, 4), 0.5)
    sensor = invisp_to_sensor_rgb(inverse)
    assert sensor.min() >= BLACK_LEVEL
    assert sensor.max() <= BLACK_LEVEL + SIGNAL_RANGE
    bayer = sample_bayer(sensor)
    packed = pack_student(bayer)
    assert packed.shape == (1, 4, 2, 2)
    assert clean_gray(sensor).shape == (1, 1, 4, 4)
    reference = ((sensor - BLACK_LEVEL) / SIGNAL_RANGE) * sensor.new_tensor((WHITE_BALANCE[0], WHITE_BALANCE[1], WHITE_BALANCE[3]))[None, :, None, None]
    expected = (reference * sensor.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]).sum(1, keepdim=True)
    torch.testing.assert_close(clean_gray(sensor), expected)


def test_noise_is_seeded_and_preserves_negative_values():
    profile = 'third_party/ELD/camera_params/release/CanonEOS5D4_params.npy'
    eld = CanonELD(profile)
    clean = torch.full((32, 32), BLACK_LEVEL)
    a = eld.apply(clean, 100.0, 11)
    b = eld.apply(clean, 100.0, 11)
    torch.testing.assert_close(a, b)
    assert np.isfinite(a.numpy()).all()
    assert (pack_student(a[None]) < 0).any()


def test_quantization_uses_one_dn_before_student_normalization(monkeypatch):
    eld = CanonELD('third_party/ELD/camera_params/release/CanonEOS5D4_params.npy')
    monkeypatch.setattr(eld, '_scale', lambda parameters, log_k, rng: 0.0)
    monkeypatch.setattr(torch, 'poisson', lambda rate, generator: rate)
    clean = torch.full((128, 128), BLACK_LEVEL)
    noisy = eld.apply(clean, 1.0, 42)
    quant = noisy - clean
    assert 0.49 < float(quant.abs().max()) <= 0.5
    assert 0.28 < float(quant.std()) < 0.30
    packed = pack_student(noisy[None])
    expected_std = torch.tensor(WHITE_BALANCE) / (np.sqrt(12) * SIGNAL_RANGE)
    torch.testing.assert_close(packed[0].flatten(1).std(dim=1), expected_std, rtol=0.04, atol=0)


def test_cuda_noise_terms_do_not_share_poisson_random_numbers(monkeypatch):
    # In the affected CUDA implementation, one shared stream produced rho~=0.12.
    eld = CanonELD('third_party/ELD/camera_params/release/CanonEOS5D4_params.npy')
    clean = torch.full((512, 512), BLACK_LEVEL, device='cuda:0')
    full = (eld.apply(clean, 100., 11) - clean).double()
    sample_scale = eld._scale
    # Consume the same profile draws, isolating shot+quantization through the public API.
    monkeypatch.setattr(eld, '_scale', lambda parameters, log_k, rng: sample_scale(parameters, log_k, rng) * 0)
    shot_quant = (eld.apply(clean, 100., 11) - clean).double()
    read_row = full - shot_quant
    correlation = torch.corrcoef(torch.stack((shot_quant.flatten(), read_row.flatten())))[0, 1]
    assert abs(float(correlation)) < .01
