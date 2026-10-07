"""GPU data/numerical precheck only: zero optimizer updates."""
import torch
from rawfeat.config import load_config
from rawfeat.data import PairGenerator
from rawfeat.losses import pair_losses
from rawfeat.model import RawFeatureExtractor
from rawfeat.noise import CanonELD
from rawfeat.runtime import configure_fp32
from rawfeat.sensor import load_invisp
from rawfeat.teacher import load_teacher
from rawfeat.training import make_microbatch, training_paths, set_training_mode


def test_full_gpu_pair_forward_backward_and_pause():
    configure_fp32(); c = load_config('configs/smoke.yaml'); device = torch.device('cuda:0')
    generator = PairGenerator(load_invisp(c['invisp_repo'], device), CanonELD(c['eld_calibration']), device)
    teacher = load_teacher(c['superpoint_repo'], device)
    paths = training_paths(c)
    model = RawFeatureExtractor().to(device)
    for step in (299, 300):
        model.zero_grad(set_to_none=True); set_training_mode(model, c, step)
        batch, counts = make_microbatch(generator, teacher, paths, c, step, step*16, device)
        active = step == 300
        assert batch['packed'].shape == (8, 4, 240, 320)
        assert counts['valid_cells'] > 0 and counts['valid_pixels'] > 0
        assert batch['matches'] is not None if active else batch['matches'] is None
        out = model(batch['packed'], compute_descriptors=active)
        losses = pair_losses(out, batch['teacher_logits'], batch['valid_cells'], batch['clean_gray'], batch['valid_pixels'], batch['matches'], .01 if active else 0.)
        assert torch.isfinite(losses['total']); losses['total'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is not None if active else p.grad is None for p in model.descriptor.parameters())
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
