"""Consistent full FP32 CUDA execution (including cuDNN convolutions)."""

import torch


def configure_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
