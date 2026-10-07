"""The specified three-stage RepVGG student and training-only gray head."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, deploy: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.relu = nn.ReLU(inplace=False)
        if deploy:
            self.fused = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=True)
        else:
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            )
            self.identity = (
                nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1)
                if in_channels == out_channels and stride == 1 else None
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "fused"):
            return self.relu(self.fused(x))
        branches = self.conv3(x) + self.conv1(x)
        if self.identity is not None:
            branches = branches + self.identity(x)
        return self.relu(branches)

    @staticmethod
    def _fuse_conv_bn(branch: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
        conv, bn = branch
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = conv.weight * scale[:, None, None, None]
        bias = bn.bias + (torch.zeros_like(bn.running_mean) - bn.running_mean) * scale
        return weight, bias

    def switch_to_deploy(self) -> None:
        if hasattr(self, "fused"):
            return
        if self.training:
            raise RuntimeError("RepVGG fusion requires eval mode and frozen BN statistics")
        kernel3, bias3 = self._fuse_conv_bn(self.conv3)
        kernel1, bias1 = self._fuse_conv_bn(self.conv1)
        kernel = kernel3 + F.pad(kernel1, (1, 1, 1, 1))
        bias = bias3 + bias1
        if self.identity is not None:
            identity = torch.zeros_like(kernel)
            indices = torch.arange(self.in_channels, device=kernel.device)
            identity[indices, indices, 1, 1] = 1
            scale = self.identity.weight / torch.sqrt(self.identity.running_var + self.identity.eps)
            kernel = kernel + identity * scale[:, None, None, None]
            bias = bias + self.identity.bias - self.identity.running_mean * scale
        fused = nn.Conv2d(self.in_channels, self.out_channels, 3, self.stride, 1, bias=True)
        fused = fused.to(device=kernel.device, dtype=kernel.dtype)
        fused.weight.data.copy_(kernel)
        fused.bias.data.copy_(bias)
        self.fused = fused
        del self.conv3, self.conv1, self.identity


class RawFeatureExtractor(nn.Module):
    """Input [N,4,H/2,W/2]; outputs detector and descriptor grids at H/8."""

    def __init__(self, auxiliary: bool = True, deploy: bool = False):
        super().__init__()
        self.stage1 = nn.Sequential(RepVGGBlock(4, 24, deploy=deploy), RepVGGBlock(24, 24, deploy=deploy))
        self.stage2 = nn.Sequential(
            RepVGGBlock(24, 48, 2, deploy), RepVGGBlock(48, 48, deploy=deploy),
            RepVGGBlock(48, 48, deploy=deploy),
        )
        self.stage3 = nn.Sequential(
            RepVGGBlock(48, 96, 2, deploy), RepVGGBlock(96, 96, deploy=deploy),
            RepVGGBlock(96, 96, deploy=deploy), RepVGGBlock(96, 96, deploy=deploy),
        )
        self.detector = nn.Sequential(RepVGGBlock(96, 128, deploy=deploy), nn.Conv2d(128, 65, 1))
        self.descriptor = nn.Sequential(RepVGGBlock(96, 128, deploy=deploy), nn.Conv2d(128, 128, 1))
        self.auxiliary = auxiliary
        if auxiliary:
            self.gray_reduce = nn.Conv2d(48, 24, 1)
            self.gray_conv = nn.Sequential(nn.Conv2d(48, 24, 3, padding=1), nn.ReLU(inplace=False))
            self.gray_output = nn.Conv2d(24, 4, 1)
        if not deploy:
            self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        final_layers = [self.detector[-1], self.descriptor[-1]]
        if self.auxiliary:
            final_layers.append(self.gray_output)
        for layer in final_layers:
            nn.init.xavier_normal_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)

    def forward(self, packed: torch.Tensor, *, compute_descriptors: bool = True) -> dict[str, torch.Tensor]:
        s1 = self.stage1(packed)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        result = {"logits": self.detector(s3)}
        if compute_descriptors:
            result["descriptors"] = F.normalize(self.descriptor(s3), dim=1)
        if self.auxiliary:
            reduced = F.interpolate(self.gray_reduce(s2), size=s1.shape[-2:], mode="bilinear", align_corners=False)
            result["gray"] = F.pixel_shuffle(self.gray_output(self.gray_conv(torch.cat((s1, reduced), dim=1))), 2)
        return result

    def deploy_copy(self) -> "RawFeatureExtractor":
        model = copy.deepcopy(self).eval()
        if model.auxiliary:
            del model.gray_reduce, model.gray_conv, model.gray_output
            model.auxiliary = False
        for module in model.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()
        return model
