"""COCO crops, independent sRGB appearance, shared-view warps, and online Raw pairs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
import cv2
import kornia
import numpy as np
import torch
from torch.nn import functional as F

from .noise import CanonELD
from .sensor import BLACK_LEVEL, clean_gray, invisp_to_sensor_rgb, pack_student, sample_bayer


HEIGHT = 480
WIDTH = 640


@dataclass(frozen=True)
class PairRequest:
    image: str
    crop_seed: int
    appearance_a_seed: int
    appearance_b_seed: int
    geometry_seed: int
    noise_a_seed: int
    noise_b_seed: int
    ratio_a: float
    ratio_b: float
    clean: bool = False


def image_paths(coco_root: str | Path, split: str) -> list[Path]:
    directory = Path(coco_root) / split
    paths = sorted(directory.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no COCO JPEG images in {directory}")
    return paths


def crop_rgb(path: str | Path, seed: int, *, fixed: bool = False) -> tuple[np.ndarray, dict]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read COCO image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    source_height, source_width = image.shape[:2]
    rng = np.random.default_rng(seed)
    extra = 1.0 if fixed else float(rng.uniform(1.0, 1.2))
    scale = max(HEIGHT / source_height, WIDTH / source_width) * extra
    resized_width = max(WIDTH, round(source_width * scale))
    resized_height = max(HEIGHT, round(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    left = (resized_width - WIDTH) // 2 if fixed else int(rng.integers(0, resized_width - WIDTH + 1))
    top = (resized_height - HEIGHT) // 2 if fixed else int(rng.integers(0, resized_height - HEIGHT + 1))
    crop = resized[top:top + HEIGHT, left:left + WIDTH].copy()
    return crop, {"source_width": source_width, "source_height": source_height,
                  "resized_width": resized_width, "resized_height": resized_height,
                  "crop_left": left, "crop_top": top, "scale_x": resized_width / source_width,
                  "scale_y": resized_height / source_height}


def appearance(image: np.ndarray, seed: int) -> np.ndarray:
    transforms = A.OneOf([
        A.NoOp(p=0.70),
        A.ColorJitter(brightness=(0.9, 1.1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0), p=0.05),
        A.ColorJitter(brightness=(1, 1), contrast=(0.85, 1.15), saturation=(1, 1), hue=(0, 0), p=0.05),
        A.RandomGamma(gamma_limit=(85, 115), p=0.05),
        A.ColorJitter(brightness=(1, 1), contrast=(1, 1), saturation=(0.8, 1.2), hue=(0, 0), p=0.05),
        A.PlasmaShadow(shadow_intensity_range=(0.1, 0.35), roughness=0.7, p=0.05),
        A.OneOf([
            A.GaussianBlur(blur_limit=0, sigma_limit=(0.3, 1.0), p=0.5),
            A.MotionBlur(blur_limit=(3, 5), angle_range=(0, 180), direction_range=(0, 0), allow_shifted=False, p=0.5),
        ], p=0.05),
    ], p=1)
    return A.Compose([transforms], seed=seed)(image=image)["image"]


def geometry_matrix(seed: int, device: torch.device, max_attempts: int = 30) -> torch.Tensor:
    """Kornia affine followed by distortion_scale=0.15 perspective."""
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    dummy = torch.zeros((1, 1, HEIGHT, WIDTH), device=device)
    for attempt in range(max_attempts):
        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(seed + attempt)
            affine = kornia.augmentation.RandomAffine(30.0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0).to(device)
            perspective = kornia.augmentation.RandomPerspective(
                distortion_scale=torch.tensor(0.15, device=device, dtype=torch.float32), p=1.0
            ).to(device)
            perspective.set_rng_device_and_dtype(device=device, dtype=torch.float32)
            affine(dummy)
            perspective(dummy)
            homography = perspective.transform_matrix @ affine.transform_matrix
        valid_b, _ = valid_masks(homography[0])
        valid_a, _ = valid_masks(torch.linalg.inv(homography[0]))
        if valid_b[1].float().mean() >= 0.5 and valid_a[1].float().mean() >= 0.5:
            return homography[0]
    raise RuntimeError("could not sample a geometric pair with >=50% common coverage")


def warp_view(image: torch.Tensor, homography: torch.Tensor, *, fill: float = 0.0) -> torch.Tensor:
    warped = kornia.geometry.transform.warp_perspective(image, homography[None], (HEIGHT, WIDTH), align_corners=True)
    if fill:
        valid = kornia.geometry.transform.warp_perspective(
            torch.ones_like(image[:, :1]), homography[None], (HEIGHT, WIDTH), align_corners=True)
        warped += (1 - valid) * fill
    return warped


def valid_masks(homography: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    full = torch.ones((1, 1, HEIGHT, WIDTH), device=homography.device)
    warped = warp_view(full, homography)
    region = torch.cat((full, (warped >= 0.999).float()), dim=0)
    padded = F.pad(region, (8, 8, 8, 8), value=0)
    eroded = kornia.morphology.erosion(padded, torch.ones((17, 17), device=homography.device))[:, :, 8:-8, 8:-8] > 0.5
    cells = F.avg_pool2d(eroded.float(), 8, 8) == 1
    return eroded, cells[:, 0]


class PairGenerator:
    def __init__(self, invisp: torch.nn.Module, eld: CanonELD, device: torch.device):
        if device.type != "cuda":
            raise RuntimeError("Raw pair generation requires CUDA")
        self.invisp = invisp
        self.eld = eld
        self.device = device

    def _finish_base(self, srgb: torch.Tensor, sensor: torch.Tensor,
                     homography: torch.Tensor, crop: dict, image: str) -> dict:
        # Warp continuous RGB/sensor values first; Bayer sampling and noise follow.
        srgb = srgb.clone()
        sensor = sensor.clone()
        srgb[1:2] = warp_view(srgb[1:2], homography)
        sensor[1:2] = warp_view(sensor[1:2], homography, fill=BLACK_LEVEL)
        valid_pixels, valid_cells = valid_masks(homography)
        bayer = sample_bayer(sensor)
        gray_srgb = kornia.color.rgb_to_grayscale(srgb)
        return {"clean_bayer_dn": bayer, "srgb_gray": gray_srgb,
                "clean_gray": clean_gray(sensor), "valid_pixels": valid_pixels,
                "valid_cells": valid_cells, "homography": homography,
                "crop": crop, "image": image}

    @torch.no_grad()
    def generate_base(self, request: PairRequest, *, fixed_crop: bool = False) -> dict:
        base, crop = crop_rgb(request.image, request.crop_seed, fixed=fixed_crop)
        images = np.stack((appearance(base, request.appearance_a_seed), appearance(base, request.appearance_b_seed)))
        homography = geometry_matrix(request.geometry_seed, self.device)
        srgb = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).contiguous().float() / 255
        raw = self.invisp(srgb, rev=True)
        return self._finish_base(srgb, invisp_to_sensor_rgb(raw), homography, crop, request.image)

    @torch.no_grad()
    def apply_noise(self, base: dict, request: PairRequest) -> dict:
        bayer = base["clean_bayer_dn"]
        noisy = bayer.clone() if request.clean else torch.stack((
            self.eld.apply(bayer[0], request.ratio_a, request.noise_a_seed),
            self.eld.apply(bayer[1], request.ratio_b, request.noise_b_seed),
        ))
        return {**base, "packed": pack_student(noisy), "noisy_bayer_dn": noisy,
                "ratios": (request.ratio_a, request.ratio_b)}

    def generate(self, request: PairRequest, *, fixed_crop: bool = False) -> dict:
        return self.apply_noise(self.generate_base(request, fixed_crop=fixed_crop), request)
