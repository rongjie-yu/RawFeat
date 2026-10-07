import cv2
import numpy as np
import torch
import pytest
import json
from torch.nn import functional as F

from rawfeat.data import HEIGHT, WIDTH, geometry_matrix, valid_masks, warp_view
from rawfeat.validation import RATIOS, read_manifest


def test_gpu_homography_and_valid_cells():
    device = torch.device("cuda:0")
    first = geometry_matrix(13, device)
    second = geometry_matrix(13, device)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    pixels, cells = valid_masks(first)
    inverse_pixels, _ = valid_masks(torch.linalg.inv(first))
    assert pixels.shape == (2, 1, HEIGHT, WIDTH)
    assert cells.shape == (2, HEIGHT // 8, WIDTH // 8)
    assert pixels[1].float().mean() >= 0.5
    assert inverse_pixels[1].float().mean() >= 0.5
    assert not pixels[0, 0, 0].any() and not pixels[0, 0, -1].any()
    assert not pixels[0, 0, :, 0].any() and not pixels[0, 0, :, -1].any()
    assert torch.equal(cells, F.avg_pool2d(pixels.float(), 8, 8)[:, 0] == 1)

    impulse = torch.zeros((1, 1, HEIGHT, WIDTH), device=device)
    source = np.array([[[320.0, 240.0]]], np.float32)
    impulse[0, 0, 240, 320] = 1
    warped = warp_view(impulse, first)
    y, x = divmod(int(warped[0, 0].argmax()), WIDTH)
    expected = cv2.perspectiveTransform(source, first.cpu().numpy().astype(np.float64))[0, 0]
    assert np.linalg.norm(np.array([x, y]) - expected) < 2


def test_fixed_manifest_counts():
    manifest, digest = read_manifest("manifests/coco_val2017_seed2027.json")
    assert len(digest) == 64
    assert len({image["image"] for image in manifest["images"]}) == 256
    for ratio in RATIOS:
        cases = [image["cases"][1 + RATIOS.index(ratio)] for image in manifest["images"]]
        assert len(cases) == 256
        if ratio > 1:
            assert sum(case["ratio_a"] == ratio and case["ratio_b"] == ratio for case in cases) == 128
            assert sum(case["ratio_a"] == 1 and case["ratio_b"] == ratio for case in cases) == 64
            assert sum(case["ratio_a"] == ratio and case["ratio_b"] == 1 for case in cases) == 64


@pytest.mark.parametrize("damage", ["duplicate", "training_source", "bucket", "clean", "ratio", "seed"])
def test_manifest_rejects_changed_fixed_protocol(tmp_path, damage):
    manifest, _ = read_manifest("manifests/coco_val2017_seed2027.json")
    image = manifest["images"][0]
    if damage == "duplicate":
        image["image"] = manifest["images"][1]["image"]
    elif damage == "training_source":
        image["image"] = image["image"].replace("val2017/", "train2017/")
    elif damage == "bucket":
        image["cases"][2]["bucket"] = "1"
    elif damage == "clean":
        image["cases"][0]["clean"] = False
    elif damage == "ratio":
        image["cases"][2]["ratio_a"] = 1.0
    else:
        image["geometry_seed"] = -1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_manifest(path)


def test_gpu_warp_matches_pixel_coordinate_homography():
    device = torch.device("cuda:0")
    y, x = torch.meshgrid(torch.arange(HEIGHT, device=device, dtype=torch.float32),
                          torch.arange(WIDTH, device=device, dtype=torch.float32), indexing="ij")
    coordinates = torch.stack((x, y), dim=0)[None]
    identity = warp_view(coordinates, torch.eye(3, device=device))
    torch.testing.assert_close(identity[:, :, 8:-8, 8:-8], coordinates[:, :, 8:-8, 8:-8], rtol=0, atol=2e-4)

    matrix = np.array([[0.94, 0.07, 21.25], [-0.03, 1.04, -12.5],
                       [0.00008, -0.00006, 1.0]], dtype=np.float32)
    homography = torch.from_numpy(matrix).to(device)
    warped = warp_view(coordinates, homography)
    destination = coordinates[0].permute(1, 2, 0).cpu().numpy().reshape(1, -1, 2)
    expected = cv2.perspectiveTransform(destination, np.linalg.inv(matrix.astype(np.float64)))[0]
    expected = torch.from_numpy(expected.reshape(HEIGHT, WIDTH, 2)).to(device).permute(2, 0, 1)
    pixels, _ = valid_masks(homography)
    torch.testing.assert_close(warped[0, :, pixels[1, 0]], expected[:, pixels[1, 0]], rtol=0, atol=1e-3)
