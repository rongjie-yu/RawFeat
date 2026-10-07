import torch

from rawfeat.features import extract_features, sample_descriptors, score_map
from rawfeat.inference import extract_bayer
from rawfeat.model import RawFeatureExtractor


def test_student_shapes_and_fusion():
    model = RawFeatureExtractor().eval()
    image = torch.randn(2, 4, 32, 40)
    with torch.no_grad():
        train_output = model(image)
        deployed = model.deploy_copy()
        deployed_output = deployed(image)
    assert train_output["logits"].shape == (2, 65, 8, 10)
    assert train_output["descriptors"].shape == (2, 128, 8, 10)
    assert train_output["gray"].shape == (2, 1, 64, 80)
    assert "gray" not in deployed_output
    for name in ("logits", "descriptors"):
        torch.testing.assert_close(train_output[name], deployed_output[name], atol=2e-5, rtol=2e-5)


def test_score_decode_and_descriptor_centers():
    logits = torch.full((1, 65, 2, 2), -20.0)
    logits[:, 3 * 8 + 4] = 20.0
    scores = score_map(logits)
    assert scores.shape == (1, 16, 16)
    assert scores[0, 3, 4] > 0.99
    grid = torch.zeros(2, 2, 2)
    grid[0, 0, 0] = 1
    grid[1, 1, 1] = 1
    samples = sample_descriptors(grid, torch.tensor([[3.5, 3.5], [11.5, 11.5]]), 16, 16)
    torch.testing.assert_close(samples, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    features = extract_features(logits, torch.randn(1, 128, 2, 2), border=0, max_points=2)
    assert tuple(features[0]["points"][0]) == (4.0, 3.0)


def test_deployment_crop_offset_and_descriptor_norm():
    model = RawFeatureExtractor(auxiliary=False).eval()
    bayer = torch.full((64, 80), 2548.0)
    local = extract_bayer(model, bayer, crop_offset=(0, 0), threshold=0.0, max_points=16)
    global_points = extract_bayer(model, bayer, crop_offset=(100, 200), threshold=0.0, max_points=16)
    torch.testing.assert_close(global_points["points"] - local["points"], torch.tensor([100.0, 200.0]).expand_as(local["points"]))
    assert local["descriptors"].shape[1] == 128
    torch.testing.assert_close(local["descriptors"].norm(dim=1), torch.ones(len(local["points"])))


def test_nms_resolves_equal_score_plateau():
    logits = torch.zeros((1, 65, 4, 4))
    dense = torch.randn((1, 128, 4, 4))
    features = extract_features(logits, dense, threshold=0, border=8, nms_radius=4, max_points=1024)[0]
    points = features["points"]
    assert len(points) == 16
    for index, point in enumerate(points):
        if index + 1 < len(points):
            assert ((points[index + 1:] - point).abs().amax(dim=1) > 4).all()


def test_invalid_pixels_do_not_suppress_valid_keypoints():
    logits = torch.full((1, 65, 4, 4), -20.0)
    logits[:, 64] = 0.0
    logits[0, 7 * 8 + 4, 0, 1] = 10.0  # (12,7) is inside the excluded border.
    logits[0, 1 * 8 + 4, 1, 1] = 8.0   # (12,9) must survive.
    dense = torch.randn(1, 128, 4, 4)
    result = extract_features(logits, dense)[0]
    assert result["points"].tolist() == [[12.0, 9.0]]
    valid = torch.ones(1, 32, 32, dtype=torch.bool)
    valid[0, 7, 12] = False
    result = extract_features(logits, dense, valid=valid, border=0)[0]
    assert result["points"].tolist() == [[12.0, 9.0]]
