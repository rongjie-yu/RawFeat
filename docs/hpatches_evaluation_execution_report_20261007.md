# HPatches synthetic Raw evaluation execution record (historical 2026-10-07)

This file records the pre-closeout implementation and is retained as historical evidence. Its old batch=4 timing and separate `hpatches-report` command are superseded; the current formal flow is batch=1 and report generation is inside `hpatches-evaluate`.

This record freezes the current full-image protocol as `rawfeat.hpatches.synthetic_raw.v1`. It uses `/homes/rongjie/datasets/hpatches-sequences-release`, 116 sequences (57 illumination, 59 viewpoint), 696 RGB images and the 580 official `1 -> 2..6` pairs. The only conditions are `clean`, `ratio1`, `ratio4`, `ratio16`, `ratio64`, and `ratio100`; every condition uses all 580 pairs, so the formal count is **3480 pairs per model**. The five noisy conditions are equally weighted for the main H-AUC@5. No `(1,r)` or `(r,1)` condition is present in the HPatches manifest or evaluator.

The fixed manifest is [manifests/hpatches_synthetic_raw_v1.json](../manifests/hpatches_synthetic_raw_v1.json), SHA256 `1e0f775acd0ac1d47d981ddbf3bbb2870e2b0e7ed390e583a1e957b5d413c147`. CPU preflight with all 1276 integrity hashes passed against dataset integrity SHA256 `06ad61ea8ebbc8a3e15461ad4e84412c336a7782d20c58b4efa5f1b0d73501c9` and archive SHA256 `99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670`.

The smoke used `i_ajuntament`, `i_autannes`, `v_abstract`, and `v_adam`: 20 base pairs, 120 pairs per method, and six conditions. Its cache contains 24 images × 6 FP32 Bayer entries and has cache identity `4bfc226a8d07df7280699ad8345ae5ad9876dea9444ce9565d5c1204bff58fca`. The final smoke output is under `outputs/hpatches_synthetic_raw_v1/smoke_final2/` (ignored runtime output); it includes the copied manifest/cache identity, command/environment manifest, image inference timings and peak allocations, both per-method and combined `per_pair.jsonl`, sequence summaries, 10,000-bootstrap records, and two point/matching/H visualizations per method.

Smoke all-sequence H-AUC@5 was student 0.358234 and MeanAD+SuperPoint baseline 0.416216; the noisy-condition mean was therefore 0.358234 versus 0.416216 (−5.798 percentage points). This is a bounded smoke check, not a formal result. Student/baseline inference timing averaged 15.0/47.7 ms per image at batch size 4, with recorded peak allocations of 595/774 MiB. The timed smoke command took 26.89 seconds; cache generation for 24 images completed in about 32 seconds.

Checks run:

```bash
/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python -m rawfeat.cli hpatches-preflight \
  --dataset-root /homes/rongjie/datasets/hpatches-sequences-release \
  --integrity /homes/rongjie/datasets/hpatches-sequences-release/dataset_integrity.json --verify-hashes
/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python -m pytest -q tests
```

The test suite passed: 78 tests. The default repository-wide `pytest` command also collects third-party vendor tests that require unavailable historical dependencies; it is not the project check used here.

The full run is intentionally not started in this session. After the smoke artifacts are reviewed, launch these commands manually:

```bash
PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
$PY -m rawfeat.cli hpatches-cache \
  --manifest manifests/hpatches_synthetic_raw_v1.json \
  --cache outputs/hpatches_synthetic_raw_v1/cache_full \
  --invisp-repo third_party/Invertible-ISP \
  --eld-calibration third_party/ELD/camera_params/release/CanonEOS5D4_params.npy \
  --device cuda:0
$PY -m rawfeat.cli hpatches-evaluate \
  --checkpoint weights/v1/checkpoint_step_096000.pt \
  --manifest manifests/hpatches_synthetic_raw_v1.json \
  --cache outputs/hpatches_synthetic_raw_v1/cache_full \
  --output outputs/hpatches_synthetic_raw_v1/full_step096000 \
  --superpoint-repo third_party/SuperPointPretrainedNetwork \
  --device cuda:0 --batch-images 4
```

For a completed run, generate the readable report without rerunning inference:

```bash
$PY -m rawfeat.cli hpatches-report \
  --evaluation outputs/hpatches_synthetic_raw_v1/full_step096000
```

This writes `report/report.html`, `report/report.md`, `report/summary.csv`, `report/pairs.html`, `report/pairs.csv`, and `report/failures.jsonl` inside the evaluation directory.

The preferred checkpoint SHA256 is `d9a738b98e26e18e420062a83e575a300172a91f87bb5f0a687539203d675075`; the baseline SuperPoint checkpoint SHA256 is `52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e`. A full cache is approximately 4.9 GB of FP32 Bayer tensors; reserve at least 8 GB of free disk for cache, temporary files, and reports. Extrapolating the smoke gives roughly 15–20 minutes for cache generation and 13–20 minutes for the two-method evaluation, with at least 2 GB host RAM and 2 GB GPU headroom (measured smoke peak allocations were below 1 GB). Runtime varies with storage and GPU load.

New/modified implementation files are `rawfeat/hpatches.py`, `rawfeat/cli.py`, `tests/test_hpatches.py`, `manifests/hpatches_synthetic_raw_v1.json`, and the target hash map in `rawfeat/gpu_resume_revision.json`. Training code, training configuration, model/weights, noise/sensor math, and COCO validation protocol were not changed. The newly added implementation was simplified under the requested code-simplifier instructions and reviewed with `ocr delegate preview/rule`; the four OCR-reviewable files were covered, while OCR explicitly excluded the Markdown files and default-path test file. No unresolved review finding remains.
