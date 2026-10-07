# Low-light Bayer feature extractor

Implementation of [the unified design and bounded detection investigation](docs/low_light_raw_feature_extractor_design.md): a single 4-channel Bayer student, online frozen Canon InvISP, calibrated ELD noise, official MagicLeap SuperPoint teacher, three losses, fixed COCO validation, Raw-SLAM MeanAD baseline, and RepVGG deployment fusion. Formal training is **not** part of the prepared artifacts and has not been started here.

Current measured state is recorded in the [single unified design §20](docs/low_light_raw_feature_extractor_design.md), [completed staged comparison](docs/detection_gray_staged_report_20261002.md), and [documentation index](docs/README.md). Both same-parent runs added exactly 500 updates and stopped at absolute step 2000; this round added 1000, historical total 9209. Full 1536-pair mean H-AUC@5 over five noise buckets is 51.8136% for continued joint and 51.5805% for staged, against parent 43.4613% and baseline 65.6030%. The staged setting has no confirmed full-geometry advantage and is deferred. Formal defaults remain KL and provisional gray weight 10; no training continues automatically. Formal-run commands below are future interface references.

The [2026-10-01 independent design review and measured gray comparisons](docs/design_calibration_review_20261001.md) record the earlier bounded training evidence; the subsequent [detector-objective and example-code probes](docs/detection_design_discussion_20261001.md) add zero-update evidence. They cover offline thresholds, fixed clean/ratio1 learning, gray weights 1/3/10, matched random online trials, and full protocol-v2 validation. Shared-gradient logging and the remaining ratio256 unit test were corrected. Lower gray weights did not consistently improve localization and the main score; gray 10 remains provisional, and formal parameter readiness is not established. All commands and new artifacts are under `outputs/checks/design_calibration_20261001`; historical runs are preserved.

The [2026-09-30 review and early-training diagnostics](docs/review_report.md) supersede the historical verification numbers below. It fixes NMS, validation comparisons and run-history protection, reruns the full baseline, and diagnoses a detection bottleneck in the earlier calibration checkpoint. The 2,000-update calibration remains historical evidence; it was not retrained after these review fixes. Read that report before launching formal training.

The user subsequently fixed the training maximum ratio at 100 and the main validation buckets at `[1,4,16,64,100]` (protocol v2). The v1 manifest/cache/baseline, including ratio256 pressure results, are preserved under `outputs/checks/ratio100/protocol_v1_history`. Gray weight 10 remains a provisional value requiring measured calibration; the historical stable run does not establish that this weight is appropriate.

Run commands from this repository root. COCO train2017/val2017 are expected under `/homes/rongjie/datasets/coco`; change `coco_root` in the YAML files if needed. The tested environments are `/homes/rongjie/software/miniconda3/envs/rawfeat` and the independently rebuilt `/homes/rongjie/software/miniconda3/envs/rawfeat-clean`; both pass CUDA, tests and the same formal preflight.

## Environment and assets

The tested runtime is Python 3.12.13, PyTorch module 2.12.1+cu130 (distribution pin 2.12.1), CUDA13.0, Albumentations 2.0.8, Kornia 0.8.2, FP32 with cuDNN TF32 disabled, on an RTX 5090. [environment.yml](environment.yml) and [requirements.txt](requirements.txt) pin the direct dependencies. To make a clean independent environment without the host Conda default channels:

```bash
CONDA=/homes/rongjie/software/miniconda3/bin/conda
ENV=/homes/rongjie/software/miniconda3/envs/rawfeat-clean
"$CONDA" create -y --override-channels -c conda-forge -p "$ENV" python=3.12.13 pip=26.1.2
"$ENV/bin/python" -m pip install -r requirements.txt
"$ENV/bin/python" -c 'import torch; assert torch.cuda.is_available() and torch.version.cuda == "13.0"; print(torch.__version__, torch.cuda.get_device_name(0))'
```

The clean rebuild was tested with the same requirements and cached wheels to avoid duplicate downloads (`--no-index --find-links outputs/checks/wheelhouse`). Its CUDA/tests/dependencies/preflight pass; [clean environment evidence](outputs/checks/clean_environment_audit.json) records the result. For the already installed and tested environment, use:

```bash
PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
export CUDA_VISIBLE_DEVICES=1  # Choose an idle physical GPU; it becomes cuda:0 in this process.
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
bash scripts/fetch_assets.sh
"$PY" -m pytest -q tests
```

The asset script fetches three official repositories at pinned commits and checks checkpoint/calibration hashes. See [asset provenance](docs/asset_provenance.md) for paths, source versions, numeric domains, and license information. It does not edit the prior projects.

## Fixed validation data and baseline

The manifest selects 256 val2017 sources with seed 2027. Each has one fixed crop, geometry, and appearance pair; the cache stores one clean case and noisy ratios `[1,4,16,64,100]`, 1,536 pairs total. The cache and baseline commands are bounded to those exact cases and may take substantial time. They are prerequisites for formal training.

```bash
"$PY" -m rawfeat.cli manifest --coco-root /homes/rongjie/datasets/coco --output manifests/coco_val2017_seed2027.json
"$PY" -m rawfeat.cli cache-validation --manifest manifests/coco_val2017_seed2027.json --coco-root /homes/rongjie/datasets/coco --output outputs/validation_cache --max-images 256 --device cuda:0
"$PY" -m rawfeat.cli baseline --manifest manifests/coco_val2017_seed2027.json --cache outputs/validation_cache --output outputs/baseline --max-images 256 --device cuda:0
```

Check `outputs/validation_cache/identity.json`, `outputs/baseline/summary.json` (`full_protocol: true`, `cases: 1536`), `outputs/baseline/per_pair.jsonl`, and the 1,536 prediction archives. The student and baseline load the same cached Bayer frames. Cache identity binds the manifest, generation code, library versions, InvISP model and ELD calibration; formal preflight checks these and the baseline teacher/evaluation code. If any of them changes, use a new cache directory and rerun the baseline. For a bounded six-case protocol smoke, add `--max-images 1` and use separate output directories; partial scores cannot select a model.

Each prediction archive saves points, scores, descriptors and paired `match_indices_a`/`match_indices_b` arrays indexing the two point lists.

## Training checks and calibration

Run the one-update GPU smoke first. It uses the full 480×640 Bayer size, 4 pairs per student forward, 4 gradient accumulation passes, all three losses, and writes a complete checkpoint. It does not require validation cache:

```bash
"$PY" -m rawfeat.cli train --config configs/gpu_smoke.yaml --skip-validation
"$PY" -m scripts.check_extreme --output outputs/checks/extreme_ratio.json --device cuda:0
```

Inspect `outputs/gpu_smoke/scalars.jsonl`, `latest.pt`, and any `failure.json`. The second command makes two fixed maximum-ratio-100 pairs and performs one backward pass without an optimizer update. Run the design's approximately 50-update check after caching at least one validation image. It validates at the end and records gradient probes:

A fresh training run refuses an existing run directory. For a repeat, add `--output outputs/checks/new_short50` (or another unused path); repeat the same override with `--resume` when continuing it. Stage gradient probes now include S1/S2/S3 norms, weighted gray norms and task gradient cosine similarities. To rerun the whole bounded sequence, set `RAWFEAT_RUN_ROOT` to an unused directory.

```bash
"$PY" -m rawfeat.cli cache-validation --max-images 1
"$PY" -m rawfeat.cli train --config configs/short_check.yaml
```

The separate 2,000-update calibration uses a fixed 2,048-image train2017 subset, 200-step warmup, ratio ramp from 500 to 1,500 up to 16, constant loss weights 1/1/10, and fixed first-16-image validation every 500 updates. It is a long GPU job; run it and review loss component gradients, clipping frequency, throughput, memory, and validation cost before editing only the empirical formal settings:

```bash
"$PY" -m rawfeat.cli cache-validation --max-images 16
"$PY" -m rawfeat.cli train --config configs/calibration.yaml
```

Keep a record of any formal YAML changes, original/new values, observations, and measured effects. The calibration checkpoint is separate from `outputs/formal`; formal training starts from random student initialization. Fixed ratio-100 cases are included in the current cache and can be evaluated independently with `--max-images 1` after loading a checkpoint. Ratio 1 is still noisy; only the `clean` bucket omits ELD noise.

## Formal training, resume, validation, deployment

After calibration settings are fixed and the full cache/baseline have passed, launch the 100,000-update formal run. `configs/formal.yaml` preserves batch 4 × accumulation 4, AdamW, warmup/cosine LR, the 5k–40k ratio ramp, and the 60k–80k gray-weight ramp. The training command refuses an incomplete 1,536-case cache or mismatched baseline.

```bash
"$PY" -m rawfeat.cli preflight --config configs/formal.yaml
"$PY" -m rawfeat.cli train --config configs/formal.yaml
"$PY" -m rawfeat.cli train --config configs/formal.yaml --resume outputs/formal/latest.pt
```

`preflight` is read-only. It checks the fixed manifest, all cached cases, InvISP/ELD asset identity, the full baseline, official SuperPoint weight and evaluation code before any formal optimizer update.

Every 2,000 updates and at the end, validation writes student and baseline per-bucket metrics and signed differences, per-pair predictions, 16 fixed pair visualizations, and a retained checkpoint. Percentage metrics use percentage-point differences. `best.pt` points to the highest mean H-AUC@5px over the five noisy buckets; `latest.pt` points to the newest complete update. The clean bucket is reported separately. Scalar JSONL and TensorBoard include losses, schedule values, ratio samples, gradient norms/clipping, supervision counts, throughput, GPU memory, and available baseline deltas. The sampler is stateless per sample number and runs in the main process, so no worker RNG stream or prefetch queue is left unrecoverable; checkpoints also save Python, NumPy, CPU/CUDA Torch RNG, optimizer, BN, configuration, scheduler values, and sample cursor.

To evaluate a selected checkpoint and compare against the fixed baseline, then fuse/export and measure FP32 efficiency:

```bash
"$PY" -m rawfeat.cli validate --checkpoint outputs/formal/best.pt --output outputs/formal/best_validation --baseline-summary outputs/baseline/summary.json
"$PY" -m rawfeat.cli export --checkpoint outputs/formal/best.pt --output outputs/deploy/rawfeat_fused.pt --max-images 256 --benchmark-repeats 50
```

The export command checks grid outputs before/after fusion, reloads the saved fused model, evaluates both on the fixed protocol, and writes parameter count, MACs, and network/full-extraction latency to `outputs/deploy/export_report.json`. `rawfeat.export.load_export` loads the fused model, and `rawfeat.inference.extract_bayer` applies the documented Canon level/WB processing, packs RGGB, and restores a supplied crop offset. Pass exposure-normalized Bayer sensor DN; unknown exposure compatibility is outside this student design.

## Historical preparation status and cost (protocol v1)

All prescribed preparation checks have completed: 18 tests passed; all 1,536 cached cases were audited for shapes, finite values, same clean/geometry bases and DN-to-packed consistency; the full official baseline produced 1,536 predictions; extreme ratio-256 backward was finite; the 50-update check and 2,000-update calibration finished. Four calibration checkpoints/96-case validations/16 visualizations were retained. Current-data resume restores equal RNG/cursor/schedules with maximum model difference 4.77e-7 and optimizer difference 2.85e-7; CUDA grid_sample backward prevents a bitwise model claim. Read-only formal preflight returns ready=true. See [implementation audit](docs/implementation_audit.md) and [calibration report](docs/calibration_report.md) for requirement coverage, exact commands, evidence and empirical decisions.

The 2,000 updates plus four validations took about **1 hour 54 minutes**, with **3.40 seconds/update** and **1.73 GiB** peak allocated memory. Clip threshold 5, gray 10→1, effective batch 16, formal LR/warmup, 100k budget were historically retained after that run; the user has since fixed the ratio maximum at 100 and identified gray weight 10 as requiring measured calibration. Compute-only projection of the formal run is approximately **95 hours /4 days**; validation and sustained-load changes are additional. Early 13–14.5-second measurements are historical; explicit contiguous NCHW and point-sampling improvements reduced an idle-device profile from 6.02 to 3.38 seconds.

The calibration checkpoint's historical protocol-v1 full noisy H-AUC@5px is **1.39%**, versus **60.74%** for the same v1 MeanAD + SuperPoint baseline. These scores include the old ratio256 pressure bucket and must not be compared with the current v2 baseline. No accuracy improvement is claimed. These are early calibration results, and formal training quality remains unknown. Formal training starts from random initialization; do not resume the calibration checkpoint into the formal course.

Fusion/reload passed and the full 1,536-case score stayed identical: maximum logits/descriptor differences 1.05e-5/5.44e-6. The deploy model has 595,105 parameters and 4.033536G MACs. On RTX 5090, FP32, batch 1, the calibration checkpoint measured 0.749 ms network and 1.445 ms full extraction. That fixed clean example produced only 46 points; repeat on the final best checkpoint to measure its actual extraction cost. Artifacts are under `outputs/checks/calibration_export`.

`bash scripts/run_required_checks.sh` executes the finite test/cache/baseline/extreme/short50/calibration2000/preflight/export sequence, with stage logs in `outputs/checks/required_runs`. `python -m scripts.audit_calibration` audits the completed calibration and writes per-checkpoint comparisons against the same first-16-image baseline subset. Historical data before warp/ADC unit repairs remains in `outputs/checks/before_geometry` and `before_quantization`; those results do not establish current completion.

The 100,000-update formal training has **not** been launched. Real SLAM, HPatches-raw/MID and FP16/INT8 evaluation remain outside this implementation scope.
