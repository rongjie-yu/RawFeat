# Implementation and verification audit

> 文档地位：历史工程实施检查，只作为证据记录。当前唯一执行方案是 [统一方案与检测排查路线](low_light_raw_feature_extractor_design.md)，本轮授权与顺序见其§17；本文的旧范围、旧协议或旧限制不覆盖当前用户指令。

The [subsequent implementation review](review_report.md) records fixes and current-worktree retests. This document preserves the earlier preparation/calibration evidence; its test counts, baseline scores and timing numbers are historical.

Status: the complete bounded check sequence finished on 2026-09-30. The formal 100,000-update training has not been started. [Calibration report](calibration_report.md) records measurements, parameter decisions and performance limitations. Historical audits/artifacts remain under `outputs/checks` and the planning ledger; they do not replace current evidence.

## Design coverage

| Design section | Implementation | Current verification | Pending formal-training evidence |
|---|---|---|---|
| §1 Goal/scope | `model.py`, `features.py`, `inference.py`: Bayer coordinates, scores, 128-D descriptors | Shape, normalization and crop-offset tests; actual GPU inference | Final accuracy and SLAM utility |
| §2 Train/deploy flow | Online frozen InvISP/teacher; one Bayer student; fusion | 50/2,000 updates completed, full-protocol export/reload verified | Best formal checkpoint |
| §3 Data/geometry/Raw/noise | `data.py`, `sensor.py`, `noise.py`: aspect-preserving COCO crops, independent Albumentations, Kornia, RGGB, calibrated DN, clean full-resolution gray | Identity/nontrivial coordinate-field tests; masks/cells; all 1,536 cached cases finite and consistent; independent ratio-256 backward | Sustained full-range course/real exposure assumptions |
| §4 Exact network | `model.py`: S1 24×2, S2 48×3, S3 96×4, prescribed heads/initialization | Unit checks; effective batch 16 CUDA training; trained-checkpoint fusion | Formal model quality |
| §5 Three losses | Official 65-class KL, balanced/spaced homography points and dual softmax, weighted gray MSE | Gradient/spacing tests; all calibration traces finite; per-component probes nonzero; no descriptor collapse observed | Full schedule convergence |
| §6 AdamW/accumulation | FP32, decay groups, 4 pairs ×4 accumulations, actual BN batch 8 views | 32,000 pairs in calibration, finite model/optimizer flow, ~1.73 GiB peak | Long-run cost distribution |
| §7 Schedules | `schedules.py`, formal/calibration YAML | Boundary tests plus all logged calibration LR/ratio/gray values checked | Formal 5k–40k and 60k–80k transitions |
| §8 Stability/order | Finite checks, divide micro loss by accumulation, one clip/update, probes and failure artifacts | Final 500 updates had no clipping; late norm median 1.82; gradient signals remain | Late formal extreme-range behavior |
| §9 Fixed validation | 256 sources, 1,536 cases, five equal noisy buckets plus clean; MNN/RANSAC/metrics | Complete cache audit; four 96-case calibration validations with 16 PNGs each; full 1,536-case original/deployed evaluation | Repeated full formal validation/best selection |
| §10 Baseline | Exact Raw-SLAM uint16 + shared double MeanAD + RGB gray + official SuperPoint | Full 1,536 predictions, all match indices valid/unique, unit descriptors; noisy score 60.739% | Trained formal student improvement |
| §11 Logs/checkpoints/resume | JSONL/TensorBoard, full state, retained validation checkpoints, best/latest | Four calibration checkpoints, latest step 2,000/cursor 32,000; current-path two-step recovery: identical RNG/cursor/schedule, FP32-close weights/moments | Formal best/history cycling |
| §12 Required trials | 50 updates, 2,000 updates on fixed 2,048-image train subset, two extreme pairs | Both trials complete; zero val overlap; four validations; empirical report keeps original settings | None for prescribed preparation trials |
| §13 Deployment/efficiency | Delete auxiliary head, fuse each RepVGG block, reload, full protocol, FP32 timings | Logits/descriptor errors 1.05e-5/5.44e-6; full score identical; 595,105 params, 4.033536G MACs; 0.749/1.445 ms | Repeat on selected formal best; current extraction timing used only 46 points |
| §14 Assets/environment | Pinned official sources/assets, independent CUDA Conda env, pinned definitions/install commands | Asset commits/hashes, actual CUDA runtime, direct dependencies and pip check | None for current environment readiness; formal run remains user-run |
| §15 References | Official MagicLeap, InvISP, ELD plus pinned Raw-SLAM interface sources | Source/asset provenance documented | None for source identity |
| §16 Conclusion | Complete runnable preparation and exact launch/resume/export commands | Read-only formal preflight ready=true; bounded preparation commands completed | Formal training and performance remain user-run/unproven |

## Authoritative evidence

| Evidence | Result |
|---|---|
| `outputs/checks/required_runs/tests.log` | 18 passed, four third-party SciPy deprecation warnings |
| `outputs/checks/full_cache_audit.json` | 1,536 cases; shapes, finiteness, same bases across buckets, whole valid cells and B overlap verified; packed range -4.0096 to 4.7487; re-pack error 4.77e-7 |
| `outputs/checks/full_baseline_audit.json` | Full predictions/descriptors/match indices and cache/baseline preflight verified |
| `outputs/checks/extreme_ratio_current.json` | Two ratio-256 pairs, one backward, zero updates; finite losses/gradients and signed inputs |
| `outputs/checks/short50_audit.json` | Step 50/cursor 800, finite checkpoint, six-case validation, update median 3.390 seconds |
| `outputs/checks/resume_current/report.json` | Identical RNG/cursor/schedules; model maximum difference 4.77e-7, optimizer 2.85e-7 |
| `outputs/checks/calibration2000_audit.json` | Step 2,000/cursor 32,000; all schedule traces checked; 2,048 unique train IDs with zero val overlap; four 96-case/16-PNG validations |
| `outputs/calibration/validation/step_*/baseline_comparison.json` | Correctly compares each partial validation to the same first-16-image baseline cases |
| `outputs/checks/calibration_export/export_report.json` | Source step 2,000; full protocol/reload/fusion agreement and measured FP32 efficiency |
| `outputs/checks/required_runs/preflight.log` | ready=true, 1,536 cached cases and current official baseline identity |
| `outputs/checks/clean_environment_audit.json` | Independent conda-forge rebuild, locked cached wheels, CUDA forward/backward, 18 tests, clean pip check and identical ready=true preflight |

Current manifest SHA-256: `34b5622eb5941649bdc39775ed4f160e9d0713fb916ae0d91a62ceeba216f9f7`.
Current cache identity: `bce4f73cdc9f9c5ec1ce84cd5f46ee27671a6b705e73c80a4eaec7fb91a8166e`.

## Executed command groups

From the repository root and installed environment:

```bash
bash scripts/fetch_assets.sh
bash scripts/run_required_checks.sh
bash outputs/checks/resume_current/run.sh
PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
"$PY" -m scripts.audit_calibration
"$PY" -m compileall -q rawfeat scripts tests
"$PY" -m pip check
"$PY" -m rawfeat.cli preflight --config configs/formal.yaml
```

The asset command verified the pinned official commits and weights. The bounded runner passed tests, complete cache/baseline, extreme check, short50, calibration2000, preflight and full-protocol export. Additional one-off numeric/provenance audits wrote the JSON evidence above. Early interrupted runs were archived at `outputs/checks/before_geometry` and `before_quantization`; current results were generated after both corrections.

## Corrections and limits

- The user selected the prior projects' full-DN exposure bridge. The student still subtracts black and normalizes once; negative/out-of-range noisy inputs are preserved. ADC quantization was corrected to +/-0.5 DN before normalization rather than inheriting a normalized q in the DN simulator.
- Kornia pixel-coordinate homography warp uses align_corners=True. Strong identity/nontrivial coordinate-field tests caught and corrected the previous half-pixel inconsistency.
- Explicit contiguous NCHW and preallocated/chunked point sampling reduced idle-device profile time 6.02→3.38 seconds while retaining exact sampled points on 16 checked cases. The original shared-device 13–14.5-second estimate is historical. Default compiled InvISP was slower and was not adopted.
- CUDA grid_sample backward is not bitwise deterministic in this build. Resume is verified numerically, with exact stored RNG/cursor/schedules.
- Calibration used the prescribed 200-step warmup and ratio course to 16. Formal YAML remains unchanged: effective batch 16, clip 5, gray 10→1, LR peak 3e-4/2k warmup, 100k budget and max ratio 256. The report explains the evidence for retaining these values.
- Calibration plus four validations took about 6,863 seconds (1.906 hours); median update 3.401 seconds, last-500 median 3.411 seconds. Compute-only formal projection is about 94.7 hours/3.95 days, excluding formal validation and sustained-load changes.
- The calibration checkpoint full noisy score is 1.388%, versus baseline 60.739%. No accuracy improvement is claimed; observed detection counts are low and the checkpoint is not a formal best model. Do not transfer its weights into the formal run or use its early AUC to change the prescribed network route.
- No formal training, final best checkpoint, real SLAM, HPatches-raw/MID or FP16/INT8 evaluation has been performed. These exclusions match the current task scope.

The independent rawfeat-clean environment was actually created using the README conda-forge command and locked requirements, reusing original pip-cache wheels via `pip install --no-index --find-links outputs/checks/wheelhouse -r requirements.txt`. Torch distribution metadata is 2.12.1 while its module/runtime is 2.12.1+cu130/CUDA13.0; the requirement now pins the distribution version. All direct pins match, CUDA ran, 18 tests passed (four SciPy warnings plus one optional Matplotlib warning), pip check is clean and cache/asset/baseline preflight matches the original environment. No full training was launched.
