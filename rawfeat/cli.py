"""Commands for fixed data preparation, training, validation and deployment."""

from __future__ import annotations

import argparse
import json

import torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rawfeat.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--coco-root", default="/homes/rongjie/datasets/coco")
    manifest.add_argument("--output", default="manifests/coco_val2017_seed2027.json")

    cache = sub.add_parser("cache-validation")
    cache.add_argument("--manifest", default="manifests/coco_val2017_seed2027.json")
    cache.add_argument("--coco-root", default="/homes/rongjie/datasets/coco")
    cache.add_argument("--output", default="outputs/checks/detection_followup_20261002/validation_cache_rngsplit")
    cache.add_argument("--max-images", type=int, default=256)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--invisp-repo", default="third_party/Invertible-ISP")
    cache.add_argument("--eld-calibration", default="third_party/ELD/camera_params/release/CanonEOS5D4_params.npy")

    baseline = sub.add_parser("baseline")
    baseline.add_argument("--manifest", default="manifests/coco_val2017_seed2027.json")
    baseline.add_argument("--cache", default="outputs/checks/detection_followup_20261002/validation_cache_rngsplit")
    baseline.add_argument("--output", default="outputs/baseline")
    baseline.add_argument("--max-images", type=int, default=256)
    baseline.add_argument("--device", default="cuda:0")
    baseline.add_argument("--superpoint-repo", default="third_party/SuperPointPretrainedNetwork")

    validate = sub.add_parser("validate")
    validate.add_argument("--checkpoint", required=True)
    validate.add_argument("--manifest", default="manifests/coco_val2017_seed2027.json")
    validate.add_argument("--cache", default="outputs/checks/detection_followup_20261002/validation_cache_rngsplit")
    validate.add_argument("--output", required=True)
    validate.add_argument("--baseline-summary")
    validate.add_argument("--max-images", type=int, default=256)
    validate.add_argument("--device", default="cuda:0")

    training = sub.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--resume")
    training.add_argument("--max-updates", type=int)
    training.add_argument("--output", help="use a separate run directory; repeat this override when resuming")
    training.add_argument("--allow-gpu-change", action="store_true",
                          help="resume on another single GPU of the same model; record the identity transition")

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", default="configs/formal.yaml")

    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--manifest", default="manifests/coco_val2017_seed2027.json")
    export.add_argument("--cache", default="outputs/checks/detection_followup_20261002/validation_cache_rngsplit")
    export.add_argument("--max-images", type=int, default=256)
    export.add_argument("--benchmark-repeats", type=int, default=50)
    export.add_argument("--device", default="cuda:0")
    diagnostics = sub.add_parser('diagnose')
    diagnostics.add_argument('--checkpoint', required=True)
    diagnostics.add_argument('--output', required=True)
    diagnostics.add_argument('--max-images', type=int, default=16)
    diagnostics.add_argument('--device', default='cuda:0')
    analysis = sub.add_parser('analyze')
    analysis.add_argument('--run', required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "manifest":
        from .validation import create_manifest, manifest_digest
        report = create_manifest(args.coco_root, args.output)
        print(json.dumps({"manifest": args.output, "sha256": manifest_digest(report),
                          "images": report["image_count"], "cases": report["case_count"]}))
        return
    if args.command == "train":
        from .training import train
        print(json.dumps(train(args.config, resume=args.resume, max_updates=args.max_updates,
                               output_root=args.output, allow_gpu_change=args.allow_gpu_change)))
        return
    if args.command == "preflight":
        from .training import formal_preflight, load_config
        try:
            report = formal_preflight(load_config(args.config))
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(json.dumps({"ready": False, "reason": str(error)}))
            raise SystemExit(1) from error
        print(json.dumps(report))
        return
    if args.command == 'analyze':
        from .analysis import analyze_run
        print(json.dumps(analyze_run(args.run), allow_nan=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; check nvidia-smi, driver and Conda PyTorch installation")
    from .runtime import configure_fp32
    configure_fp32()
    device = torch.device(args.device)
    if args.command == "cache-validation":
        from .data import PairGenerator
        from .noise import CanonELD
        from .sensor import load_invisp
        from .validation import cache_validation
        generator = PairGenerator(load_invisp(args.invisp_repo, device), CanonELD(args.eld_calibration), device)
        report = cache_validation(args.manifest, args.coco_root, args.output, generator, max_images=args.max_images)
    elif args.command == "baseline":
        from .teacher import load_teacher
        from .validation import evaluate
        report = evaluate(args.manifest, args.cache, args.output,
                          teacher=load_teacher(args.superpoint_repo, device), max_images=args.max_images)
    elif args.command == 'diagnose':
        from pathlib import Path
        from .diagnostics import diagnose_model
        from .identity import check_schema
        from .model import RawFeatureExtractor
        from .teacher import load_teacher
        stored = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        check_schema(stored)
        model = RawFeatureExtractor().to(device).eval()
        model.load_state_dict(stored['student'], strict=True)
        report = diagnose_model(model, load_teacher(stored['config']['superpoint_repo'], device), stored['config'],
                                Path(args.output), stored['step'], images=args.max_images,
                                geometry=stored['step'] > stored['config']['schedule']['detection_updates'])
    elif args.command == "validate":
        from .model import RawFeatureExtractor
        from .validation import evaluate
        stored = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        from .identity import check_schema
        check_schema(stored)
        if stored['step'] <= stored['config']['schedule']['detection_updates']:
            raise ValueError('A validation uses diagnose for detection/gray; geometry is unavailable')
        model = RawFeatureExtractor().to(device).eval()
        model.load_state_dict(stored["student"], strict=True)
        report = evaluate(args.manifest, args.cache, args.output, model=model,
                          baseline_summary=args.baseline_summary, max_images=args.max_images)
    else:
        from .export import export_checkpoint
        report = export_checkpoint(args.checkpoint, args.output, cache_root=args.cache,
                                   manifest=args.manifest, device=device, max_images=args.max_images,
                                   benchmark_repeats=args.benchmark_repeats)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
