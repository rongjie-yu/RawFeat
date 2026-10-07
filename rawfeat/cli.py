"""Commands for fixed data preparation, training, validation and deployment."""

from __future__ import annotations

import argparse
import json
import sys

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
    export.add_argument("--checkpoint", default="weights/v1/checkpoint_step_096000.pt")
    export.add_argument("--output", default="outputs/deployment/v1/step096000/deployment.pt")
    export.add_argument("--manifest", default="manifests/hpatches_synthetic_raw_v1.json")
    export.add_argument("--cache", default="outputs/hpatches_synthetic_raw_v1/cache_full")
    export.add_argument("--max-images", type=int, default=256)
    export.add_argument("--benchmark-repeats", type=int, default=100)
    export.add_argument("--warmup", type=int, default=20)
    export.add_argument("--force", action="store_true")
    export.add_argument("--device", default="cuda:0")
    diagnostics = sub.add_parser('diagnose')
    diagnostics.add_argument('--checkpoint', required=True)
    diagnostics.add_argument('--output', required=True)
    diagnostics.add_argument('--max-images', type=int, default=16)
    diagnostics.add_argument('--device', default='cuda:0')
    analysis = sub.add_parser('analyze')
    analysis.add_argument('--run', required=True)

    hpatches_preflight = sub.add_parser('hpatches-preflight')
    hpatches_preflight.add_argument('--dataset-root', default='/homes/rongjie/datasets/hpatches-sequences-release')
    hpatches_preflight.add_argument('--integrity', default='/homes/rongjie/datasets/hpatches-sequences-release/dataset_integrity.json')
    hpatches_preflight.add_argument('--verify-hashes', action='store_true')

    hpatches_manifest = sub.add_parser('hpatches-manifest')
    hpatches_manifest.add_argument('--dataset-root', default='/homes/rongjie/datasets/hpatches-sequences-release')
    hpatches_manifest.add_argument('--integrity', default='/homes/rongjie/datasets/hpatches-sequences-release/dataset_integrity.json')
    hpatches_manifest.add_argument('--output', default='manifests/hpatches_synthetic_raw_v1.json')
    hpatches_manifest.add_argument('--verify-hashes', action='store_true')

    hpatches_cache = sub.add_parser('hpatches-cache')
    hpatches_cache.add_argument('--manifest', default='manifests/hpatches_synthetic_raw_v1.json')
    hpatches_cache.add_argument('--cache', default='outputs/hpatches_synthetic_raw_v1/cache')
    hpatches_cache.add_argument('--invisp-repo', default='third_party/Invertible-ISP')
    hpatches_cache.add_argument('--eld-calibration', default='third_party/ELD/camera_params/release/CanonEOS5D4_params.npy')
    hpatches_cache.add_argument('--device', default='cuda:0')
    hpatches_cache.add_argument('--sequences', nargs='+')
    hpatches_cache.add_argument('--force', action='store_true')

    hpatches_evaluate = sub.add_parser('hpatches-evaluate')
    hpatches_evaluate.add_argument('--checkpoint', default='outputs/deployment/v1/step096000/deployment.pt')
    hpatches_evaluate.add_argument('--manifest', default='manifests/hpatches_synthetic_raw_v1.json')
    hpatches_evaluate.add_argument('--cache', default='outputs/hpatches_synthetic_raw_v1/cache_full')
    hpatches_evaluate.add_argument('--output', default='outputs/hpatches_synthetic_raw_v1/final_step096000_single_image')
    hpatches_evaluate.add_argument('--superpoint-repo', default='third_party/SuperPointPretrainedNetwork')
    hpatches_evaluate.add_argument('--device', default='cuda:0')
    hpatches_evaluate.add_argument('--sequences', nargs='+')
    hpatches_evaluate.add_argument('--target-indices', nargs='+', type=int)
    hpatches_evaluate.add_argument('--max-pairs', type=int)
    hpatches_evaluate.add_argument('--batch-images', type=int, default=1)

    hpatches_deploy_smoke = sub.add_parser('hpatches-deploy-smoke')
    hpatches_deploy_smoke.add_argument('--checkpoint', default='weights/v1/checkpoint_step_096000.pt')
    hpatches_deploy_smoke.add_argument('--deployment', default='outputs/deployment/v1/step096000/deployment.pt')
    hpatches_deploy_smoke.add_argument('--manifest', default='manifests/hpatches_synthetic_raw_v1.json')
    hpatches_deploy_smoke.add_argument('--cache', default='outputs/hpatches_synthetic_raw_v1/cache_full')
    hpatches_deploy_smoke.add_argument('--output', default='outputs/deployment/v1/step096000/smoke')
    hpatches_deploy_smoke.add_argument('--device', default='cuda:0')
    hpatches_deploy_smoke.add_argument('--sequences', nargs='+',
                                       default=['i_ajuntament', 'i_autannes', 'v_abstract', 'v_adam'])
    hpatches_deploy_smoke.add_argument('--batch-images', type=int, default=1)
    hpatches_deploy_smoke.add_argument('--latency-warmup', type=int, default=20)
    hpatches_deploy_smoke.add_argument('--latency-repeats', type=int, default=100)

    hpatches_report = sub.add_parser('hpatches-report')
    hpatches_report.add_argument('--evaluation', default='outputs/hpatches_synthetic_raw_v1/full_step096000')
    hpatches_report.add_argument('--output')
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
    if args.command == 'hpatches-preflight':
        from .hpatches import preflight_hpatches
        try:
            report = preflight_hpatches(args.dataset_root, args.integrity, verify_hashes=args.verify_hashes)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            print(json.dumps({'ready': False, 'reason': str(error)}))
            raise SystemExit(1) from error
        print(json.dumps(report, allow_nan=False))
        return
    if args.command == 'hpatches-manifest':
        from .hpatches import create_manifest, manifest_digest
        try:
            manifest = create_manifest(args.dataset_root, args.integrity, args.output,
                                       verify_hashes=args.verify_hashes)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            print(json.dumps({'ready': False, 'reason': str(error)}))
            raise SystemExit(1) from error
        print(json.dumps({'manifest': args.output, 'sha256': manifest_digest(manifest),
                          'sequences': manifest['sequence_count'], 'images': manifest['image_count'],
                          'pairs': manifest['pair_count'], 'condition_pairs': manifest['condition_pair_count']}))
        return
    if args.command == 'hpatches-report':
        from .hpatches_report import write_hpatches_report
        try:
            report = write_hpatches_report(args.evaluation, args.output)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            print(json.dumps({'ready': False, 'reason': str(error)}))
            raise SystemExit(1) from error
        print(json.dumps(report, allow_nan=False))
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
    elif args.command == 'hpatches-cache':
        from .hpatches import build_cache
        from .noise import CanonELD
        from .sensor import load_invisp
        generator = load_invisp(args.invisp_repo, device)
        report = build_cache(args.manifest, args.cache, generator, CanonELD(args.eld_calibration), device,
                             sequence_names=args.sequences, force=args.force)
    elif args.command == 'hpatches-evaluate':
        from .hpatches import evaluate_hpatches
        from .export import DEPLOYMENT_FORMAT, load_export
        from .model import RawFeatureExtractor
        from .teacher import load_teacher
        stored = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if stored.get('format') == DEPLOYMENT_FORMAT:
            model = load_export(args.checkpoint, device)
        else:
            model = RawFeatureExtractor().to(device).eval()
            model.load_state_dict(stored['student'], strict=True)
        superpoint = load_teacher(args.superpoint_repo, device)
        report = evaluate_hpatches(
            args.manifest, args.cache, args.output, model=model, superpoint=superpoint,
            checkpoint_path=args.checkpoint,
            superpoint_checkpoint_path=superpoint.rawfeat_checkpoint_path,
            sequence_names=args.sequences, max_pairs=args.max_pairs,
            target_indices=args.target_indices,
            batch_images=args.batch_images, command=' '.join(sys.argv),
        )
    elif args.command == 'hpatches-deploy-smoke':
        from .export import load_export
        from .hpatches import compare_deployment_smoke
        from .identity import check_schema
        from .model import RawFeatureExtractor
        stored = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        check_schema(stored)
        training_model = RawFeatureExtractor().to(device).eval()
        training_model.load_state_dict(stored['student'], strict=True)
        deployment_model = load_export(args.deployment, device)
        report = compare_deployment_smoke(
            args.manifest, args.cache, args.output,
            training_model=training_model, deployment_model=deployment_model,
            checkpoint_path=args.checkpoint, deployment_path=args.deployment,
            sequence_names=args.sequences, batch_images=args.batch_images,
            latency_warmup=args.latency_warmup, latency_repeats=args.latency_repeats,
            command=' '.join(sys.argv),
        )
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
    elif args.command == "export":
        from .export import export_checkpoint
        report = export_checkpoint(args.checkpoint, args.output, cache_root=args.cache,
                                   manifest=args.manifest, device=device, max_images=args.max_images,
                                   benchmark_repeats=args.benchmark_repeats, warmup=args.warmup,
                                   overwrite=args.force)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
