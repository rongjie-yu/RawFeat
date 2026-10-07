# RawFeat v1 weights

First trained version, from the completed 100000-update run with detector+gray pretraining (50000), matching ramp (5000), and joint training (45000). The preferred model is the **96000-update C-phase checkpoint**, selected by the predefined five-noise validation score rule.

| Checkpoint | Role | Five-noise H-AUC@5 | Clean H-AUC@5 |
|---|---|---:|---:|
| [checkpoint_step_096000.pt](checkpoint_step_096000.pt) | Preferred C-phase best | 68.0846% | 87.9471% |
| [checkpoint_step_100000.pt](checkpoint_step_100000.pt) | Final budget endpoint | 67.7004% | 86.9108% |
| [checkpoint_step_055000.pt](checkpoint_step_055000.pt) | Matching-ramp boundary reference | 68.2572% | 87.1606% |

MeanAD+SuperPoint scores 65.6030% on the same fixed 256-source/1536-pair protocol. The 55000 checkpoint is the highest observed score across all recorded phases; it is a B-phase boundary and was excluded from the predefined C-phase best selection. These results do not establish that the stage allocation or gray10 is optimal, or that staged training outperforms equally budgeted joint training.

[manifest.json](manifest.json) records each file's SHA256, size, step, training identity, and validation score. The files are byte-for-byte copies of the original **full, unfused training checkpoints**, containing student weights, the auxiliary gray head, Adam state, and RNG. They can be loaded with the current `RawFeatureExtractor`; they are separate from the fused format consumed by `load_export`.

Load the preferred model from the repository root:

```python
import torch
from rawfeat.model import RawFeatureExtractor

checkpoint = torch.load(
    "weights/v1/checkpoint_step_096000.pt",
    map_location="cpu",
    weights_only=False,
)
model = RawFeatureExtractor()
model.load_state_dict(checkpoint["student"], strict=True)
model = model.eval().to("cuda")
```

Use `rawfeat.inference.extract_bayer` with exposure-normalized Canon Bayer sensor DN, following the [input-domain contract](../../docs/asset_provenance.md). The gray head is training supervision; a deployment copy removes it and fuses RepVGG. Formal-run fusion/latency validation has not been performed for this version; use the existing `python -m rawfeat.cli export --help` interface for a separately validated export.

The [full training analysis](../../docs/formal_training_analysis_20261007.md) includes stage curves, detector/gray limitations, and source-paired uncertainty estimates. [Result snapshots](../../docs/results/v1/) retain the run identity, configuration, summary statistics, curves, and baseline/55000/96000/100000 per-pair validations. Datasets, validation tensors, third-party assets, and full per-update logs remain local; their identities and asset-fetch instructions are recorded in the repository.
