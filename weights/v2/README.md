# RawFeat V2 weights

V2保留 **27000、28000、29000、30000** 四个训练结果，不指定单一优选权重。网络结构和loss公式沿用V1；从[V1/96000](../v1/checkpoint_step_096000.pt)初始化，使用MegaDepth:COCO=6:4、两侧独立对数均匀噪声、最大ratio225，联合训练30000次更新。30000是训练预算终点。

| Checkpoint | HPatches原五噪声平均 | clean | ratio150 | ratio225 |
| --- | ---: | ---: | ---: | ---: |
| V1初始化参考 | 48.608% | 64.131% | 23.948% | 18.123% |
| [checkpoint_step_027000.pt](checkpoint_step_027000.pt) | 47.645% | 63.002% | 25.781% | 19.167% |
| [checkpoint_step_028000.pt](checkpoint_step_028000.pt) | 47.661% | 63.931% | 23.952% | 20.889% |
| [checkpoint_step_029000.pt](checkpoint_step_029000.pt) | 47.565% | 63.869% | 25.402% | 19.537% |
| [checkpoint_step_030000.pt](checkpoint_step_030000.pt) | 47.117% | 62.906% | 26.448% | 18.581% |

以上为完整HPatches合成Raw评估，每档116序列、580对图像，指标为H-AUC@5。原五噪声指ratio1/4/16/64/100，150/225单独报告。V1参考是V2的step0，其student参数和BN与发布的V1/96000逐项相同。四个V2结果体现强噪声收益与原档性能回退的取舍，尚未全面优于V1。训练固定验证选择的`best_geometry.pt`仍是初始化权重；HPatches结果仅作独立比较。

[manifest.json](manifest.json)记录四个文件的SHA-256、大小、训练step/游标/身份、严格加载检查、固定验证和HPatches结果。文件是原训练checkpoint的逐字节副本，包含未融合student、gray头、BN、AdamW、调度及RNG。当前模型参数量含gray为673349；推理入口的数值域约定见[资产及输入约定](../../docs/asset_provenance.md)。

## 加载

从仓库根目录，在项目Python环境中执行：

```python
import torch
from rawfeat.model import RawFeatureExtractor

checkpoint = torch.load(
    "weights/v2/checkpoint_step_030000.pt",
    map_location="cpu",
    weights_only=False,
)
model = RawFeatureExtractor()
model.load_state_dict(checkpoint["student"], strict=True)
model = model.eval().to("cuda")
```

四个文件使用同一加载方式。精确续训需要原run目录、日志、配置、数据和环境身份，流程见[V2训练手册](../../docs/v2_training_runbook.md)。发布文件保留完整训练状态，部署融合/ONNX导出需要另行生成并验证；本次未宣称四个权重均完成部署验证。

## 重跑HPatches

先按[项目环境与数据准备](../../README.md#7-环境配置)准备完整HPatches、InvISP、ELD和SuperPoint资产，再在项目环境执行。GPU编号按实际资源设置：

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m rawfeat.cli hpatches-cache --cache outputs/v2_evaluation/hpatches_cache

python -m rawfeat.cli hpatches-evaluate \
  --checkpoint weights/v1/checkpoint_step_096000.pt \
  --cache outputs/v2_evaluation/hpatches_cache \
  --output outputs/v2_evaluation/full/step_000000/legacy
python -m rawfeat.cli hpatches-v2-extra \
  --checkpoint weights/v1/checkpoint_step_096000.pt \
  --cache outputs/v2_evaluation/hpatches_cache \
  --output outputs/v2_evaluation/full/step_000000/extra

for step in 027000 028000 029000 030000; do
  python -m rawfeat.cli hpatches-evaluate \
    --checkpoint "weights/v2/checkpoint_step_${step}.pt" \
    --cache outputs/v2_evaluation/hpatches_cache \
    --output "outputs/v2_evaluation/full/step_${step}/legacy"
  python -m rawfeat.cli hpatches-v2-extra \
    --checkpoint "weights/v2/checkpoint_step_${step}.pt" \
    --cache outputs/v2_evaluation/hpatches_cache \
    --output "outputs/v2_evaluation/full/step_${step}/extra"
done
python scripts/compare_hpatches_checkpoints.py --root outputs/v2_evaluation/full
```

缓存、输出目录使用新路径；已有完整结果可以直接交给汇总脚本，无需重新推理。汇总保持五档和两档分数独立，并按序列配对bootstrap输出区间，未做多重比较校正。

## 结果与来源

- [发布结果与复核入口](../../docs/results/v2/README.md)：四个V2结果、V1初始化参考、固定验证、训练摘要、manifest身份和序列统计。
- [完整HPatches比较](../../docs/results/v2/hpatches/comparison.md)及[机器可读结果](../../docs/results/v2/hpatches/comparison.json)。
- [训练优化方案](../../docs/v2_training_optimization_plan_20261017.md)、[MegaDepth调查](../../docs/megadepth_v2_usage_research_20261017.md)、[实现验证](../../docs/v2_implementation_validation_20261017.md)、[梯度诊断修复](../../docs/v2_probe_repair_20261024.md)。

数据集、生成的训练索引/缓存、第三方资产和完整逐步日志保留本地。`v2-prepare`生成`manifests/v2/`运行索引；发布结果目录另存原始固定验证清单及数据来源哈希。历史文档中step0和10k用于初始化/课程检查，V2发布权重仅为本页列出的四个。
