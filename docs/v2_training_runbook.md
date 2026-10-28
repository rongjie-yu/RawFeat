# V2 全量训练步骤

30k训练和完整HPatches评估已于2026-10-28完成；发布的27000、28000、29000、30000步结果见[结果归档](results/v2/README.md)及[权重说明](../weights/v2/README.md)。下文保留数据准备、新训练和历史修复恢复的操作步骤。

V2加载V1/96000完整训练权重，单次训练30000更新；MegaDepth:COCO=6:4，独立对数均匀采样，上限100→225，10000步以后保持225。网络和损失公式沿用V1。配置见 [formal_v2.yaml](../configs/formal_v2.yaml)，方案依据见 [V2训练方案](v2_training_optimization_plan_20261017.md)。

以下命令在RawFeat仓库根目录执行。GPU 3是2026-10-17首次短测所用卡，执行前用`nvidia-smi`确认当前资源；单卡可见的物理GPU在程序内为`cuda:0`。2026-10-24修复验证使用GPU 0，已有run的续训命令见第5节。

```bash
cd /homes/rongjie/projects/RawFeat
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RAWFEAT_PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
```

## 1. 提取已下载的MegaDepth RGB

五个分片已通过已有全量SHA-256审计，不需要重新下载。下面是长I/O任务，只提取JPG，不合并生成额外213GB归档；保留原分片。

```bash
mkdir -p /homes/rongjie/datasets/megadepth/rgb
set -o pipefail
cat /homes/rongjie/datasets/megadepth/archives/MegaDepth_v1.tar.gz.part-{00,01,02,03,04} |
  tar -xzf - -C /homes/rongjie/datasets/megadepth/rgb \
    --wildcards --wildcards-match-slash '*.jpg'
```

要求管道退出0且RGB目录有图像。后面的清单准备会逐图解码、散列、去除精确重复及损坏图。实际全量RGB解压和清单准备由用户执行；短测试仅提取隔离目录中的少量真实MegaDepth图像。

## 2. 准备全量源清单与固定验证清单

```bash
"$RAWFEAT_PY" -m rawfeat.cli v2-prepare \
  --coco-root /homes/rongjie/datasets/coco \
  --megadepth-root /homes/rongjie/datasets/megadepth/rgb \
  --output manifests/v2/sources.json
```

要求COCO train2017完整118287张。MegaDepth采用仓库固定的GlueFactory clean训练白名单（153场景）与本地图像交集，均匀保留10个完整场景，每场景20源作为验证。`0015/0022`及官方其他测试场景不用于训练/选模。清单记录每张MegaDepth源的尺寸和SHA-256；损坏图和重复图不进入训练。精确去重不能代替跨地点的语义近重复检查。

产物为`manifests/v2/sources.json`和同目录`validation.json`。重新运行只接受内容完全相同的已有清单；如有意改变数据，使用新路径并在训练前更新配置，不能在同一run中途换清单。

## 3. 用GPU生成V2验证缓存

```bash
"$RAWFEAT_PY" -m rawfeat.cli v2-cache --config configs/formal_v2.yaml
"$RAWFEAT_PY" -m rawfeat.cli preflight --config configs/formal_v2.yaml
```

全量缓存应有 **3648对**：COCO256源+MegaDepth200源，各8档（clean、1、4、16、64、100、150、225）。原COCO的1536条缓存通过硬链接复用，原文件内容不改；新增缓存单独存放在`outputs/v2_validation_cache`。这一步需要旧COCO验证缓存和同一文件系统。缓存命令可重新执行补齐未完成条目，但代码/资产/manifest身份变化必须使用新缓存目录。

预检应返回`ready: true`、`cached_cases: 3648`。它检查真实数据、缓存和CUDA，不进行优化器更新。保留原V1的配置、权重和缓存。

## 4. 启动30k训练

建议在tmux中运行：

```bash
tmux new -A -s rawfeat-v2
```

在tmux中重新执行本文开头的目录、GPU和Python变量设置，然后启动：

```bash
"$RAWFEAT_PY" -m rawfeat.cli train \
  --config configs/formal_v2.yaml \
  --output outputs/formal_v2_30k
```

首次启动会加载V1/96000，仅导入student参数和BN，重置AdamW、游标、调度和best记录。先执行step0完整验证，保存`initialization.json`、`initial_validation.json`及`checkpoint_step_000000.pt`，随后进行训练。初始化验证期间尚无训练scalar输出。

每更新16对；每5更新精确48 MegaDepth+32 COCO。完整30000更新共480000对，其中288000 MegaDepth、192000 COCO。全程检测/匹配/gray权重1/1/10；骨干峰值LR3e-5，头1e-4。1k warmup后余弦下降，10k达到ratio225。

每1000步保存检查点和诊断，每2000步完整验证，课程边界1k/10k/30k另存。`scalars.jsonl`记录实际两库/场景配额、ratio直方图、loss、LR、梯度和计时，`analysis/`保存曲线。V2选模为七噪声H-AUC@5的0.6 MegaDepth场景均值+0.4 COCO均值，旧COCO五档分数和clean相对step0下降容差各1个百分点；同分保留较早步。`best_geometry.pt`可能保留step0，表示后续没有合格的更高分检查点。

## 5. 中断后恢复

```bash
"$RAWFEAT_PY" -m rawfeat.cli train \
  --config configs/formal_v2.yaml \
  --resume outputs/formal_v2_30k/latest.pt \
  --output outputs/formal_v2_30k
```

保持代码、环境、配置、数据清单及缓存身份一致；总预算始终是30000成功更新。仅同型号GPU映射改变时可加`--allow-gpu-change`。`--max-updates`是绝对暂停位置，不修改总预算或课程。

### 2026-10-24：第1000步梯度诊断中断后的恢复

正式run在2026-10-18完成1000更新后，梯度探针因有限FP32梯度重构偏差超过1e-5而退出。原权重及四个微批输入已重现：相对L2约1.51e-5，单元素最大偏差约3.52e-6。修复后继续记录偏差及告警，NaN/Inf仍报错；网络、损失、数据和优化器更新不变。详见 [修复验证记录](v2_probe_repair_20261024.md)。

此run跨越修复版本恢复时，保留`--allow-probe-fix`；该参数只接受`rawfeat/probe_resume_revision.json`中固定的完整源码哈希迁移，不允许任意源码、配置、数据或依赖变化。此次修复短测使用GPU 0，原run manifest保留GPU 3记录，因此后续恢复此run保留`--allow-probe-fix --allow-gpu-change`两个参数，分别授权源码修复及同型号GPU映射变化。

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$RAWFEAT_PY" -m rawfeat.cli train \
  --config configs/formal_v2.yaml \
  --resume outputs/formal_v2_30k/latest.pt \
  --output outputs/formal_v2_30k \
  --allow-probe-fix --allow-gpu-change
```

无需重新准备清单、重建验证缓存或从V1重新训练。训练总预算仍是30000成功更新。

## 6. 训练完成后的评估

完整V2固定验证：

```bash
"$RAWFEAT_PY" -m rawfeat.cli v2-validate \
  --config configs/formal_v2.yaml \
  --checkpoint outputs/formal_v2_30k/best_geometry.pt \
  --output outputs/v2_evaluation/fixed_best
```

HPatches旧五噪声协议沿用原manifest和几何，使用新缓存目录以匹配当前代码身份。已有HPatches缓存本轮检查均与当前身份不匹配，不覆盖或绕过其校验：

```bash
"$RAWFEAT_PY" -m rawfeat.cli hpatches-cache \
  --cache outputs/v2_evaluation/hpatches_cache

"$RAWFEAT_PY" -m rawfeat.cli hpatches-evaluate \
  --checkpoint outputs/formal_v2_30k/best_geometry.pt \
  --cache outputs/v2_evaluation/hpatches_cache \
  --output outputs/v2_evaluation/hpatches_legacy
```

HPatches新增150/225对称曝光档，从上述同一clean缓存在线生成确定性噪声，RawFeat和基线读取同一输入：

```bash
"$RAWFEAT_PY" -m rawfeat.cli hpatches-v2-extra \
  --config configs/formal_v2.yaml \
  --checkpoint outputs/formal_v2_30k/best_geometry.pt \
  --cache outputs/v2_evaluation/hpatches_cache \
  --output outputs/v2_evaluation/hpatches_extra
```

扩展单独保存逐对/逐序列结果、单图计时及序列bootstrap区间。其两档均值与旧五档均值分别报告；不将它们直接比较为同一个主分数。HPatches只在选定权重后评估，不参与每轮调度或选模。

## 短测试

`configs/smoke_v2.yaml`使用隔离的少量真实MegaDepth图像、两库固定验证及相同网络。压缩到20更新，2步warmup、5步达到225，能覆盖4个完整配额块和恢复边界；它不能证明泛化提升。审查及实际执行结果见 [实现与验证记录](v2_implementation_validation_20261017.md)。恢复状态逐项精确验证，独立CUDA轨迹不保证逐位一致。
