# V2 实现与验证记录（2026-10-17）

本文记录2026-10-17的实现与短测状态。30k训练及完整HPatches评估现已完成，正式发布的四个V2结果见[结果归档](results/v2/README.md)。

已实现V2训练支持，正式30k训练尚未启动。网络、损失公式及原合成数学文件保持原内容；数据配比6:4、独立对数均匀ratio100→225、V1/96000初始化、分组学习率及全程联合训练按 [方案](v2_training_optimization_plan_20261017.md) 落实。完整启动命令见 [操作步骤](v2_training_runbook.md)。

## 实现范围

- `data_v2.py`：固定clean场景白名单、JPEG解码及精确去重、整场景留出、源文件大小/mtime快照、按游标生成80对中48/32的配额，MegaDepth内部场景等概率。
- `config.py/schedules.py/training.py/identity.py`：V2配置、加载student与BN但不导入V1优化器、新run与严格resume、骨干0.3倍学习率、七档噪声记录与来源/场景计数。
- `validation_v2.py/diagnostics.py/analysis.py`：旧COCO样本复用、新150/225档、独立MegaDepth验证、场景均值、旧COCO容差选模、完整与子集诊断曲线分开。
- `hpatches_v2.py`：独立150/225扩展，GPU生成ELD后两种方法共用相同输入，单图GPU网络推理，逐对/逐序列结果及bootstrap。旧HPatches五噪声协议保留。
- 新增 `formal_v2.yaml`、`smoke_v2.yaml`、CLI准备/缓存/评估入口及测试。

## OCR审查

使用 [open-code-review-delegate技能](/homes/rongjie/.agents/skills/open-code-review-delegate/SKILL.md)，OCR v1.12.12只负责文件筛选及规则，审查由当前agent执行，未调用外部LLM端点。

实际命令：

```bash
ocr delegate preview --format json --background 'RawFeat V2: unchanged network/loss; V1 step96000 weights only; reset optimizer; 30k joint updates; MegaDepth COCO 6:4, scene disjoint; independent log-uniform ratio100 to225; preserve V1 caches; strict deterministic resume; fixed seven-noise validation; GPU smoke only after review; full training is user-run.'
ocr delegate rule --format json <preview中全部reviewable_files路径>
```

18个reviewable条目逐项登记：13个本次实现文件已审查，5个原有无关下载依赖/脚本/研究记录明确跳过，覆盖率72.22%，条目登记率100%。被默认排除的相关Markdown及测试另行人工审查。逐文件清单、规则和修复记录见本地 [.planning审查JSON](../.planning/2026-10-17-v2-training-optimization-design/ocr_review.json)。没有未解决审查项。

修复了加权失败数变成小数、选模标记未持久化、源文件变更未被预检识别及HPatches基线要求CPU输入的适配问题。保持历史GPU迁移registry内容不变，将旧测试的“未来源码必须始终等于历史哈希”改为固定registry哈希，并新增历史迁移不得授权V2源码变化的拒绝测试。

## 实际验证结果

环境：RTX5090物理GPU3，`CUDA_VISIBLE_DEVICES=3`，PyTorch `2.12.1+cu130`，CUDA13.0，FP32；OMP/MKL各4线程。

| 检查 | 结果 |
|---|---|
| 完整回归 `python -m pytest -q tests` | **97 passed**，24.23秒 |
| HPatches适配修正后的相关回归 | **36 passed**，5.40秒 |
| 原V1正式preflight | ready=true，1536缓存对，原cache identity保持一致 |
| V2隔离清单和GPU缓存 | ready=true，4源×8档=32对 |
| 连续CUDA联合短训练 | 20更新，sample_cursor=320，loss/梯度有限，裁剪0次 |
| 同配置暂停恢复 | 7+13更新，最终20更新；记录连续无重复 |
| 每条短训练来源配额 | MegaDepth192对、COCO128对；4个完整80对块均48/32 |
| 最大噪声范围 | 第5步达到225上限，实际训练采样最大ratio222.3748；固定验证包含精确225 |
| 优化器及BN | 88个参数状态均更新20次；所有BN计数相对V1初始化增加80 |
| 恢复瞬间状态 | 8项审计值与暂停检查点**精确一致** |
| HPatches扩展短验证 | 1个真实图像对×150/225×2种方法完成，8次单图推理，非完整性能评估 |

短测试从原归档首片只提取场景0162/0001各6张真实RGB到隔离目录，耗时33.62秒。训练用0162的6源，整场景0001保留、选2源验证；另外使用2个固定COCO验证源。原COCO训练库118287源均通过文件存在及元数据校验，短训练实际只抽样128对。没有解压全量MegaDepth，也没有执行正式30k优化。

## 恢复一致性的准确含义

实际恢复进程在第7步载入后、执行第8步之前写入的startup审计，与保存的第7步检查点逐项比对。`student_sha256`（含BN）、`optimizer_tensors_sha256`、Torch RNG、Python/NumPy RNG、Adam步数、sample cursor、scheduler和identity SHA共8项精确一致。

两次独立CUDA训练轨迹**不逐位一致**。差异在暂停前的第2步已出现；共同审计步的输入摘要一致，20步total loss最大差为 `5.6624e-7`，最终参数最大绝对差 `2.2754e-5`、相对L2差 `9.5617e-7`。BN等缓冲的最大绝对差为 `0.00110817`；优化器张量最大差 `4.9841e-5`。最初将独立轨迹按极严逐元素容差断言的检查未通过，不能据此宣称训练逐位可复现。

当前描述子采样使用 `features.py` 中的CUDA `grid_sample`。PyTorch 2.12文档明确其CUDA反传可能非确定；观察到的恢复前差异与这一限制相符，但本次未通过算子消融确定唯一数值差异来源。恢复状态的精确性由上述检查点审计单独验证，没有放宽生产恢复身份策略。[PyTorch官方说明](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.grid_sample.html)

原HPatches缓存与当前代码身份不一致，本轮在新路径生成一个序列的小缓存进行测试；全量步骤同样使用新路径，不覆盖或绕过旧缓存身份。HPatches扩展基线输入适配修正在训练/恢复测试后完成；训练数学源码未变，修正后额外GPU评估及36项相关回归通过。

## 可复核命令与证据

以下命令均在仓库根目录执行，Python为 `/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python`，GPU环境同上：

```bash
python -m pytest -q tests
python -m pytest -q tests/test_hpatches.py tests/test_v2_training.py tests/test_gpu_resume.py
python -m rawfeat.cli preflight --config configs/formal.yaml
python -m rawfeat.cli preflight --config configs/smoke_v2.yaml
python -m rawfeat.cli train --config configs/smoke_v2.yaml --output outputs/checks/v2_20261017/continuous
python -m rawfeat.cli train --config configs/smoke_v2.yaml --max-updates 7 --output outputs/checks/v2_20261017/resumed
python -m rawfeat.cli train --config configs/smoke_v2.yaml --resume outputs/checks/v2_20261017/resumed/latest.pt --output outputs/checks/v2_20261017/resumed
python -m rawfeat.cli hpatches-cache --cache outputs/checks/v2_20261017/hpatches_cache --sequences i_ajuntament
python -m rawfeat.cli hpatches-v2-extra --config configs/smoke_v2.yaml --checkpoint outputs/checks/v2_20261017/continuous/latest.pt --cache outputs/checks/v2_20261017/hpatches_cache --output outputs/checks/v2_20261017/hpatches_extra --sequences i_ajuntament --max-pairs 1
python .planning/2026-10-17-v2-training-optimization-design/verify_v2_smoke.py
git diff --check
```

训练命令的输出目录已经存在，以上是本轮实际执行记录；重复短测需换新输出目录。详细证据见 [短测核查JSON](../.planning/2026-10-17-v2-training-optimization-design/smoke_verification.json)、[恢复审计JSON](../.planning/2026-10-17-v2-training-optimization-design/exact_restore_verification.json)；完整日志与原始输出保留在相应planning和outputs目录。

尚未验证的是全量MegaDepth的实际场景/图数、完整3648对缓存构建和正式30k效果。当前短测只确认实现与数据/训练/恢复链路可运行，不证明泛化或最终精度提升。正式训练纯更新耗时还会受数据冷读、GPU负载及完整验证影响，不按某一次短测计时作保证。
