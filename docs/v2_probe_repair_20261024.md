# V2 第1000步梯度诊断修复

本文记录2026-10-24的故障修复与短续训状态。后续30k训练及完整HPatches评估均已完成，正式发布结果见[结果归档](results/v2/README.md)。

正式训练于2026-10-18 05:03（北京时间）完成第1000次更新后，在`gradient_probe`抛出`accumulated task gradients do not reconstruct total`。`recovery.pt`保存了全部1000次更新、cursor16000、88组Adam状态、调度及RNG；无需重训这1000步。

## 重现证据

使用原始源码、原恢复权重和第1000步的四个微批输入，在RTX5090/CUDA/FP32上重现相同错误。组合输入SHA-256与原scalar逐位相同：`6574bc2421e18f1742387fea87b1b588312b0044df35d21ff3b506266119e66d`。

| 项目 | 观察值 |
| --- | --- |
| 分项梯度之和与总损失梯度的相对L2偏差 | 1.509372941654874e-5 |
| 原硬断言阈值 | 1e-5 |
| 单元素最大绝对偏差 | 3.5241246223449707e-6 |
| det / gray / match / total梯度范数 | 0.41052 / 0.07020 / 1.21090 / 1.28325 |
| 非有限梯度 | 无 |
| 探针前后原模型和Adam计数 | 未变；Adam均为1000 |

数学等价的分别反传与联合反传，在FP32下受运算顺序影响，不能把1e-5偏差当作训练失败的充分条件。[PyTorch数值精度说明](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)明确说明这类等价计算不保证逐位一致；[grid_sample说明](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.grid_sample.html)也记录CUDA反传的非确定性。此次没有把偏差归因到某一个具体内核。

## 修复范围

- 有限梯度重构偏差超过原阈值时，记录`gradient_sum_relative_l2`、`gradient_sum_absolute_l2`、`gradient_sum_tolerance`、`gradient_sum_within_tolerance`并发出运行时告警，继续训练。
- 每个分项及独立total梯度出现NaN/Inf时仍抛出异常；全零梯度的偏差记为0，避免0/0。
- 探针仍在模型副本上运行，不将诊断梯度用于优化器。原训练loss/backward、网络结构、学习率、ratio课程、数据配比均未改变。
- 新增显式`--allow-probe-fix`，仅授权固定完整源码哈希的单向迁移，并在恢复审计中记录。其余身份检查保持严格；历史GPU迁移登记未改。

## 验证

修复后先完成`open-code-review-delegate`审查，再运行回归和CUDA验证。OCR共21个可审查条目全部登记：5个本次修复文件已审查，16个既有无关改动注明跳过；另补充审查2个测试和2个文档。没有未解决问题，完整记录在证据目录的`ocr_review.json`。

| 验证 | 实际结果 |
| --- | --- |
| 完整测试 | 114 passed，14个既有弃用警告，21.74秒 |
| 原第1000步输入的CUDA重放 | 通过；相对偏差1.4608951994564121e-5，告警及JSON均记录，原模型未变，Adam仍为1000 |
| 正式run短续训 | 从1000到1005完成5次更新，80对=48 MegaDepth+32 COCO，loss和梯度均有限 |
| 精确恢复审计 | student、descriptor、Adam张量、Torch CPU/CUDA RNG、Python/NumPy RNG、Adam计数、游标、调度8项逐项一致 |
| 更新后检查 | 88组Adam计数均1005；BN计数增加20；cursor16080；峰值分配显存1467162624字节 |
| 原始记录 | 原recovery.pt、run manifest及前1000步scalar/journal内容保留；历史GPU迁移登记未改 |
| 当前恢复点 | `outputs/formal_v2_30k/latest.pt`指向`checkpoint_step_001005.pt` |

所有命令在仓库根目录执行，`RAWFEAT_PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python`；其中日志重定向路径见证据目录。

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RAWFEAT_PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
REPAIR_DIR=.planning/2026-10-17-v2-training-optimization-design/repair_20261024
"$RAWFEAT_PY" -m pytest -q tests
PYTHONPATH=. "$RAWFEAT_PY" -u "$REPAIR_DIR/reproduce_probe.py" before
PYTHONPATH=. "$RAWFEAT_PY" -u "$REPAIR_DIR/reproduce_probe.py" after
"$RAWFEAT_PY" -u -m rawfeat.cli train \
  --config configs/formal_v2.yaml \
  --resume outputs/formal_v2_30k/latest.pt \
  --output outputs/formal_v2_30k \
  --allow-probe-fix --allow-gpu-change --max-updates 1005
PYTHONPATH=. "$RAWFEAT_PY" "$REPAIR_DIR/verify_recovery.py"
git diff --check
```

`before`重放实际在修复源码前运行；以上命令是执行记录，不应在当前修复源码上重跑它并解释为原错误。5步短续训的优化器更新时间合计44.52秒。测试结束已暂停，没有启动余下28995步全量训练，也尚不能报告V2最终效果。

证据目录：`.planning/2026-10-17-v2-training-optimization-design/repair_20261024/`。`before/`保留修复前所有训练源码、原恢复checkpoint和原run manifest；`before_probe.json`保存原错误及数值证据。

## 继续全量训练

在仓库根目录选择有足够显存的RTX5090；此次短测使用`CUDA_VISIBLE_DEVICES=0`，原run manifest保留GPU 3记录，因此之后恢复保留`--allow-probe-fix --allow-gpu-change`两个参数。详见 [训练手册](v2_training_runbook.md#2026-10-24第1000步梯度诊断中断后的恢复)。

```bash
cd /homes/rongjie/projects/RawFeat
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python -m rawfeat.cli train \
  --config configs/formal_v2.yaml \
  --resume outputs/formal_v2_30k/latest.pt \
  --output outputs/formal_v2_30k \
  --allow-probe-fix --allow-gpu-change
```

总预算仍是30000更新，`--max-updates`仅供短测设置绝对停止位置，完整续训不添加它。
