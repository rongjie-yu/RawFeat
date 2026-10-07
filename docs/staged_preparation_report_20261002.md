# RawFeat 分阶段准备交付

**代码准备与有限短训完成，全量未启动。** 没有启动、后台排队、自动延长或定时安排完整训练。正式输出目录尚不存在；由用户按文末命令接管。

## 唯一实现与清理

正式从随机初始化开始，100000更新：A50000检测+gray，B5000渐进匹配，C45000联合。只保留mass检测（Bernoulli有无项+教师质量归一化条件位置项，系数1），正常BN，gray10/.01固定，ratio最大100。S1/S2/S3和三头、InvISP/full-DN/独立CUDA噪声流/RGGB/坐标/mask/teacher/descriptor插值/MeanAD数学保持。LR首步3e-6、warmup2000达到3e-4、保持至50000，随后cosine到100000的3e-6。A描述子参数/BN/grad/Adam暂停，所有参数始终注册；B/C正常恢复。

`configs/formal.yaml`与`configs/smoke.yaml`共享一个schema、调度、loss和训练器；smoke只压缩边界、日志频率和规定验证子集。旧KL选择器、冻结/重估BN、calibration/short配置与历史实验driver移除。共删除48项：33个历史脚本、10个docs候选Python、3个旧配置、rawfeat/bn.py、BN实验测试；没有另存旧Python为第二条执行路线。历史Markdown、outputs、数据、官方依赖、缓存和基线保留。没有git add或提交。

[清理前120项路径/SHA](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/before_inventory.json)；[48项删除及理由/SHA](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/deletions.json)；[保护资产复核](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/protected_assets_recheck.json)。保护的10个上游/模型数学源码SHA未变；旧outputs43625文件、官方依赖258文件、数据163965文件的路径/大小清单未变，指定权重/标定/缓存/基线SHA逐一未变。此目录清单校验不是所有125GB历史文件内容的全量SHA。

50000阶段比例与gray10是本次明确工程选择，尚未证明最优。SuperPoint论文200000次检测预训练不证明RawFeat50000等价或最优。

## 简化、OCR审查与复查

实际应用 [code-simplifier技能](/homes/rongjie/.codex/skills/code-simplifier/SKILL.md) 和 [open-code-review-delegate技能](/homes/rongjie/.agents/skills/open-code-review-delegate/SKILL.md)，按Python风格整理。共享update_schedule、配置校验、原子JSON输出和best选择，移除兼容/重复/旧driver；保留清晰线性数据流。[简化记录](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/code_simplifier.md)。

OCR v1.12.11实际执行delegate preview/rule --format json，主代理读取untracked完整源码并按规则与完整方案审查，无外部LLM。原方案全文读完，OCR使用2504字符忠实摘要（<8000），没有截断关键要求。CLI提示git2.34.1低于建议2.41及背景建议2000字符，但命令成功返回完整JSON；没有unknown --format或缺失CLI。

OCR总29项，reviewed28、skipped1（历史docs/review_findings.json）；reviewed覆盖率96.5517%，逐项accounted100%。补充覆盖13个被默认排除的测试、5个当前文档/依赖以及48个删除，共95项：reviewed94、历史skipped1，当前可执行代码/配置/测试/删除审查覆盖100%。official/generated/history逐文件排除理由见清单。

[最终逐(path,status)覆盖](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/review_final.json)；[最终findings与修复历史](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/review_findings_final.json)；[实际preview](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/ocr_preview_final.json)；[实际rules](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/ocr_rules_final.json)。

确认问题全部修复并重新审查：残留docs/losses.py；完整B/C检测诊断源数；post-step故障真实更新账本；导出离散核查/延迟范围；best/last分析；早期曲线横轴与raw灰度可读性；最近3普通checkpoint轮换；FP32融合错误的绝对几何分数guard。无未解决正确性/训练/恢复/维护问题。短训前通过审查，短训及导出发现的问题另修复、测试、复查，没有沿用修改前结论。

## 准确验证命令与结果

公共前缀（本次物理GPU1，进程内cuda:0）：

```bash
cd /homes/rongjie/projects/RawFeat
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export RAWFEAT_PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
```

```bash
"$RAWFEAT_PY" -m pytest -q tests
```

最终54 passed，12.28s，10个第三方弃用warnings（colour-demosaicing/scipy、thop/distutils、官方InvISP torch.qr/lu）。最后仅调整曲线平滑图例后，`"$RAWFEAT_PY" -m pytest -q tests/test_analysis.py`通过，并重生成实际smoke曲线。`"$RAWFEAT_PY" -m compileall -q rawfeat`退出0。覆盖数学平均/梯度、极端logits、gray不等mask、descriptor半像素、噪声独立流/负值/DN、坐标/mask、A暂停BN/Adam/decay、阶段/LR/noise边界、逐参数恢复计数与身份、best选择、轮换、绘图、融合数值边界。CPU合成数学测试不计实训更新；GPU数据前反向测试没有optimizer.step。

[最终测试日志](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/tests_delivery.log)；[最后绘图测试](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/test_analysis_final.log)；[命令历史](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/commands.md)。preflight/train/validate/diagnose/export/analyze --help逐一退出0；正式命令仅parse校验未执行，[参数核验](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/manual_commands_parse_checked.json)。

实际唯一训练命令：

```bash
"$RAWFEAT_PY" -m rawfeat.cli train --config configs/smoke.yaml --max-updates 300
"$RAWFEAT_PY" -m rawfeat.cli train --config configs/smoke.yaml --resume outputs/checks/staged_preparation_20261002/smoke/latest.pt
```

总配置始终600，A300/B100/C200，warmup20，noise30..240，未修改total恢复。两段各300，前300没有重做。训练日志 [A0→300](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/train_to_300.log)、[恢复300→600](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/resume_to_600.log)。

终点CLI实际运行（全部0更新）：

```bash
"$RAWFEAT_PY" -m rawfeat.cli validate --checkpoint outputs/checks/staged_preparation_20261002/smoke/latest.pt --output outputs/checks/staged_preparation_20261002/final_validation --baseline-summary outputs/checks/detection_followup_20261002/baseline_rngsplit/summary.json --max-images 32
"$RAWFEAT_PY" -m rawfeat.cli diagnose --checkpoint outputs/checks/staged_preparation_20261002/smoke/latest.pt --output outputs/checks/staged_preparation_20261002/final_diagnostics --max-images 32
"$RAWFEAT_PY" -m rawfeat.cli export --checkpoint outputs/checks/staged_preparation_20261002/smoke/latest.pt --output outputs/checks/staged_preparation_20261002/export_verified/model.pt --max-images 32 --benchmark-repeats 50
"$RAWFEAT_PY" -m rawfeat.cli analyze --run outputs/checks/staged_preparation_20261002/smoke
```

validate完整复现训练终点geometry buckets，diagnose summary与训练终点完全一致，export_verified通过逐对数值边界核查，analyze完成全部PNG/PDF与报告。32源/192对明确是子集，没有冒充1536。

## 实际更新与完整恢复身份

|项目|真实学生优化器更新|
|---|---:|
|从随机初始化0→300|300|
|同状态恢复300→600|300|
|pilot、GPU恢复对照、数值/只读前向预检|0|
|本次新增合计/上限|600/1000|
|历史单列|9209|
|生命周期合计|9809|

[实际账本/最终审计](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/smoke_audit.json)；[优化器逐更新journal](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/optimizer_updates.jsonl)；[600行原始scalars](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/scalars.jsonl)；[完整run_manifest](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/run_manifest.json)；[实际配置](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/resolved_config.json)。

实际训练identity `dfd52db20506937e3be1d35c7201bab2e5e1a6a0a630a020076798f700819a4a`，最终完整checkpoint SHA `63800014928977a6c64067c18046987553ed0900a7a4a6bfb9f531338f6100fd`。完整state保存BN、Adam、step600、cursor9600、下次调度、Python/NumPy/torch CPU/CUDA RNG、config/源/代码/依赖/资产身份、clip计数和验证摘要。

A300描述子参数/BN与初始完全一致、grad=None、Adam0；共享/检测/gray Adam300。恢复student/descriptor/optimizer/RNG/cursor/scheduler逐hash相同；预重建的第301步4micro输入hash `a3950949f8c57f3d08d786c5ce592152579acd7c347c922ce3b02a4c84a4ec90`与真实训练相同。第301步match.01、lr.0002999918576507785、ratio cap100，没有隐含预热/Adam重置。终点共享/检测/gray Adam600、描述子300，BN forward counters分别2400/1200。所有模块参数实际改变、所有loss/梯度有限。[恢复预检](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/resume_precheck.json)、[A边界](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/boundary_000300.json)、[最终边界](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/boundary_000600.json)。

训练无OOM/非有限/恢复丢失。每次成功更新保留小型可靠状态副本，故障保存最后可靠完整state与请求/真实成功计数；不将部分BN状态冒充可靠更新前状态。灾难性进程/系统断电只能从磁盘完整checkpoint恢复并据journal记录损失；本次没有触发此路径。

600完成后仅修复analysis绘图、training普通checkpoint轮换及export数值审计。loss/schedule/model/optimizer/data数学未改。原manifest/checkpoint身份保留，没有改SHA标签或放宽trainer身份检查；当前正式从头会创建新的代码身份，旧smoke已完成且不得作为正式初始化。[修复前后代码SHA](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/post_smoke_code_revision.json)保存轨迹身份与交付源码差异。最终CLI分析/验证/诊断/导出用实际600checkpoint再次通过，无额外训练。

## 分项、几何与曲线

|阶段终点|det|occupancy|position|gray原始|match原始|match系数|total|
|---|---:|---:|---:|---:|---:|---:|---:|
|A300|2.054225|0.138059|1.916166|0.0006604|null|0.0|2.060829|
|B400|2.009304|0.140555|1.868749|0.0004774|1.1839996874332428|1.0|3.198077|
|C600|1.942572|0.135704|1.806868|0.0005817|0.9567386209964752|1.0|2.905128|

终点192对主分数H-AUC@5五噪声等权=17.0131%，同192对MeanAD+SP=60.9793%；完整历史1536基线为65.6030%，不能把这两组分数混用。A无随机描述子几何，best_det/best_geometry在smoke都不伪造完整协议选择；full256 best选择规则已实现且测试，best_geometry仅C。

|bucket|teacher1px|teacher3px|实际点数/视角|oracle Top1|gray加权MSE|gray普通MSE|edge梯度MSE|H-AUC@5|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clean|3.996%|16.375%|257.8|85.60%|0.0002626|0.0012965|0.0044096|23.916%|
|1|3.970%|16.245%|257.4|85.03%|0.0002634|0.0013006|0.0044156|19.052%|
|4|3.998%|16.105%|256.5|83.54%|0.0002681|0.0013115|0.0044322|22.782%|
|16|3.894%|16.046%|252.0|81.52%|0.0002975|0.0013800|0.0045131|19.271%|
|64|3.579%|14.089%|232.4|73.39%|0.0004932|0.0016700|0.0049937|11.939%|
|100|3.034%|12.182%|209.5|65.84%|0.0009029|0.0021827|0.0056482|12.022%|

oracle仅已知教师/均匀位置检索，不能替代自动点几何。空集null；actual ratio和bucket分开；teacher1px不是H-AUC@1。终点描述子offdiagonal cosine约.018–.024，实际点数约209–258/视角，没有明显输出塌缩。短训分数不证明长阶段优劣、gray10最优或共享特征充分降噪。gray普通误差/边缘/抗噪只作诊断；未自动调权或追加训练。

[逐view诊断](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/diagnostics/step_000600/per_view.jsonl)；[自动点逐pair](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/validation/step_000600/per_pair.jsonl)；[完整四micro梯度/Adam探针](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/probes/step_000600.json)。探针均0更新，eval/gradient使用副本并恢复RNG；gray S3直接梯度为0，B/C描述子梯度有效。

|曲线|PNG|PDF|
|---|---|---|
|各项原始/加权loss|[PNG](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/losses.png)|[PDF](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/losses.pdf)|
|阶段/lr/noise/clip/耗时|[PNG](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/schedule.png)|[PDF](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/schedule.pdf)|
|1px/3px/固定K/背景/相位/几何|[PNG](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/detection_geometry.png)|[PDF](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/detection_geometry.pdf)|
|gray加权/普通/边缘/抗噪|[PNG](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/gray.png)|[PDF](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/gray.pdf)|

raw与mean20分开标识，平滑不跨阶段；300/400边界明确；早期x轴仅已完成更新。训练期间诊断/边界自动刷新，结束自动生成[分析报告](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/report.md)与[可分析summary](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/smoke/analysis/summary.json)，记录best/last/基线、逐pair失败与各噪声指标；smoke验证此接口。

## 融合检查与性能

首次export的旧绝对几何guard失败，产物与调查保留在export/及export_discrete_investigation.json，没有删掉失败或改基线。重新实现逐192case dense与公共点descriptor检查atol/rtol1e-4，NMS/阈值/TopK/排序/MNN只能在实测FP32差异界内改变。

精确计数：192中19对点顺序改变，**其中18对只排序、1对clean邻像素NMS集合改变**；只有该clean case匹配坐标集合改变。主噪声分数变化来自相同坐标匹配集的输入顺序改变、固定seed的RANSAC对索引顺序敏感。原17.01314%、融合17.07344%、差+0.060299pp。不宣称几何逐位相等，也没有修改NMS或RANSAC数学。

所有case logits最大差1.02519989e-05、dense descriptor最大差2.66730785e-06；公共点descriptor最大差7.71135092e-07；decoded score最大差1.06170774e-07。数值边界校验通过。给出的离散容差为2×实测score误差+FP32 epsilon；NN gap为2×实测affinity误差+8×epsilon。

[可用导出模型](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/export_verified/model.pt)；[导出报告](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/export_verified/export_report.json)；[逐pair融合边界](/homes/rongjie/projects/RawFeat/outputs/checks/staged_preparation_20261002/export_verified/fusion_per_pair.jsonl)。

|项目|实测|
|---|---:|
|融合参数（删除gray）|595105|
|MACs/batch1|4033536000|
|网络 p10/median/p90|0.7458/0.7496/0.7577 ms|
|完整Bayer预处理+网络+NMS+descriptor提取 p10/median/p90|2.3749/2.4067/2.5248 ms|

5090、FP32、batch1、4×240×320、warmup10、重复50。完整提取延迟不含训练InvISP/noise/teacher及RANSAC。物理GPU当前其他任务负载可能改变耗时。

## 环境、预检与100000耗时估计

解释器`/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python`，Python3.12.13，PyTorch2.12.1+cu130/CUDA13.0，NVIDIA driver580.178.04，物理GPU1 RTX5090 32GB（进程内cuda:0），FP32/TF32关闭。固定依赖及官方源码/权重SHA见run_manifest。训练峰值allocated约1.46GiB，最终preflight空闲约30.85GiB；磁盘8.0TB可用。未干预其他GPU任务。

preflight ready：训练118287源、完整cache1536、manifest `d8248fb8ee148609e3d7898dab5044745c27c1d0cd7f4fc9d396494bcab0be0b`、cache identity `d950a2368a67a363bf28ed8fe91029974eac05ede17225020d5fe2c0c0b1757d`、基线65.60296854%。[最终preflight](/homes/rongjie/projects/RawFeat/.planning/2026-10-02-rawfeat-unique-staged-preparation-202610/preflight_final.json)。

|阶段|实测平均秒/更新|pairs/s|正式更新数|估计小时|
|---|---:|---:|---:|---:|
|A|2.8225|5.669|50000|39.20|
|B|3.6764|4.352|5000|5.11|
|C|3.4927|4.581|45000|43.66|

优化器更新部分合计约87.97小时。实测A16源诊断4.29–4.68s、A32源验证8.68s，B400/C600的32源诊断+geometry分别62.39/51.03s，C16源诊断21.35s。按源数线性放大，25次A完整验证+26次B/C完整验证与中间49次诊断粗估额外约3.9h；这种放大未实际跑完整1536，RANSAC与I/O不保证线性。

存盘每次实测0.025–0.070s，正式100次常规/边界checkpoint粗估数秒级I/O。21次完整梯度探针、曲线/报告刷新、初始化、设备共享负载和最终额外validate/export没有全部单独计时，另外增加成本。训练态可靠小副本开销已在更新耗时中。可规划约92h再加这些额外成本；这是基于一次smoke的容量估计，不是精确完成时间承诺。

## 用户手动命令（未执行）

先选择可用物理GPU，并在同一环境、源码、config和output继续恢复。下例使用本次核验的GPU1；正式run用独立目录，没有smoke checkpoint、skip-validation或压缩阶段。

```bash
cd /homes/rongjie/projects/RawFeat
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export RAWFEAT_PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
export RAWFEAT_RUN=/homes/rongjie/projects/RawFeat/outputs/formal_staged_100k_20261002
export RAWFEAT_CACHE=/homes/rongjie/projects/RawFeat/outputs/checks/detection_followup_20261002/validation_cache_rngsplit
export RAWFEAT_BASELINE=/homes/rongjie/projects/RawFeat/outputs/checks/detection_followup_20261002/baseline_rngsplit/summary.json

# 1. GPU/环境与资产预检（只读）
nvidia-smi
"$RAWFEAT_PY" -m rawfeat.cli preflight --config configs/formal.yaml

# 2. 从随机初始化完整100000；仅你手动执行
"$RAWFEAT_PY" -m rawfeat.cli train --config configs/formal.yaml --output "$RAWFEAT_RUN"

# 3. 中断后用同run/config/output的最新完整状态（与2二选一）
"$RAWFEAT_PY" -m rawfeat.cli train --config configs/formal.yaml --output "$RAWFEAT_RUN" --resume "$RAWFEAT_RUN/latest.pt"

# 4. 训练完成：完整256源/1536对验证、诊断、融合导出、曲线/报告
"$RAWFEAT_PY" -m rawfeat.cli validate --checkpoint "$RAWFEAT_RUN/latest.pt" --cache "$RAWFEAT_CACHE" --baseline-summary "$RAWFEAT_BASELINE" --output "$RAWFEAT_RUN/final_validation" --max-images 256
"$RAWFEAT_PY" -m rawfeat.cli diagnose --checkpoint "$RAWFEAT_RUN/latest.pt" --output "$RAWFEAT_RUN/final_diagnostics" --max-images 256
"$RAWFEAT_PY" -m rawfeat.cli export --checkpoint "$RAWFEAT_RUN/latest.pt" --cache "$RAWFEAT_CACHE" --output "$RAWFEAT_RUN/export/model.pt" --max-images 256 --benchmark-repeats 50
"$RAWFEAT_PY" -m rawfeat.cli analyze --run "$RAWFEAT_RUN"
```

启动命令训练JSONL、TensorBoard、validation、diagnostics、probes、checkpoints及analysis都在RAWFEAT_RUN。边界继续last；不自动回滚best_det。正式best_det和best_geometry留在同目录，最终也可用相同validate参数检查best_geometry.pt并对比last。

**代码准备与有限短训完成，全量未启动。** 到此停止训练工作，未创建后台/定时/自动任务。
