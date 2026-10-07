# 全实现审查、修复与训练前诊断（2026-09-30）

> 文档地位：历史实现复核，只作为证据记录。当前唯一执行方案是 [统一方案与检测排查路线](low_light_raw_feature_extractor_design.md)，本轮授权与顺序见其§17；本文的旧范围、旧协议或旧限制不覆盖当前用户指令。

本报告保留协议v1的审查与复测结果。用户随后确认正式max ratio=100、主验证[1,4,16,64,100]；旧v1 manifest/cache/baseline归档至`outputs/checks/ratio100/protocol_v1_history`。以下旧分数与梯度解释不代表灰度权重10已被实测确认合适。新协议检查产物在`outputs/checks/ratio100`。

## 结论与覆盖

已对照 `low_light_raw_feature_extractor_design.md` 全文审查当前实现。OCR delegate v1.12.11 首次 workspace preview 共31个可审文件，31个 reviewed，0个 skipped，coverage_rate=100%。清单及规则保存在 `outputs/checks/review/ocr_preview.json`、`ocr_rules.json`，逐文件记录在 `.planning/2026-09-30-review-implementation-design/review_checklist.json`。OCR默认排除的7个测试、README、4个设计/审计文档和requirements均补充审查。规划文件属于过程记录，未当成实现要求。新诊断脚本和全部修改再次复查。

修复7组确认问题，详情和技能要求的结构化字段见 [review_findings.json](review_findings.json)。最终复查范围内未发现剩余可确认的代码或设计一致性问题；这不证明不存在未知缺陷，也不等于最终模型质量达标。

| 问题 | 影响 | 修复 |
|---|---|---|
| NMS等分数区域错误抑制 | 默认边界过滤后可整张0点；原测试空数组也通过 | 稳定贪心处理同分候选，检查数量及间距 |
| 无效位置先参与NMS | 无效边界/warp点压掉有效邻居 | 在最大池化前应用边界及valid mask |
| 新训练复用历史目录 | 覆盖保留的检查点、拼接不同运行的日志 | 拒绝覆盖；增加`train --output`；验证resume后才写元数据 |
| 子集学生与全量基线比较 | 分数和差值不可比 | 从逐对基线选择同一子集、重新聚合，失败样本仍计入 |
| 清单仅验证数量 | 重复源、训练集混入、clean/ratio/bucket/seed错误也被接受 | 校验固定协议语义，六种破坏用例回归 |
| 验证允许train模式/空规模 | 更新BN、输出依赖验证batch；空指标出错 | 要求eval，缓存和验证规模限制为1..256 |
| 正式训练允许部分验证 | 预检通过但不能产生best | formal要求全部256图 |

结构、S1/S2/S3宽度与块数、三个头、初始化、FP32、AdamW分组、有效batch16、在线冻结InvISP/官方教师、三项损失公式、每视角/每图对等权平均、三条课程和部署融合均保留设计。数据合成沿用已经确定的full-DN桥接；未引入额外损失、clean描述子分支、噪声条件、自动权重、EMA或训练后BN重估。

## 复测与产物

本次复测保留旧2k校准记录，并将旧基线完整归档至 `outputs/checks/review/baseline_before_review`。NMS/评价修改后重新运行全部1536对基线，并把新基线放回默认 `outputs/baseline`。数据合成代码未改变，因此原始缓存身份不变，无需重算InvISP缓存。

当前完整基线H-AUC@5px为 **60.74274%**；旧值60.73890%属于修改前结果。正式预检再次 `ready=true`，缓存1536对，identity仍为 `bce4f73cdc9f9c5ec1ce84cd5f46ee27671a6b705e73c80a4eaec7fb91a8166e`。

| 当前代码复测 | 结果 | 证据 |
|---|---|---|
| 全套测试 | 28 passed；4条第三方弃用警告 | `outputs/checks/review/tests.log` |
| 编译/依赖/shell | compileall、pip check、bash -n均通过 | 下列命令 |
| 50步完整分辨率/有效batch16 | step50/cursor800，有限模型及损失/梯度，6对末次验证；新增阶段探针有限 | `short50/latest.pt`、`short50/scalars.jsonl` |
| 当前路径恢复 | 连续两步 vs 一步+恢复，RNG/游标/调度一致；模型最大差5.104e-7，优化器矩差2.313e-6 | `resume/report.json` |
| ratio256两对反向 | loss/gradient有限，0次更新，保留负值 | `extreme.json` |
| 完整基线 | 1536对、各桶256对、60.74274% | `outputs/baseline/summary.json` |
| 部分验证比较 | 1图6对学生只比较同1图基线，baseline_images=1 | `subset_comparison/summary.json` |
| 正式预检 | ready=true，缓存/新基线一致 | `preflight.json` |
| 融合/重载/完整验证 | logits/descriptor误差1.049e-5/5.443e-6；重载一致；原始和融合1536对主分数一致为1.358345% | `export/export_report.json` |

表中省略前缀的证据位于 `outputs/checks/review/`。当前共享GPU测得网络0.820ms、完整提取14.843ms，仅用于记录当前运行条件。进一步用完全相同的预计算张量、交替运行旧/新NMS，壁钟中位数为0.890/1.197ms，修复增加约0.307ms；不足以把整个提取流程的差异归因于NMS。此检查仅有46点，最终1024点成本和空闲GPU延迟仍需复测，不能沿用旧报告的1.445ms。

实际执行命令（完整基线在GPU0运行，生成的新目录审计完成后移回默认baseline，旧目录保留；测试最终使用GPU1）：

```bash
PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
CUDA_VISIBLE_DEVICES=1 "$PY" -m pytest -q tests
"$PY" -m compileall -q rawfeat scripts tests
"$PY" -m pip check
bash -n scripts/fetch_assets.sh scripts/run_required_checks.sh
CUDA_VISIBLE_DEVICES=0 "$PY" -m rawfeat.cli baseline --output outputs/checks/review/baseline --max-images 256
CUDA_VISIBLE_DEVICES=0 "$PY" -m rawfeat.cli train --config configs/short_check.yaml --output outputs/checks/review/short50
bash outputs/checks/review/resume/run.sh
CUDA_VISIBLE_DEVICES=1 "$PY" -m scripts.check_extreme --output outputs/checks/review/extreme.json
CUDA_VISIBLE_DEVICES=1 "$PY" -m rawfeat.cli validate --checkpoint outputs/calibration/latest.pt --output outputs/checks/review/subset_comparison --baseline-summary outputs/baseline/summary.json --max-images 1
CUDA_VISIBLE_DEVICES=1 "$PY" -m rawfeat.cli export --checkpoint outputs/calibration/latest.pt --output outputs/checks/review/export/model.pt --max-images 256 --benchmark-repeats 20
CUDA_VISIBLE_DEVICES=1 "$PY" -m scripts.diagnose_checkpoint --checkpoint outputs/calibration/latest.pt --output outputs/checks/review/checkpoint_diagnostic.json
CUDA_VISIBLE_DEVICES=1 "$PY" -m outputs.checks.review.benchmark_nms
"$PY" -m rawfeat.cli preflight --config configs/formal.yaml
```

所有命令在仓库根目录，Python为 `/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python`，GPU通过 `CUDA_VISIBLE_DEVICES` 映射到cuda:0。正式100k训练未启动；本次未重新训练完整2k校准模型。修正后的训练路径由50步、恢复和极端输入检查验证；修正前2k检查点仅用于定位当前训练风险。

## 关键诊断证据

新增 `scripts/diagnose_checkpoint.py`：固定前4对验证图像，分别检查clean/1/16/256；不更新优化器。学生按8视角运行，冻结教师按每对2视角运行。BN批统计比较只在可丢弃副本上执行，不保存统计，不修改检查点；正式指标始终使用eval模式。来源是旧2k校准检查点，证据为 `outputs/checks/review/checkpoint_diagnostic.json`，这是小样本诊断，不是模型选择分数。

| eval条件 | 学生点/视角均值 | 教师点3px召回 | 已知对应位置描述子双向检索 |
|---|---:|---:|---:|
| clean | 33.625 | 2.17% | 93.19% |
| ratio1 | 33.625 | 2.11% | 93.36% |
| ratio16 | 33.625 | 2.04% | 89.84% |
| ratio256 | 274.25（4–1024，波动大） | 3.74% | 12.57% |

同一clean数据教师平均402.875点/视角。教师关键点分数中位数0.10138；相同位置的学生分数中位数0.001243，低于既定0.005阈值。教师/学生平均dustbin概率约0.94271/0.94344，已经接近，但空间峰值尚未接近。总体KL下降不能单独证明检测定位能力已经学会。

clean的eval与批统计诊断点数、KL接近，当前小样本证据不支持把轻噪声低分主要归因于BN模式。ratio256为校准课程之外的强噪声，出现BN模式敏感性和描述子退化，属于要在正式噪声课程过渡时观察的风险；不能据此认定校准失败或直接改网络。

## 优先优化建议

以下是针对当前证据的实验建议，未自动改动正式超参数。经验参数是否采用，应依据受控对照；调参应明确固定项和实验变量，参见 [Google Research Tuning Playbook](https://github.com/google-research/tuning_playbook)。

1. **先验证检测峰值能否学会。** 做独立的固定8–16对、ratio1、固定增强/几何/噪声的100–300步小样本拟合实验；保持网络和三损失。观察教师位置响应、3px召回、点数和KL的同步趋势，再与重新随机采样的数据比较。固定样本拟合仍无峰值改善，应先定位监督/优化信号；固定样本能拟合而随机数据改善慢，才考虑数据难度和样本效率。步数是诊断建议，不是保证收敛的理论界限。
2. **从早期就把检测与描述子拆开诊断。** 每500步对同一固定数据报告点数分位数、教师位置响应/召回、dustbin、已知几何对应的描述子检索率，再报告实际MNN/RANSAC。已有约93%的oracle检索和约2%的教师点召回，优先级应落在检测端。离线阈值扫描可区分分数偏低与峰值位置错误，但不能用更低阈值替换正式0.005协议来掩盖训练问题。
3. **按共享阶段看梯度，再调经验系数。** 本次已增加S1/S2/S3的det/match/gray范数、加权gray范数、S1/S2的det-match与主任务-gray余弦。只检查首层不足以支持整个骨干的结论。若检测梯度长期弱且峰值召回停滞，先做学习率与灰度经验权重的单变量短对照；若要改det/match系数，应同步更新设计与固定配置。可比较gray=3与10、峰值LR=1e-4与3e-4，保持其余条件和总更新数相同。这些是候选点，不是已验证的最优值。
4. **在噪声课程关键节点检查，而不等100k结束。** 每个既有2k正式验证仍跑完整协议；5k、约10k/20k、40k附近另外运行同一固定诊断，观察clean与轻噪声是否遗忘、极端档是否出现点数爆发而匹配精度下降。现有前16图都属于两边同倍率，应补充固定诊断位置128和192附近的一轻一重配对；这些诊断不能替代1536对正式分数，也不构成早停规则。
5. **BN和速度分别验证。** 保持有效batch16，可独立比较4对×4与8对×2，实际BN视角数从8变16。先确认轻噪声/强噪声诊断的变化，再决定是否采用；无需修改BN结构或重估统计。梯度累积不扩大BN统计batch，相关机制见 [Tuning Playbook的BN说明](https://github.com/google-research/tuning_playbook#batch-normalization-implementation-details)。当前服务器GPU均有其他负载，本次速度只能证明检查可运行，不能与旧空闲设备3.4秒/步或毫秒级延迟直接比较。
6. **保留数据统计诊断。** 极端输入有限不证明噪声统计与真实相机一致。按亮度段记录负值/饱和占比、shot/read/row方差与噪声种子，再核对已确定的full-DN曝光桥接与实际入口。保持相机、WB、CFA和标定固定；不要根据少数极端样本自行更换这条已经确认的路线。

不建议仅凭这次早期AUC换结构、提高训练预算或一次性改多个超参数。更有价值的下一步是用固定小样本验证检测峰值学习，再依据阶段梯度和受控试验校准经验参数。

## 复现新增诊断和独立运行

```bash
PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
export CUDA_VISIBLE_DEVICES=1
"$PY" -m scripts.diagnose_checkpoint --checkpoint outputs/calibration/latest.pt --output outputs/checks/my_diagnostic.json
"$PY" -m rawfeat.cli train --config configs/short_check.yaml --output outputs/checks/new_short50
"$PY" -m rawfeat.cli train --config configs/short_check.yaml --output outputs/checks/new_short50 --resume outputs/checks/new_short50/latest.pt
```

短检查只有50步预算；最后一条不会额外训练已经完成的50步，只用于恢复同一状态。真正的中途恢复对照见 `outputs/checks/review/resume/run.sh`。重跑有限检查脚本应设置新的 `RAWFEAT_RUN_ROOT`，防止覆盖已有训练记录。
