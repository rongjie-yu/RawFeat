# 新会话执行prompt：唯一方案代码清理、审查、有限短训与手动全量命令

请在 /homes/rongjie/projects/RawFeat 完成本任务：把所有自有实现整理到当前唯一方案，清理旧路线，使用指定技能简化与审查，修复并复查通过，实际完成一次有限预算短训，最后交付由我手动启动的完整训练命令。

**你不能启动全量训练、后台排队全量任务或自动延长短训。** 本次我已选择保留分阶段方案；过去禁止未来正式训练、正式默认KL和“阶段未采用”的历史叙述不覆盖当前决定，但启动完整任务仍由我亲自执行。

## 1. 唯一要求与必读文件

完整阅读：
/homes/rongjie/projects/RawFeat/docs/low_light_raw_feature_extractor_design.md

该文件已整体重写为当前正式目标：100000更新，检测+gray50000、匹配渐进5000、联合45000。从随机初始化开始，mass为唯一检测loss、正常BN、gray10全程固定、ratio最大100。

随后读取必要最新证据、asset_provenance和实际代码。历史报告用于理解已确认契约与失败，不作为并列方案；不要重跑历史A–E、BN冻结或250+50+200对照。原正式配置尚未适配，本次必须真正修改它。

必须读取并应用：
- $code-simplifier:code-simplifier
  /homes/rongjie/.codex/skills/code-simplifier/SKILL.md
- $open-code-review-delegate
  /homes/rongjie/.agents/skills/open-code-review-delegate/SKILL.md

本项目是Python，按项目风格应用技能；ES modules/React条款不适用。用户已明确授权整体清理，不只限于局部新增代码，也不要求保留已废弃的功能或兼容旧实验checkpoint。

## 2. 当前唯一实现目标

- 共享骨干S1/S2/S3及现有检测/描述子/gray三头保留；gray直接监督S1/S2，S3间接受益。无新跳连、loss、BN方案或S3灰度连接。
- 检测仅保留Bernoulli有无项+教师质量归一化条件位置项，系数1；删除原65类KL训练选择器。
- gray目标为对应clean线性灰度、现有掩码与加权MSE，weight10/epsilon.01固定。10是本次明确选择，尚未证明最优，不自动调权或衰减。
- 从随机初始化进行A/B/C，不加载历史mass2000或smoke作为正式初始权重。
- 下一次更新k=step_index+1：A为1..50000；B为50001..55000，match=(k-50000)/5000；C为55001..100000，match1。
- A暂停描述子前向/匹配计算、参数和BN，grad=None，Adam状态不推进；B/C恢复正常训练。检测和gray始终反向，所有参数始终在可恢复的优化器结构中。
- LR前2000线性3e-6→3e-4，随后保持到50000，B/C cosine至100000的3e-6；不重置Adam或另外隐式预热。
- noise前5000为含ELD的ratio1，5000..40000按方案cosine-log扩展至100，之后固定loguniform[1,100]；50000接入匹配时noise分布不再变化。
- 保持已确认InvISP/full-DN/独立CUDA噪声流/RGGB/坐标/mask/teacher/descriptor插值/基线数学。4对×累积4、FP32/TF32关闭、AdamW与clip5沿用。

SuperPoint原论文是200000次Synthetic Shapes检测预训练，后适配真实图像并联合。当前50000是RawFeat在10万总预算下的预定工程选择，不得宣称论文证明了该比例或与200000等价。

## 3. 先清理再简化

先建立独立规划目录、当前文件路径/SHA清单和保护资产清单。确认仓库可能大量untracked，不能只看git diff就认定没有改动；不要擅自全仓库git add或提交。

整理唯一运行路径：
- 仅一个当前训练器、loss、阶段调度和配置schema。
- configs/formal.yaml与configs/smoke.yaml只在时长/记录频率上不同，使用同一算法。
- 删除旧KL、冻结/重估BN实验选项、旧calibration/short配置与模式、历史实验driver和报告生成器、docs/files等候选Python副本。
- 从旧诊断中保留当前真正需要的能力，合并到当前诊断/分析模块；不导入旧脚本维持另一条路线，不写旧checkpoint兼容层。
- 保留必要资产准备、验证、MeanAD基线、推理导出和当前行为的测试。移除旧方案专属测试，补齐有意义的数学、暂停、边界、恢复及数据契约测试。
- 历史Markdown、outputs中的结果/配置记录/检查点、数据、官方依赖、缓存和基线保留；它们是证据，不是当前入口。不得将旧Python复制到另一个可执行目录换名保留。
- README、docs入口、命令、tests/imports和配置与当前方案一致，不保留互相冲突的默认值。

保留已验证的上游数学，避免无必要修改数据/噪声/基线文件触发身份变化。若确需改变代码fingerprint，不绕过身份检查、不覆盖旧资产、不随意把旧baseline改标签；依据方案明确验证等价性或创建新身份的配套资产。

实现适配后必须执行code-simplifier，对本次整体修改保持功能一致地简化：清晰、直接、少重复，不增加抽象和兼容负担。

## 4. OCR审查、修复、复查

随后实际执行open-code-review-delegate工作流：
1. 检查ocr版本，运行ocr delegate preview --format json，取得reviewable/excluded清单。
2. 对reviewable文件运行ocr delegate rule --format json。由你读取实际diff或untracked完整源码，应用规则和完整方案人工审查；OCR侧不需要外部LLM。
3. 以(path,status)逐项记录覆盖，不能漏掉新文件、删除和测试。official/generated/history资产明确排除原因；删除前SHA清单弥补untracked删除在git diff中不可见的问题。
4. 覆盖loss平均、坐标/噪声、A/B/C、逐参数Adam步数、BN、梯度累积/clip、数据游标/RNG、保存恢复、配置、验证选择、日志/绘图、异常恢复、性能与遗留入口。
5. 修复所有确认的实质问题，再复查修复及相关上下文，直到无未解决的正确性/训练/恢复/维护问题。测试通过不等于review通过。
6. 输出findings和修复历史、total/reviewed/skipped/coverage与逐文件理由。不得仅凭OCR命令exit0声称审查通过。

严格按技能处理CLI缺失、unknown --format和背景大小限制；完整方案超过8000字符时写忠实短摘要供OCR，仍人工完整阅读原文，不能截掉关键要求。

在清理、简化、修复完成后形成最终审查。短训若发现问题，修复后做相应测试和再次审查，不能沿用修改前通过结论。

## 5. 有限短训与预算

只有实现/数值/恢复预检查和review通过后，实际运行一次从头smoke：
- 总600：A300、B100、C200。
- warm-up20；LR到300保持峰值，301..600 cosine衰减。
- noise30开始扩展、240达到100，其他算法与正式一致。
- 可在300步停止并从同一完整状态恢复剩余300，验证A→B边界；不能重做前300，也不能改total再恢复导致调度变化。

所有真实学生优化器更新，包括pilot、GPU恢复对照和主短训，合计最多1000，写实际账本；历史9209单列。不要额外无依据重开600/2000步，不启动第三组或完整任务。GPU为预期路径，先验证环境与占用，不静默用CPU或干预其他任务。GPU失败/OOM先保留可靠完整状态、记录已成功更新，再只恢复未完成部分。

short验收：loss/grad有限、参数实际更新、A头参数/BN/Adam保持、B/C有效恢复、全局计数/RNG/cursor一致、没有明显输出塌缩、日志/曲线/验证可用。不能用600步低主分数否定完整阶段设计，也不能要求达到基线才结束准备。

固定32源/192对跑必要终点验证与诊断，不冒充完整1536。正式完整验证、诊断与checkpoint选择依方案实现；smoke必须实际验证其接口和A阶段不会用随机描述子选best_geometry。

## 6. 记录能力必须实际落地

每个更新保存JSONL：step、phase、lr、各任务系数、ratio统计、cursor、有效监督数、det/occupancy/position、gray、match原始/加权loss、total、clip、显存和耗时。A的match未计算项为null/available=false，不填虚假0。TensorBoard同口径。

实现固定分噪声诊断：1px/3px、固定K、相位/背景、实际自动点几何；gray加权/普通误差、边缘和抗噪指标。actual ratio与bucket分开，空集null，oracle/自动点、子集/完整、teacher1px/H-AUC@1分清。

保存代码/config/依赖/资产/源清单身份、完整BN/Adam/RNG/cursor/调度、阶段边界和best/last。诊断使用eval副本，不污染训练。只在启动/边界/恢复/故障抽检输入hash，不每步复制全部GPUtensor。

必须由短训日志实际生成loss、阶段/lr/noise、定位/背景/几何和gray曲线，保存可分析原始数据；标出阶段边界，raw与平滑分开。正式训练结束自动生成最终分析报告的能力应实现并用smoke数据验证。

## 7. 最终交付与手动启动

完成后给我：
- 唯一方案与代码一致性/删除清单；
- code-simplifier结果、OCR全部覆盖与修复复查通过证据；
- 准确测试命令/结果；
- 有限短训的实际更新账本、配置、恢复身份、曲线、分项/几何、失败与结论；
- 运行环境、GPU、输出目录和资产预检结果；
- 根据smoke实测A/B/C吞吐估算100000更新耗时，列出验证/存盘额外成本，不承诺预测精度。

提供并核验我手动执行的准确命令：
1. 选择GPU和环境、preflight；
2. 从随机初始化启动configs/formal.yaml的完整训练；
3. 同一run/config/output从最新完整checkpoint恢复；
4. 完整最终验证、导出、曲线/报告汇总。

沿用python -m rawfeat.cli的单一入口，参数先用--help、smoke与恢复核验。正式启动命令不得带smoke checkpoint、--skip-validation或压缩阶段。输出目录独立，避免覆盖历史资产；resume命令保持同配置/身份/路径。

交付写明“代码准备与有限短训完成，全量未启动”。不要执行完整命令，不创建定时/自动任务，不继续训练等我接管。
