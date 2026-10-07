# 文档入口与地位

当前唯一执行方案是 [低光Raw提取器统一方案](low_light_raw_feature_extractor_design.md)。它统一ratio100、当前实现、已确认事实、待验证候选和有限排查授权；最新实测与停止状态见§20，§19及此前内容为历史证据。先完整阅读该文件；下一会话的任务文本见 [检测排查执行prompt](detection_followup_prompt.md)。

| 文档或目录 | 地位与用途 |
|---|---|
| `low_light_raw_feature_extractor_design.md` | 唯一当前方案；§20为最新阶段对照实测和停止状态，前文保留设计及历史证据。 |
| `detection_followup_prompt.md` | 本轮结束交接状态；只读取最新证据，不自动恢复训练或重跑历史流程。 |
| [detection_gray_staged_report_20261002.md](detection_gray_staged_report_20261002.md) | §20最新实测：同源两组各500至2000，总账9209；持续联合51.8136%、阶段51.5805%，阶段未显示完整优势，暂缓采用；训练已停止。 |
| [detection_followup_report_20261002.md](detection_followup_report_20261002.md) | §17本轮实测报告：7209更新、噪声修复、五固定组、在线/噪声对照、完整1536对及效率；不定义并列执行方案。 |
| [detection_bn_residual_report_20261002.md](detection_bn_residual_report_20261002.md) | §19完成的BN有限实测，历史总账8209，正常BN mass1500为本轮父候选。 |
| [detection_methods_review_20261002.md](detection_methods_review_20261002.md) | 四个剩余问题的原论文/官方源码查阅、RawFeat适配限制及策略依据；研究附件，不是并列方案。 |
| `design_calibration_review_20261001.md` | 历史实测证据，记录当时的固定/随机训练、灰度对照和完整验证；不定义当前授权。 |
| `detection_design_discussion_20261001.md` | 原讨论与实例核对的证据说明；执行路线以统一方案为准。 |
| `asset_provenance.md` | 官方资产、版本、哈希与实际数值域来源。 |
| `review_report.md`、`implementation_audit.md`、`calibration_report.md`、`review_findings.json` | 更早的检查记录；涉及协议v1/ratio256或旧限制时按历史理解，不直接用于当前结果比较。 |
| `files/*.py` | 其他AI提供的候选实例代码，供核对与局部移植；正式实现位于`rawfeat/`。 |
| `losses.py` | 与`files/losses.py`相同的参考副本，不是训练入口。 |
| `low_light_raw_feature_extractor_design_v2.md`、`files/low_light_raw_feature_extractor_design_v2.md` | 已改为参考指针，避免与统一方案并列；其他AI原文完整保存在archive。 |
| `archive/` | 归档原文和校验清单，只供溯源，不定义当前任务。 |

原65类KL仍是正式代码默认；质量归一化位置项已可切换且有限训练收益得到支持，最新同源mass2000持续联合/阶段完整主分数51.8136%/51.5805%，父mass1500为43.4613%，同身份基线65.6030%；本轮阶段配置暂缓采用，训练已全部停止，总账9209，正式全量未启动。当前缓存/基线使用修复后的独立随机流噪声身份，旧资产保留。无跨阶段跳连仍是结构对照基线，灰度10仍未标定。`configs/formal.yaml`中的100000步是未来正式参考预算，本轮没有启动。

新会话不需要自行在多个旧方案中选择。用户当前指令优先；统一方案给出执行目标，实际代码与实验给出现状证据。若复核发现差异，记录并修正已确认错误，不把归档版本自动升级为当前要求。
