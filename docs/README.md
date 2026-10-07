# 当前方案与证据

唯一当前目标：[分阶段训练设计](low_light_raw_feature_extractor_design.md)。当前实现使用 mass 检测、正常 BN、gray10 固定，从随机初始化进行 A50000/B5000/C45000。此版本已由用户完成100000更新；[首版权重](../weights/v1/README.md)与[训练分析](formal_training_analysis_20261007.md)记录最终情况。

| 文件 | 地位 |
|---|---|
| low_light_raw_feature_extractor_design.md | 唯一算法、训练、验证与交付契约 |
| [formal_training_analysis_20261007.md](formal_training_analysis_20261007.md) | 完整100000更新的独立审计、阶段/gray曲线、同身份1536对结果与源图配对区间；不修改算法契约 |
| [results/v1/](results/v1/) | 首版完整训练身份、配置、验证逐对结果、分析统计和曲线快照 |
| staged_preparation_report_20261002.md | 本次清理、简化、OCR修复复查、测试、有限短训及手动命令证据 |
| asset_provenance.md | 保持的官方资产、标定、数值域与缓存身份 |
| detection_followup_prompt.md | 本次执行任务原文，保留作为需求记录 |
| detection_gray_staged_report_20261002.md | 历史9209总账与继续训练对照，不定义当前训练入口 |
| detection_bn_residual_report_20261002.md | 历史BN证据，实验入口已删除 |
| detection_followup_report_20261002.md | mass和独立CUDA噪声流修复、同身份完整基线 |
| detection_methods_review_20261002.md、archive/、其他Markdown | 历史来源；旧默认值与限制不覆盖当前方案 |
| docs/files/ | 仅保留历史Markdown；候选Python副本已删除 |
| outputs/ | 保留历史结果、配置记录、缓存、checkpoint；不作为源码入口 |

`configs/formal.yaml` 与 `configs/smoke.yaml` 仅压缩时长、验证子集及记录频率，使用相同算法。阶段比例和gray10是明确选择，尚未证明最优。
