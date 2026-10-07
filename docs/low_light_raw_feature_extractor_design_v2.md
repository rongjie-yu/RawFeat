# 旧v2入口

当前唯一目标是[分阶段全量训练方案](low_light_raw_feature_extractor_design.md)，新会话执行[清理、审查、短训prompt](detection_followup_prompt.md)。本文件不定义第二套方案或旧训练限制。

其他AI原文仅供溯源：[归档](archive/other_ai_design_v2_20261001.md)。最新方案从随机初始化执行50000检测+gray、5000匹配接入、45000联合；完整训练由用户手动启动，生产实现尚待适配。
