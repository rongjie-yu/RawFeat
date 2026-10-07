# rawsp：Raw 域 SuperPoint 格式提取器（开发顺序第 1 项：网络模块）

## 内容
- `rawsp/repblock.py`   RepVGG 稠密块（训练期 3x3+1x1+identity，推理期融合为单个 3x3）
- `rawsp/student.py`    学生网络（packed RGGB 输入，SuperPoint 格式输出，可选训练期辅助去噪头）
- `rawsp/superpoint_ref.py`  原版 SuperPoint 结构（效率基线；之后作冻结教师）
- `rawsp/measure.py`    MACs / 参数量 / 延迟（学生与原版同一流程，默认 FP32，可选 ONNX+ORT）
- `tests/test_network.py`  单元测试

## 运行
    python -m pytest tests -v
    python -m rawsp.measure --height 480 --width 640 --device cuda --onnx

## 验证状态（如实记录）
- 已验证（numpy 独立复算）：融合数学（误差约 1e-13）；MACs/参数量：
  SuperPoint 26.05 GMACs / 1.301M；学生最小规格 4.52 GMACs / 0.619M（480x640）。
- 未在沙箱中运行：所有 PyTorch 代码与测试（沙箱无 torch 且无网络）。请先跑 pytest，
  失败的用例把报错发给我。
- `superpoint_ref.py` 的模块命名按官方 SuperPointNet（凭记忆），加载官方权重时若键不匹配请告知。

## 约定
- 输入 H、W 为 8 的倍数；packed 输入为 [B,4,H/2,W/2]，通道 [R,G1,G2,B]，规范相位 RGGB。
- 检测 65 通道的最后一个通道是 dustbin；通道 k 对应 cell 内 (k//8, k%8)。
- 网络输出的描述子未归一化（L2 归一化与插值放在后处理）。
