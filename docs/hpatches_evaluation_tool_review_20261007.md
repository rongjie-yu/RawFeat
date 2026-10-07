# HPatches 评测工具核对

本文件记录已下载工具的边界，防止新会话把 patch benchmark 当成完整图像 H-AUC 工具。

## 已下载的工具

### 官方 hpatches-benchmark

路径：`/homes/rongjie/datasets/hpatches-benchmark`；commit：`e0c993c7756d26129fef47f752f79e4cfe6681aa`。

这是 HPatches 官方 Python/Matlab **patch descriptor** benchmark。`python/hpatches_eval.py` 读取每个序列的 `ref/eK/hK/tK.csv` 描述子，执行 verification、matching、retrieval；`docs/tasks.md` 的任务文件也以 patch index 为单位。它没有读取完整序列的 `1.ppm…6.ppm`，没有自动检测点、全图 MNN、RANSAC 或 H-AUC@1/3/5 入口。因此它只作为 HPatches patch 协议、数据命名、引用和任务定义的参考，不能直接评估 RawFeat 的完整检测链路。

官方数据说明明确区分 patch archive 与 Full image sequences；完整图像另有 `hpatches-sequences-release.zip`。见 [官方数据仓库](https://github.com/hpatches/hpatches-dataset) 和 [官方 benchmark](https://github.com/hpatches/hpatches-benchmark)。

### SuperPoint full-image 参考实现

路径：`/homes/rongjie/datasets/SuperPoint-reference`；commit：`1411bbd68c50163555d39c1b26e9e046ebd48f27`。这是稀疏下载的原始 TensorFlow SuperPoint 参考代码，保留了 `patches_dataset.py`、`descriptor_evaluation.py`、配置和数据加载相关文件。

它确认了几个成熟做法：读取 `1.ppm` 与 `2…6.ppm`、读取 `H_1_i`；HPatches 参考配置使用 `[480,640]`；不同源尺寸先按最大轴比例缩放，再中心裁剪/填充，并显式把 H 变换到预处理坐标；全图单应性评价最多1000点，使用最近邻/交叉检查与 OpenCV RANSAC，并以1/3/5像素角点阈值报告正确率。论文还说明标准 HPatches 全图实验在480×640、最多1000点上进行。

这份 TensorFlow 代码不是当前 RawFeat 的执行器。它的 `keep_shared_points` 会用真值 H 预先保留映射后仍在目标图内的点，且 `correctness` 只返回“角点误差≤阈值”的二值值；当前 RawFeat 主协议需要保留完整自动点集、记录连续 H-AUC 和失败，所以新会话只能借鉴数据读取与坐标变换，不能直接复制其评价结果。

## 对当前 RawFeat 协议的结论

1. 完整图像主评估必须新增独立 HPatches adapter/manifest/cache/evaluator；不能把 `rawfeat.validation.read_manifest` 或官方 patch benchmark 直接换 root 复用。
2. 当前主指标继续使用 RawFeat 的 H-AUC@1/3/5、角点误差、失败、MNN、匹配精度、正确匹配数、点数、重复性和条件定位误差；同时增加 HPatches 常见的 `P(corner_error≤1/3/5px)`，避免把 H-AUC 与阈值成功率混称。
3. 主链路不使用 GT H 筛点、不把共同可见点作为自动匹配候选；GT H 只用于评估重复性、正确匹配和估计矩阵误差。这样检测瓶颈不会被“oracle shared points”隐藏。
4. 同一缓存的 noisy Bayer 必须同时提供给 student 与 MeanAD+SuperPoint baseline；COCO 的65.6030%不能移植为 HPatches baseline 分数，baseline 必须在 HPatches 自己的相同噪声条件上重算。
5. 公开 SuperPoint 的 H-AUC/正确率只能作文献参考，不能作为本项目的 baseline；它使用不同的 sRGB输入、NMS/top-K和可能的 shared-point过滤。

## 来源

- [HPatches full image sequences说明](https://github.com/hpatches/hpatches-dataset#full-image-sequences)
- [官方 hpatches-benchmark](https://github.com/hpatches/hpatches-benchmark)
- [SuperPoint HPatches配置](https://github.com/rpautrat/SuperPoint/blob/master/superpoint/configs/superpoint_hpatches.yaml)
- [SuperPoint full-image descriptor evaluator](https://github.com/rpautrat/SuperPoint/blob/master/superpoint/evaluations/descriptor_evaluation.py)
- [SuperPoint论文 HPatches实验](https://arxiv.org/html/1712.07629v4#S7.3)
