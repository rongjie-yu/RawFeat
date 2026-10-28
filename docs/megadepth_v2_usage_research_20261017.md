# RawFeat V2：MegaDepth 单图合成训练使用方案（2026-10-17）

**沿用已下载的 Cornell MegaDepth v1 包中 `dense*/imgs/*.jpg`，将它接成第二个 RGB 源。训练按 MegaDepth:COCO2017 = 6:4 抽样，MegaDepth 内先抽场景、再抽图片；两库完整共用 V1 合成、监督和网络流程。** 本文研究数据使用与拆分，不更改网络、损失、训练配置或 checkpoint；根据用户最新要求，最大ratio为225，保持两侧独立对数均匀采样，具体课程见 [V2训练方案](v2_training_optimization_plan_20261017.md)。

## 1. 已下载的 RGB 可以直接使用

Cornell 原包的 196 个场景各含一个或多个 `denseX` 连通分量，每个分量有 `imgs` 与 `depths`。`imgs` 是 MVS 使用的去畸变图像；官方 README 明确：有效深度对应的图像只是这些 RGB 的子集。因此 V2 应读取全部合格 `imgs/*.jpg`，**不要要求图片存在对应 `.h5`，也不要仅用有深度的图片建立单图池**。README 描述 MVS 时最长边上限为 1600；本地实际尺寸分布仍应在解压后的清单中核实。[Cornell v1 README](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/README.md)

三个容易混淆的数据版本用途不同：

| 来源 | 图像内容与当前用途 |
|---|---|
| 已下载 `MegaDepth_v1.tar.gz.part-*` | 原始 v1 的去畸变 MVS RGB + 清理后深度；RGB 足够支持当前单图合成 |
| Cornell `MegaDepth_SfM` | 原始互联网照片与稀疏 SfM 模型，不是当前已下载包；无需为本版另下约 667 GB 包 |
| LoFTR / D2-Net `Undistorted_SfM` | 为相机内外参与图像一致性重新准备的去畸变 RGB；LoFTR 使用这个版本做真实多视图几何监督 |

Cornell SfM README 将 `images` 定义为原始互联网照片；LoFTR 明说自己从原始 v1 只用深度，并另用 D2-Net 的 RGB，是因为原包去畸变图像不附对应内参。这不意味着原包 RGB 无法用于 V2。V2 的真值来自采样单应，不依赖内参、位姿、深度、重叠矩阵或 LoFTR 图像对索引。[Cornell SfM README](https://www.cs.cornell.edu/projects/megadepth/dataset/MegaDepth_SfM/README.md)、[LoFTR 官方训练说明](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#megadepth)

本地下载文档记录固定 HF revision `107225256a43e2792d3ed404b790e1b05e173785`，五片合计 213,461,598,257 字节且已有完整 SHA-256 审计。本次又读取镜像的固定 revision API，核对五片大小与 LFS SHA-256，均与该记录一致；没有重新扫描 213 GB 归档。主任务本次本地检查确认 RGB 目录尚空，**下载完整不等于 RGB 已解压可训练**。[镜像固定 revision 元数据](https://huggingface.co/api/datasets/personal-haiduong/MegaDepth-v1/tree/107225256a43e2792d3ed404b790e1b05e173785?recursive=false)、[本地下载与审计记录](megadepth_hf_download_20261014.md)

## 2. 场景拆分：完整核对 LoFTR 官方索引

LoFTR 配置中训练读取 `train_list.txt`，验证读取 `val_list.txt`；单独的 MegaDepth-1500 测试入口读取仓库里的五项列表。不能将难度分桶后的子项数当成独立地点数。[LoFTR train/val 配置](https://github.com/zju3dv/LoFTR/blob/master/configs/data/megadepth_trainval_640.py)、[LoFTR MegaDepth-1500 配置](https://github.com/zju3dv/LoFTR/blob/master/configs/data/megadepth_test_1500.py)

本次从 LoFTR 官方文档链接的 [训练数据目录](https://drive.google.com/drive/folders/1SrIn9WJ1IuG08yh2nEvIsLftXHLrrIwh) 定位 [megadepth_indices.tar](https://drive.google.com/file/d/1YMAAqCQLmwMLqAkuRIJLDZ4dlsQiOiNA/view)，通过 HTTP Range 读取其末尾的 `trainvaltest_list`，得到：

| 官方文件 | 分桶子项数 | 去除 `_overlap` 后的原始场景 |
|---|---:|---|
| `train_list.txt` | 368 | 153 个，集合与 GlueFactory `train_scenes_clean.txt` **完全相同** |
| `val_list.txt` | 5 | `0015, 0022` |
| `test_list.txt` | 24 | `0008, 0019, 0021, 0024, 0025, 0032, 0063, 1589` |

训练与验证、训练与测试的场景交集均为空。仓库公开的 [MegaDepth-1500 列表](https://github.com/zju3dv/LoFTR/blob/master/assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt) 与上述 `val_list.txt` 的五项一致。GlueFactory 的 [clean train](https://github.com/cvg/glue-factory/blob/main/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean.txt)、[clean valid](https://github.com/cvg/glue-factory/blob/main/gluefactory/datasets/megadepth_scene_lists/valid_scenes_clean.txt)、[clean test](https://github.com/cvg/glue-factory/blob/main/gluefactory/datasets/megadepth_scene_lists/test_scenes_clean.txt) 也与这些原始场景集合一致。

Range 审计：归档总大小 1,676,636,160 字节；最后一次仅读取 `1676619264–1676636159`，响应 206，共 16,896 字节。提取出的原始文本 SHA-256：

```text
train_list.txt  6e1f2cf7924774190867e541b1693adae2869653ab41f4788ec2fccbfbccfecc
val_list.txt    903aed9b560881bdf211dcd307cbc84563fb59642273c15002cf0336f7c3ee62
test_list.txt   4ed2973ce55bd614efdbf1739b38c5591dc57e1f009dcc0935d2a2667e24a2cc
```

**V2 拆分：**以这份 153 场景白名单与本地有效 RGB 的交集建立图像池；按固定 seed 留出10个完整场景作为内部合成验证，每场景20源，其余训练。先查看本地场景规模与内容，再固定具体名单；覆盖不足时在训练前记录并调整清单。`0015/0022` 留给 MegaDepth-1500 最终评估，另8个官方测试场景整体保留；未进入白名单的本地场景先不自动加入。这样不会用与最终 benchmark 相同的场景选择学习率、停止步数或 checkpoint。COCO 则继续 train2017 训练、val2017 验证。

GlueFactory 官方说明旧拆分曾与 IMC 场景重叠，clean 拆分移除了该重叠；这支持优先用已清理白名单，但不能证明其排除了任意未来 IMC 年份或其他自选 benchmark。对计划报告的其他测试集，还需核对地点与图片重叠。[GlueFactory 官方说明](https://github.com/cvg/glue-factory#training)

## 3. 6:4 应落实到实际样本，而非文件数量

建议每个训练样本先决定来源，概率分别为 `0.6/0.4`。命中 MegaDepth 后均匀抽训练场景，再在该场景内均匀抽有效 RGB；命中 COCO 后均匀抽 train2017 RGB。由此 MegaDepth 场景 `s` 内单张图的概率为 `0.6/(S·N_s)`，避免大场景凭图数占据大多数 MegaDepth 样本。LoFTR 默认采用 `scene_balance`，GlueFactory 也按场景限制抽样数量；均支持先控制场景贡献的做法。[LoFTR sampler 默认配置](https://github.com/zju3dv/LoFTR/blob/master/src/config/default.py)、[GlueFactory 数据加载与抽样实现](https://github.com/cvg/glue-factory/blob/main/gluefactory/datasets/megadepth.py)

6:4 按**送入合成流程的源图样本数**统计，不能只拼接两个目录后随机洗牌。V2保持每更新16对，以5次更新的80对安排48 MegaDepth与32 COCO，记录窗口实际比例；多卡时统计全局样本数。两库使用相同噪声课程和增强分布，避免 `ratio225` 与某一来源绑定。

场景单位必须是最上层地点 ID，不能把同一地点的 `dense0/dense1` 切到训练与验证两边。建清单时保留 `scene_id、dense_id、relative_path、width、height`；同名图先核对内容再去重，同一图像的重复版本不能跨集合。场景拆分降低近重复泄漏风险，但不同 ID 仍可能拍到同一地标；应检查保留集与训练集的重复文件及明显近重复，发现后整组归入同一侧。196 地点与多个连通分量的关系来自 [Cornell 原始说明](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/README.md)；去重与分组规则是本项目的建议。

## 4. 此路线能够增加什么，以及仍需验证什么

V2 数据流保持 V1：同一 RGB crop 生成两个独立外观视角，采样 `H`，InvISP 输出连续 sensor RGB，在连续域施加几何，再 RGGB、ELD 和 packed Raw；监督仍来自两侧干净视角。尤其继续在 Bayer 采样前做几何，不对马赛克直接 warp。[项目当前流程](../README.md#1-数据如何合成)

**推断：**MegaDepth 的旅游/建筑源图补充 COCO 的内容、纹理、重复结构与外观分布，有助于测试数据扩充是否改善泛化；但两视角来自同一图片，无法由此学到 MegaDepth 真实多视图中的视差和遮挡。JPEG→InvISP→ELD 也仍是合成相机域，不等同于真实拍摄的低照 Raw；`ratio225` 的效果必须用固定低噪声档与新增高噪声档共同验证，不能只报告高噪声平均值。这里是对现有合成方式的边界分析，不是已有实测结论。

实施前尚待完成：RGB 提取和有效图清单；白名单在本地的实际覆盖、每场景图数/尺寸/重复率；独立内部验证场景的最终名单；代表样本经现有 resize/crop 后的内容质量。全部完成后，再固定清单摘要与 seed 开启 V2。本文没有解压全包、训练或进行性能评测。
