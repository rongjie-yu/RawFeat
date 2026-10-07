# HPatches 完整图像五档合成 Raw 评估执行说明

你在 `/homes/rongjie/projects/RawFeat` 工作。请先阅读并以当前唯一方案为准：

- `docs/low_light_raw_feature_extractor_design.md`
- `README.md`、`weights/v1/README.md`、`weights/v1/manifest.json`
- `docs/formal_training_analysis_20261007.md`
- `docs/hpatches_raw_evaluation_plan.md`
- `docs/hpatches_evaluation_tool_review_20261007.md`
- `/homes/rongjie/datasets/hpatches-sequences-release/dataset_integrity.json`

用户已经下载并验收完整 HPatches。当前实现已提供独立的 HPatches 完整图像评估入口；评估使用部署版 RawFeat、单图推理、六条件可视化、耗时统计，并在同一条 `hpatches-evaluate` 命令中自动生成报告。本方案不训练、不改变训练算法/权重/COCO协议、不重新下载数据。

## 已固定的外部资产

- Full image data：`/homes/rongjie/datasets/hpatches-sequences-release`
  - `dataset_integrity.json` 已登记官方 archive SHA256 `99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670`
  - 116 sequences：57 `i_` illumination、59 `v_` viewpoint；每个有 `1.ppm`…`6.ppm` 与 `H_1_2`…`H_1_6`，共696图、580参考→目标对
- 原来的 `/homes/rongjie/datasets/hpatches-release` 是65×N patch archive，只作官方 patch benchmark 参考，不能当作完整图像输入。
- Official patch benchmark：`/homes/rongjie/datasets/hpatches-benchmark`，commit `e0c993c7756d26129fef47f752f79e4cfe6681aa`。它只能评估patch descriptor verification/matching/retrieval，不能直接评估本任务的全图检测、MNN、RANSAC和H-AUC。
- SuperPoint full-image reference subset：`/homes/rongjie/datasets/SuperPoint-reference`，commit `1411bbd68c50163555d39c1b26e9e046ebd48f27`。只能借鉴读取 `1.ppm`/`H_1_i`、480×640 ratio-preserving resize和H适配；不要运行它的TensorFlow模型，也不要复制其使用GT H预先筛shared points的主评估逻辑。
- 首选部署模型：`outputs/deployment/v1/step096000/deployment.pt`，源 checkpoint 为 `weights/v1/checkpoint_step_096000.pt`；比较方法为 RawFeat 和 SuperPoint。`weights/v1/checkpoint_step_100000.pt`、`checkpoint_step_055000.pt` 是预登记次级候选，不要自动加入。

## 强约束

1. 不修改 `rawfeat/losses.py`、`model.py`、`training.py`、训练配置、v1权重、COCO manifest/cache或已有COCO评估定义。新增代码必须是HPatches adapter/cache/evaluator及必要CLI/测试/文档。
2. 不把官方patch benchmark的指标或公开SuperPoint论文数字写成RawFeat HPatches主结果。README、当前设计和历史报告不能被改成第二套训练方案。
3. 不使用真值单应矩阵提前过滤、截断或重排候选点；完整图像的所有模型点进入阈值/NMS后MNN/RANSAC。GT H只用于评估可见性、正确匹配、重复性、角点误差和阈值统计。
4. 不做域适配、BN校准、再训练、权重更新、测试时优化、额外颜色增强、随机裁剪、随机几何warp、局部质心后处理或新匹配器。
5. 先做CPU预检、契约测试和有限GPU smoke；正式评估固定单图 `batch_size=1`，不允许用 batch=4 替代单图耗时。完整评估命令应自动生成指标、可视化、耗时和报告，不要求第二条报告命令。
6. 使用 `[$code-simplifier:code-simplifier](/homes/rongjie/.codex/skills/code-simplifier/SKILL.md)` 简化新增/修改代码；随后执行 `[$open-code-review-delegate](/homes/rongjie/.agents/skills/open-code-review-delegate/SKILL.md)`，由主会话逐项审查、修复并复查到无未解决问题。不要保留历史/候选可执行评估路线。

## 数据与坐标协议

### 图像和H

- 每个 sequence 使用 `1.ppm` 作为参考，使用 `2.ppm`…`6.ppm` 作为目标，共五对，真值为对应 `H_1_2`…`H_1_6`，方向为参考坐标→目标坐标。文件是空格分隔的3×3矩阵；不要按patch版的 `H_ref_*` CSV解析。
- sequence 类型只按目录名前缀：`i_` 或 `v_`。不依赖不完整/不一致的 `attribs.txt` 来决定类型。
- PPM全部RGB。读取后转成RGB float `[0,1]`，不得把BGR直接送入InvISP。

### 预处理

- 固定输出高480、宽640，输入模型是4×240×320。每一张源图单独按 `s=max(640/W,480/H)` 缩放，令缩放后两个轴都不小于目标，再中心裁剪到640×480；使用明确的OpenCV线性插值和实际输出尺寸，不做非等比拉伸、不填黑边、不裁成任意随机crop。
- 记录每张图实际 `W'`,`H'`,`sx=W'/W`,`sy=H'/H` 和中心裁剪左/上偏移 `cx`,`cy`。采用像素中心坐标时：

  `T=[[sx,0,(sx-1)/2-cx],[0,sy,(sy-1)/2-cy],[0,0,1]]`。

  对每个目标：`H_processed = T_target @ H_1_target @ inv(T_reference)`。不要用原生H、不记录名义scale、或在图像已resize后再次缩放H。
- 在CPU测试中验证：恒等H、H正逆方向、四角投影、resize前后随机点投影一致；至少保存几组参考/目标可视化或数值证据。

### 合成Raw

- 对处理后的每张RGB图单独走当前已验证路径：InvISP reverse → `invisp_to_sensor_rgb` → RGGB `sample_bayer` → ELD noise → 当前 `pack_student`。复用现有 `sensor.py`、`noise.py` 和官方资产，不复制噪声数学。
- `clean`：不调用ELD，使用clean Bayer；`ratio=1,4,16,64,100`：两视角使用独立固定ELD seed。ratio1含ELD噪声，不能当clean。
- seed必须由固定seed2027、sequence排序位置、图像编号、ratio和条件字段通过稳定的`SeedSequence`/哈希生成；禁止Python进程hash和未登记随机流。同一图像+ratio在不同配对/条件中复用同一缓存。
- noisy Bayer必须以FP32保存；允许负值和超白中间量，不能uint16截断、不能静默clip、不能FP16压缩。缓存identity要包含数据完整性manifest、当前RawFeat数据/噪声源码SHA、InvISP/ELD/标定SHA、预处理公式、seed和条件表。

## 五档条件与样本量

为保持与当前COCO五桶的语义一致，同时避免把ratio1视角藏在高ratio均值中，使用以下固定条件：

- clean：`(clean,clean)`，580对。
- ratio1：`(1,1)`，两侧独立ratio1 ELD，580对。
- 每个 `r ∈ {4,16,64,100}`：只使用对称条件 `(r,r)`，各580对。训练时两侧 ratio 独立连续采样，可能出现不相等样本，但没有显式 `(1,r)`/`(r,1)` 训练层；这两类不进入首轮正式评估。

因此每个方法完整是 `580 × (1 + 1 + 4) = 3480` 对。应按**图像级缓存**复用前向特征，不能重复推理同一图像+ratio。

五个含噪档（1、4、16、64、100）等权平均H-AUC@5；clean单独报告。`(1,r)`/`(r,1)`如以后需要，只能另立不纳入主分数的诊断协议。

## 模型、基线和指标

- RawFeat：加载 `outputs/deployment/v1/step096000/deployment.pt`，eval模式、BN固定，部署模型已融合RepVGG并删除gray头；输出只包含logits和128维descriptor。保持当前阈值0.005、NMS4、border8、max_points1024和整数检测点。
- SuperPoint：使用当前 `baseline_features` 路径的 MeanAD+SuperPoint，输入必须来自**相同HPatches图像、相同processed尺寸、相同noisy Bayer缓存**。不能套COCO的65.6030%，也不能把原始sRGB SuperPoint数字当本评估结果。
- 每一对两侧提取全图候选；descriptor mutual nearest neighbor；保持当前 `rawfeat.metrics.pair_metrics` 的3px reprojection threshold、OpenCV RANSAC `maxIters=10000`、confidence0.999、固定RNG seed。不要照搬SuperPoint reference的GT shared-point预筛。
- 对每对保留：`corner_error`、H-AUC@1/3/5、`corner_error≤1/3/5px`比例指示量、失败、MNN matches、correct_matches、precision、detected_a/b、repeatability、条件定位误差。H-AUC定义保持当前实现：`mean(max(0,1-corner_error/x))`；失败贡献0。H-AUC不是阈值成功率，报告时分开命名。
- 分别汇总 all(116序列)、illumination(57)、viewpoint(59)，并按clean/ratio条件报告。每序列有5个目标对，先对序列内5对求均值，再做all/type汇总，避免把同一reference的5对当作完全独立样本。
- 对 RawFeat−SuperPoint 的主H-AUC@5差值做 sequence-cluster bootstrap：固定seed20261007、10000次；每次重采样116个sequence，连同其5个目标、全部条件一起抽样。报告点估计、95%区间、序列正负数。不要把pair级bootstrap当作独立不确定性，也不要因在HPatches上选最高checkpoint而宣称独立泛化。

## 工程输出与日志

建议新增独立输出根：`outputs/hpatches_synthetic_raw_v1/`，不覆盖COCO output。至少保存：

- frozen manifest、dataset/cache identity、完整命令、环境/依赖/GPU、checkpoint和baseline SHA；
- image-level cache index（图像、condition、seed、T）和逐图推理耗时/显存；
- 每对 `per_pair.jsonl`（sequence、type、target index、condition、ratio_a/b、H_processed、所有指标）；
- 按条件/all/illum/view汇总JSON/Markdown、sequence-cluster bootstrap JSON、失败清单、角点误差分位数和>20px尾部；
- 固定少量可视化：原/目标处理图、检测点、前100 MNN线、估计/真值角点；不能只保存最高分样例。

## 执行阶段

1. CPU preflight：检查路径、dataset_integrity、116/696/580计数、所有图和H、输出尺寸/T/H方向契约；不载入训练或更新模型。
2. 单元/契约测试：解析、resize/H、稳定seed、条件计数、cache identity、失败/空匹配、H-AUC公式、sequence bootstrap；必要时使用mock小数组，不伪造完整分数。
3. GPU smoke：固定2个`i_`和2个`v_` sequence，20基础对，覆盖clean/ratio1及每个r的`(r,r)`，最多120 pair/method；RawFeat和SuperPoint使用同一缓存、单图推理。检查数值域、H方向、显存、耗时、可视化、失败保留。
4. smoke通过后交付完整单图评估命令和实测预算。完整运行使用同一manifest/cache和部署 checkpoint；`hpatches-evaluate` 必须在同一运行中写出 `summary.json`、`latency_summary.json`、全部可视化以及 Markdown/HTML/CSV 报告，不需要再运行 `hpatches-report`。
5. 完整结束后运行只读分析，不能按HPatches得分自动替换v1 preferred checkpoint、修改训练方案或删除逐对证据。

完成后向用户汇报：改了哪些文件、测试命令/结果、smoke结果、完整命令和预算、主协议是否仍按以上定义执行。若实现中发现上述协议与现有代码有冲突，先记录具体冲突和影响，停止完整评估并报告，不要静默改变口径。
