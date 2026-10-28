# RawFeat V2 训练优化方案（2026-10-17，实现版）

以 V1 已发布的 **step96000 训练权重初始化一个新 run，进行 30000 次联合更新**；MegaDepth:COCO2017 按源图样本数固定为 **6:4**，最大 ratio 提高到 **225**，两侧独立对数均匀采样。网络、检测/描述子/gray 头、损失公式和部署接口沿用 V1。

根据用户修订，本版采用单一训练方案：统一对数均匀采样，不设强噪额外配额或训练对照分支。训练支持已实现，完成OCR审查与CUDA短测试；正式30k训练尚未启动。启动和恢复命令见 [V2全量训练步骤](v2_training_runbook.md)，验证结果见 [实现与验证记录](v2_implementation_validation_20261017.md)。

本方案的6:4指V2阶段新增训练样本的比例；V1已经完成的COCO预训练是初始化历史，单独保留在lineage中。

## 1. V1 证据支持优化什么

本次重新读取了原 run 的全部 100000 行 `scalars.jsonl`，并提取相同 256 源完整验证的阶段指标；没有重跑 GPU 推理。统计记录在本任务的 `.planning/2026-10-17-v2-training-optimization-design/v1_v2_evidence.json`，历史完整分析见 [V1 训练分析](formal_training_analysis_20261007.md)。

| 观察 | 实际记录 | 对 V2 的启示 |
|---|---|---|
| 检测较早进入平台 | 20k 教师1px召回44.68%，46k最高45.81%，50k为43.76% | 已有成熟 V1 时无需重新付出50k检测专训；不能据此认定从零训练20k必然足够 |
| 匹配接入伴随检测回落 | 50k→55k召回43.76%→41.84%；clean召回61.62%→57.83% | 联合适配时降低共享层更新幅度，观察定位是否保留 |
| 后期几何收益有限 | 55k H-AUC@5=68.2572%，96k=68.0846%，100k=67.7004% | 优先改变数据与调度，不直接追加相同100k训练 |
| 描述子条件检索已较强 | 最终clean97.72%、ratio100约90.16% | 从已有描述子继续联合训练，避免再次长时间冻结 |
| 强噪精细定位仍有缺口 | 最终ratio100 H-AUC@1=4.68%，基线6.16% | 除H-AUC@5，还需关注1/3px几何与实际ratio的检测定位 |
| 跨数据表现需要兼顾低噪 | V1 HPatches五噪声均值48.6307%，基线46.8948%；clean/ratio1/4点估计稍低 | 扩展到225时继续跟踪低噪和clean表现 |

55k、96k、100k的几何差异在既有配对置信区间中未得到稳定胜负确认，不能把55k称为确定最优，或把后期变化断言为过拟合。96k作为初始化来自已发布的预定选模结果，避免重新挑选起点。HPatches结果见 [V1 完整评估](results/hpatches_v1/summary.json)。

共享层的稀疏梯度探针提示匹配梯度有时较强，但不能证明检测回落全部由任务冲突造成。降低共享学习率会同时缩小两类梯度带来的参数变化，**不会改变匹配/检测梯度的相对比例**。

## 2. MegaDepth 如何使用

### 2.1 本地状态与提取范围

已下载的是 Cornell MegaDepth v1 原包的连续五片，共 **213461598257 字节**。本次检查文件大小，并读取已有完整 SHA-256 审计，五片均通过；没有重做全量哈希。`/homes/rongjie/datasets/megadepth/rgb` 当前为 **0 个条目、0 张 JPG**，所以准备工作还包括提取 RGB、建立清单。

原包 `dense*/imgs` 提供去畸变 RGB，深度只覆盖 RGB 子集。当前单图合成监督不依赖深度、内参或位姿，因此读取全部合格 RGB，不以是否存在 `.h5` 筛选，也无需为本版重下 LoFTR 的 SfM 图像库。[Cornell 原始数据说明](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/README.md)、[LoFTR 数据准备说明](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#megadepth)

后续 RGB 提取可执行以下长 I/O 命令；本轮没有执行。它只输出 JPG，不生成额外213GB合并包。现有校验记录及完整命令见 [下载记录](megadepth_hf_download_20261014.md)。

```bash
set -o pipefail
cat /homes/rongjie/datasets/megadepth/archives/MegaDepth_v1.tar.gz.part-{00,01,02,03,04} |
  tar -xzf - -C /homes/rongjie/datasets/megadepth/rgb \
    --wildcards --wildcards-match-slash '*.jpg'
```

提取后检查管道退出码、实际 RGB 数量、逐图解码情况与逐场景计数。先不估计本地有效图数或场景覆盖；这需要完整清单才能回答。

### 2.2 按完整地点拆分，再按场景采样

使用 LoFTR/GlueFactory **clean 训练白名单的153个场景与本地有效 RGB 的交集**。从中固定留出10个完整场景作内部合成验证，目标每场景20张，共200源；具体名单在解压、检查实际覆盖后固定。若覆盖不足，应记录实际选择并在训练前重新锁定方案，不能训练期间换场景。

`0015/0022` 为 LoFTR 验证及 MegaDepth-1500 场景，保留给以后评估；另8个官方测试场景也排除于训练/选模。LoFTR官方索引的153训练场景与GlueFactory clean训练集合已核对一致；详细列表、原始链接与审计见 [MegaDepth专项调查](megadepth_v2_usage_research_20261017.md)。[clean训练白名单](https://github.com/cvg/glue-factory/blob/main/gluefactory/datasets/megadepth_scene_lists/train_scenes_clean.txt)、[MegaDepth-1500列表](https://github.com/zju3dv/LoFTR/blob/master/assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt)

清单保留 `dataset、scene_id、dense_id、relative_path、width、height`。同地点的多个 `denseX` 必须处于同一split；同图的重复副本去重，检查训练与保留集的相同/明显近重复图，必要时按关联地点整组拆分。默认只剔除损坏和明确重复的源，不把低纹理图片全部筛掉。clean白名单降低已知泄漏风险，仍需对以后实际使用的测试集核对地点重叠。

MegaDepth内部先均匀选训练场景，再均匀选场景内图像；COCO内部均匀选 `train2017` 图像。这样大地标不会仅凭照片数量占据大部分MegaDepth配额。以最上层地点ID作为场景，不能把 `dense0/dense1` 当独立地点。[LoFTR场景平衡采样配置](https://github.com/zju3dv/LoFTR/blob/master/src/config/default.py)、[GlueFactory实现](https://github.com/cvg/glue-factory/blob/main/gluefactory/datasets/megadepth.py)

### 2.3 将6:4落实到实际训练对

继续使用每微批4对、累积4次，即每更新16对。不为整除10改变batch或BN行为；以 **5次更新=80对** 为采样块，安排 **48 MegaDepth + 32 COCO**。

例如每块五次更新的MegaDepth数量为 `10,9,10,9,10`，对应COCO为 `6,7,6,7,6`；顺序按固定seed打乱。每微批安排2或3张MegaDepth源，其余COCO，避免整个微批只来自一个库。每个完整80对块精确6:4，块内暂停时报告实际比例。

一个源图生成一个合成训练对；6:4统计源图请求次数，允许同一源跨请求再次出现。数据源/场景/图像/增强/噪声的随机数都由稳定样本编号推导，避免依赖可变文件扫描顺序或进程随机hash。记录每库、每场景、每噪声段的实际样本量，并支持恢复到块内任意sample cursor。

两库使用完全相同的流程：**RGB crop → 两侧独立外观 → 采样H → InvISP → 连续sensor RGB几何变换 → Bayer → ELD → packed Raw**。H仍发生在Bayer采样之前，教师和gray目标来自干净视角。MegaDepth在本版补充图像内容与纹理分布，不直接引入真实多视图的视差、遮挡监督。

## 3. ratio225与统一对数均匀采样

### 3.1 从已有100范围平滑扩展

V1已训练到100，V2不从ratio1重新开始课程。以本run成功更新数 `k` 定义：

```text
u(k) = clip((k - 1000) / 9000, 0, 1)
h(k) = (1 - cos(pi*u(k))) / 2
R(k) = exp(log(100) + h(k) * log(225/100))
```

因此前1000步上限为100；1001–10000步平滑扩到225，其中5500步上限恰好为150；10000步后保持225。两个数据源共用同一课程。课程只改变上限，每个阶段都保持同一种对数均匀分布。

### 3.2 全程采用单一采样分布

每个训练对的两侧分别独立采样：

```text
U_a, U_b ~ Uniform(0,1), independent
ratio_a = exp(U_a * log(R(k)))
ratio_b = exp(U_b * log(R(k)))
```

两侧ratio、ELD噪声实例相互独立，保持V1的连续曝光差异。最大范围时即 `logU[1,225]`，不额外抽强噪声桶，也不强制对称曝光。clean不等于ratio1，后者仍有ELD噪声；沿用V1监督方式，不额外加入clean训练配额。

在最终范围，采样分布为：

```text
P(单侧ratio>100) = log(225/100)/log(225) = 14.9726%
P(双侧ratio>100) = 2.2418%
P(至少一侧ratio>100) = 27.7034%
P(单侧ratio>150) = log(225/150)/log(225) = 7.4863%
median(ratio) = sqrt(225) = 15
```

以上是数学分布性质，用于核对训练采样计数，不是效果预测。它自然提高原ratio100以上的覆盖，同时保留完整低噪范围。

保持 Canon 标定、黑白电平、CFA、FP32 与不裁剪 noisy DN 的原有约定。ratio通过缩短曝光、加噪、放大还原作用于输入，不等价于把所有噪声项标准差统一乘2.25。官方ELD也在更高曝光倍率范围使用噪声合成，但这不直接证明本项目ratio225的真实域精度。[ELD官方噪声实现](https://github.com/Vandermode/ELD/blob/master/noise.py)、[真实数据采集条件](https://github.com/Vandermode/ELD#eld-dataset)

## 4. 30k训练调度

### 4.1 初始化与优化器

- 初始化：`weights/v1/checkpoint_step_096000.pt` 的完整 `student`，含gray参数及BN缓冲，严格匹配全部键；不从已融合、删除gray的部署权重开始。
- 新建V2 run，重置AdamW状态、调度、sample cursor和best记录；固定seed42。记录初始化checkpoint的SHA与V1 lineage。
- 全程联合训练，`λ_det=1、λ_match=1、λ_gray=10`；不再重复A阶段或冻结描述子。保持BN正常训练。
- 维持AdamW、weight decay `1e-4`、clip `5`、FP32、4对×4累积，不在首轮同时改变精度或有效batch。

改变数据和配置不是原run的精确resume。后续需要独立的“加载初始化权重”入口，不能关闭V1的恢复身份检查。V2自身中断后仍严格恢复V2的配置、清单、样本游标和优化器。

### 4.2 分组学习率与阶段

共享骨干指 `stage1/2/3`；头包含 detector、descriptor、gray。所有参数继续更新，骨干学习率固定为头的0.3倍。

| V2成功更新 | 目的 | 骨干学习率 | 各头学习率 | 噪声上限 |
|---|---|---|---|---|
| 1–1000 | 新数据和新优化器适应 | `3e-6 → 3e-5`，线性warmup | `1e-5 → 1e-4` | 100 |
| 1001–10000 | 扩展强噪覆盖 | 随统一余弦曲线下降 | 随统一余弦曲线下降 | 100→225 |
| 10001–30000 | 在完整范围联合细化 | 连续下降至 `3e-6` | 连续下降至 `1e-5` | 225 |

1000步之后到30000步使用一条连续余弦曲线，不在10k边界重启学习率。具体为：

```text
head_lr(k<=1000) = 1e-5 + (1e-4 - 1e-5)*(k-1)/999
head_lr(k>1000)  = 1e-5 + (1e-4 - 1e-5)*(1+cos(pi*(k-1000)/29000))/2
backbone_lr(k)  = 0.3 * head_lr(k)
```

V1峰值为全模型 `3e-4`；此处头峰值是其1/3、骨干峰值为其1/10。目的是在复用已有能力时限制表示漂移，让新场景和高噪声产生渐进适配。具体比例需要验证，不声称它能单独解决多任务梯度冲突。

30000次更新共 **480000对**，其中 **288000 MegaDepth、192000 COCO**。V1 C阶段平均更新耗时3.564秒，线性外推仅更新部分约29.7小时；MegaDepth较大JPEG的解码及新增验证会改变实际耗时，最终以V2 GPU smoke计时估计。

全程固定 `λ_det=1、λ_match=1、λ_gray=10`，按上述调度完成单次30k联合训练，并保存固定验证表现最好的检查点和最终检查点。

## 5. 验证与选模

### 5.1 先保留旧口径，再增加新口径

训练前在固定缓存上记录step0初始化模型，之后每2000步完整验证、每1000步小诊断及checkpoint，关键课程边界另外留权重。小诊断固定16个COCO源和24个MegaDepth源，后者按场景轮转；完整与子集曲线分开记录。

| 验证内容 | 用途 |
|---|---|
| 原COCO256源、clean+1/4/16/64/100 | 保留V1可比较的五噪声H-AUC@5，原manifest/cache只读 |
| 同COCO源新增ratio150/225，独立扩展manifest/cache | 测量旧内容在新噪声范围的变化，先评价V1作为step0参考 |
| MegaDepth保留的10场景×20源，clean+1/4/16/64/100/150/225 | 验证新增场景泛化，先场景内平均、再场景等权 |
| 选定权重后HPatches，另加150/225扩展 | 评估训练未见的真实图像对与低噪保留；不用于每轮选模 |

新增合成验证沿用当前固定COCO的配对曝光规则和其余几何参数；旧六档样本保持不变。分别报告 `(r,r)` 与曝光不对称子集，并按**视角实际ratio**统计检测，避免把含ratio1视角的高噪桶当成纯强噪检测能力。HPatches已有V1报告，属于训练未见的公开评估集，不把它描述成完全未知的盲测。

同时记录 H-AUC@1/@3/@5、1px/3px召回、phase Top1、匹配正确率、正确匹配数及失败/大误差尾部。新增150/225两档，不能用“七档均值”直接对比V1的“五档均值”。JPEG→InvISP→ELD仍是合成域；真实低照泛化需另用实际数据检查。

### 5.2 训练前固定选模规则

以七个含噪档 `1/4/16/64/100/150/225` 等权平均定义各域分数，再取：

```text
S_V2 = 0.6 * S_MegaDepth_scene_mean + 0.4 * S_COCO_source_mean
```

这里6:4是显式选择的验证权重，不由两个验证集的图数自动决定。step0及所有完整验证checkpoint一起参与选择。工程容差为原COCO五档主分数、COCO clean H-AUC@5相对step0各下降不超过 **1个百分点**；在满足容差的检查点中选 `S_V2` 最高者，同分取较早步，没有改善时保留V1。容差在训练前固定，并结合1/3px和强噪尾部判断是否达到发布目标，不能事后为某检查点改规则。

发布比较在COCO按源、MegaDepth按场景、HPatches按序列做配对重采样。训练内反复选模的验证提升不能当成独立测试证据；只在选定权重后读取保留测试集。除了综合分数，还应完整呈现各数据域和噪声档的变化。

### 5.3 单次训练执行与记录

只运行上述一条30k训练，不增加消融run或中途分支。训练前GPU smoke用于核对数值、梯度、采样比例与恢复，不用于比较训练方案的效果。

全程记录两库实际比例、场景分布、实际ratio直方图、学习率、各项loss与固定验证。断点恢复继续使用已锁定的30k配置、课程和sample cursor。

## 6. 实现范围与执行顺序

1. 提取MegaDepth RGB，生成可解码图像/场景清单，落实白名单、去重与独立验证split。
2. 扩展数据选择与训练身份记录：两库采样、80对配额、场景权重、relative source ID；不再只hash文件basename。
3. 增加V2配置和权重初始化入口，明确区分初始化与严格resume；把noise ceiling从代码常量变为明确配置。
4. 实现全程联合、分组LR、100→225课程及单一对数均匀采样；采样计数增加 `(100,150]`、`(150,225]`，更新分析与诊断里的上限假设。
5. 新增MegaDepth与150/225验证清单/缓存，保留V1旧协议；训练前跑step0比较。
6. 在GPU上做短smoke，检查4×4累积、80对6:4、来源/场景/噪声计数、有限loss/梯度，以及中断恢复的请求/优化器一致性；只对这些新增行为写有意义的测试。
7. 锁定配置后由用户启动长训练，按预定checkpoint回传结果再分析；全量RGB准备与30k训练仍由用户执行，本轮只在隔离数据上做短测试。

不改 `rawfeat/model.py`、损失公式、教师/InvISP权重、相机标定、匹配器或部署输出。V2的直接目标是：利用新增地点和纹理分布提高泛化，扩展到225，并保留V1已有的低噪和定位能力。

## 7. 本轮核查范围

本轮实际执行：`git status --short`、`git diff --stat`、针对配置/数据/调度/训练/验证的`rg`与`sed`读取；`python3`读取V1全部scalar与14个完整诊断、计算采样概率与阶段耗时；读取现有分片SHA审计JSON、统计RGB目录，并只读取原包前12个tar条目。以上是只读数据分析，未用CPU替代任何GPU训练或推理。

文档与工作区检查命令在交付前执行并记录到本任务`progress.md`：`git diff --check`、`git diff --exit-code -- rawfeat configs tests`、`python3 -m json.tool .planning/2026-10-17-v2-training-optimization-design/v1_v2_evidence.json`，以及新增Markdown的UTF-8、围栏和本地链接检查。尚未验证：完整RGB解压/解码、实际训练场景覆盖、新采样器和V2 GPU效果。

用户修订后的225课程与分布计算单独记录在 `.planning/2026-10-17-v2-training-optimization-design/v2_ratio225_schedule.json`。旧证据JSON中的150及额外强噪模式计算仅为初稿历史，不是当前训练配置。
