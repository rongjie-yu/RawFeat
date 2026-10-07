# 低光Raw特征提取器：当前分阶段全量训练方案

更新日期：2026-10-02。本文件是唯一当前目标方案；历史设计已移至[归档快照](archive/design_before_formal_staged_20261002.md)。用户已选择保留分阶段方法，并计划手动启动一次全量训练。当前代码、配置清理与有限短训准备已经完成，执行证据及手动命令见[准备报告](staged_preparation_report_20261002.md)。本文定义唯一训练目标；完整训练仍未启动。

## 0. 当前决策与执行边界

唯一训练路线：**检测与降噪共同学习 → 渐进接入匹配 → 保持检测约束的联合训练**。从随机初始化开始，使用下表唯一正式预算：

| 阶段 | 已完成更新数k | 本阶段更新数 | 训练目标 |
|---|---:|---:|---|
| A 检测与降噪 | 1–50000 | 50000 | mass检测 + 10×gray；描述子暂停 |
| B 匹配接入 | 50001–55000 | 5000 | 检测、gray持续；匹配系数0→1 |
| C 联合细化 | 55001–100000 | 45000 | mass检测 + 匹配 + 10×gray |

总计100000次优化器更新，不是每阶段再训练100000次。正式run从step0开始，历史9209总账另列；同run恢复不重置计数。一个更新包含4个microbatch，每个4对、8视角，总有效batch16对。阶段边界由绝对全局计数决定，不能在恢复时重新预热或重新开始阶段。

- 当前选用：质量归一化位置检测损失、正常BN、现有RepVGG结构、固定gray10、ratio上限100。
- 本次选择分阶段是用户的设计决定；有限巩固实验未证明它优于持续联合，不能把选择表述成已验证优势。
- gray10及阶段比例是本次完整运行的预定设置，尚未标定最优；不在准备阶段自动搜索替代值。
- 新会话只整理代码、审查、修复和有限短训，并交付完整命令。**全量训练由用户手动启动，代理不得启动、后台排队或自动恢复全量任务。**
- 设计制定会话仅形成方案与交接；本次实现清理、审查、测试和600更新短训的事实由准备报告单独记录。
- 准备任务原文见[执行prompt](detection_followup_prompt.md)，执行状态见准备报告；历史报告只作为证据，不构成并列方案。

## 1. 检测预训练时长的依据

[SuperPoint原论文§6](https://arxiv.org/html/1712.07629v4)明确记载MagicPoint在Synthetic Shapes上进行了200000次检测训练，随后用Homographic Adaptation适配真实图像，再联合学习描述子。论文使用batch32、Adam学习率0.001；这些数值不是当前RawFeat的默认配置。论文没有给出可直接移植的RawFeat检测/联合比例，作者[官方仓库说明](https://github.com/magicleap/SuperPointPretrainedNetwork#additional-notes)也明确未发布训练和评估代码。

因此本方案借鉴“先充分学习检测，再联合”的顺序，采用50000次检测+gray训练，**不是声称50000等价于论文200000**。RawFeat采用已训练教师、真实COCO图像、带噪Bayer、不同损失/分辨率/更新单位，直接搬用论文步数没有充分依据。

沿用噪声课程在40000步达到ratio100；A阶段随后还有10000步只学习完整噪声范围下的检测与恢复，再在50000步接入匹配。B/C合计50000次匹配活跃更新，保留10万步总预算。A阶段约800000个图像对请求、1600000个视角；这是数据曝光量，不是独立图像数。

阶段时长在启动前固定。记录曲线供最终分析，不根据验证结果自动无限延长阶段。到100000步停止；未来是否增加预算另行讨论。

## 2. 数据、增强和数值域

### 2.1 数据与尺寸

- 训练：完整COCO train2017，118287个源图像；验证：固定COCO val2017的256源，与训练源不重合。
- Bayer原图高480、宽640；输入为4×240×320，固定RGGB，打包顺序[R,G1,G2,B]。
- 保持宽高比缩放到覆盖裁剪，额外倍率1.0–1.2；训练随机裁剪，验证固定裁剪。尺寸均满足8的倍数。
- 每个请求的源图像、裁剪、外观、几何、噪声和对应点seed由训练seed与绝对sample_number决定；阶段切换不能改变已定采样流。

### 2.2 每视角独立sRGB外观增强

最多选择一种操作，Albumentations锁定现有已核对版本：

| 操作 | 概率 | 参数 |
|---|---:|---|
| NoOp | 70% | 保留外观 |
| 亮度 | 5% | 0.9–1.1 |
| 对比度 | 5% | 0.85–1.15 |
| Gamma | 5% | 0.85–1.15 |
| 饱和度 | 5% | 0.8–1.2 |
| PlasmaShadow | 5% | 强度0.1–0.35，roughness0.7 |
| 模糊 | 5% | Gaussian与Motion各半 |

Gaussian sigma0.3–1.0；Motion核3/5、角度0–180度、direction0、allow_shifted=False。NoOp仍包含几何和Raw噪声。保留当前成熟库实现，不添加sRGB噪声、额外压缩或其他增强。

### 2.3 InvISP、几何与Bayer

每视角的同一增强结果用于教师sRGB灰度和冻结Canon InvISP。先获得连续三通道sensor RGB，与对应sRGB使用同一warp，再采样Bayer、添加ELD；不直接warp马赛克或已生成噪声。

Canon InvISP输出限于[0,1]后作2.2次幂并乘4095，除RGB白平衡[1,1024/2020,1458/2020]，按14335/4095映射；clean信号限于[0,14335]后加黑电平2048。白电平16383。官方资产与版本依据见[asset_provenance](asset_provenance.md)，不得把InvISP输出误当作已经减黑的线性Raw。

A为基础裁剪；B使用Kornia仿射再轻透视：旋转±30度、缩放0.8–1.2、平移各轴≤10%、distortion_scale0.15。共同覆盖各视角至少50%；保存H_A_to_B及裁剪/缩放关系。warp沿用align_corners=True。有效区域收缩8个原图像素，检测cell必须64像素全部有效。保持现有坐标与掩码定义。

### 2.4 曝光和ELD

沿用已修复随机流的full-DN模型：

\[
x_{noisy}=r[x_{clean}/r+n_{ELD}(x_{clean}/r)].
\]

x_clean包含黑电平；ELD在该域生成Poisson散粒、Tukey读出、行及一个DN单位量化噪声。Poisson/read/row/quant使用已验证的独立确定随机流；不同视角也独立。先乘回ratio，再减2048、除14335、打包并乘四通道WB[1,1024/2020,1024/2020,1458/2020]。不重复曝光归一化或减黑，不裁剪带噪输入，保留负值。

不因清理代码改变此物理路径、标定、量化单位或独立流。测试覆盖无噪声恒等、DN/packed域、负值、ratio100、分项独立性和有限反向。物理/协议变化必须先作为确认的差异处理，不能隐藏在简化中。

### 2.5 gray目标

从相同warp后的clean sensor RGB减黑、归一化并乘WB，取全分辨率：

\[
Y=0.299R+0.587G+0.114B.
\]

目标保留外观增强，只引导传感器噪声恢复；不要求撤销阴影或模糊。不要用低分辨率packed灰度替代监督目标。

## 3. 网络与梯度范围

| 模块 | 相对原图尺寸 | 通道 | 块数 |
|---|---|---:|---:|
| 输入 | H/2×W/2 | 4 | — |
| S1 | H/2×W/2 | 24 | 2 |
| S2 | H/4×W/4 | 48 | 3，首块stride2 |
| S3 | H/8×W/8 | 96 | 4，首块stride2 |
| 检测头 | H/8×W/8 | 128→65 | 独立RepVGG块+1×1线性输出 |
| 描述子头 | H/8×W/8 | 128→128 | 独立RepVGG块+1×1线性输出 |

RepVGG训练分支为3×3+BN、1×1+BN及适用的identity+BN，相加后ReLU；正常训练BN eps1e-5/momentum0.1。部署eval固定统计，融合单个带偏置3×3卷积。4对/8视角是实际BN batch，累积4不扩大其统计batch。

gray：S2经1×1降到24、双线性上采样到S1（align_corners=False），与S1拼接；3×3卷积24+ReLU；1×1输出4，再pixel_shuffle(2)恢复H×W。它直接监督S1/S2，S3间接受益。部署删除gray头。**不声称gray loss下降就证明共享特征已充分降噪。**

检测与匹配可反向到S1/S2/S3，gray直接到S1/S2、到S3为0。当前不加入跳连、相位浅路径、ALIKE坐标监督、注意力或新归一化。

初始化：学生全部随机初始化；普通Conv用Kaiming，三个头最终Conv用Xavier gain0.1/bias0，BN gamma1/beta0。不加载历史mass2000或短训权重启动正式阶段A；历史检查点仅作参考。

## 4. 唯一监督目标

### 4.1 mass检测损失

教师为冻结官方MagicLeap SuperPoint，在当前视角的无ELD sRGB灰度上运行。teacher logits detach，温度1，保留dustbin；教师描述子不用于向量蒸馏。

每cell定义m=sum(P[:64])、q=P[:64]/m。每视角有效cell数N、教师总质量M：

\[
L_{occ}=\frac{\sum_c v_c KL_{Bern}(m_{T,c}\Vert m_{S,c})}{N},
\quad
L_{pos}=\frac{\sum_c v_c m_{T,c}KL(q_{T,c}\Vert q_{S,c})}{\max(M,1)}.
\]

\[
L_{det}=\operatorname{mean}_{pairs,views}(L_{occ}+L_{pos}).
\]

使用log_softmax/logsumexp稳定计算，条件位置系数1。这是唯一当前检测训练目标：新实现删除原65类KL训练分支和objective选择器，不保留可切换历史方案。分解项用于当前诊断，独立数学参考可写在测试中。

### 4.2 匹配损失

每对最多512对应点：教师点75%、空间均匀点25%，最小间距8个原图像素。双视角教师点均衡选取，用homography产生精确对应，过滤无效/重复/过近点，不复制凑数量。

网格与插值后描述子分别L2归一化。坐标归一化使用(points+0.5)/[W,H]×2−1，grid_sample双线性、align_corners=False；不要将其改成warp的align_corners=True。

\[
S_{ij}=a_i^Tb_j/0.1,\quad
L_{match}=-\frac{1}{2N}\sum_i[
\log softmax_{row}(S)_{ii}+\log softmax_{col}(S)_{ii}].
\]

每对对应平均、双向等权，再batch等权。目标来自教师/均匀对应，不经过学生NMS坐标；记录这一定位监督边界。

### 4.3 gray辅助降噪

\[
w_p=(Y_p+0.01)^{-2},\quad
L_{gray}=\frac{\sum_p m_pw_p(\hat Y_p-Y_p)^2}{\sum_p m_pw_p}.
\]

逐视角归一化再对pairs/views等权。权重来自clean目标。全程λ_gray=10；删除旧10→1灰度衰减调度。普通MSE、边缘和抗噪指标用于诊断，不成为新损失。

\[
L(k)=L_{det}+\lambda_m(k)L_{match}+10L_{gray}.
\]

A阶段匹配没有计算时日志记录null/available=false，不能用0伪装“真实匹配loss为零”。

## 5. 正式阶段与学习率

k是本次已完成更新数，step_index从0开始；计算下一次更新的调度时用k=step_index+1：

\[
\lambda_m(k)=
\begin{cases}
0,&1\le k\le50000\\
(k-50000)/5000,&50000<k\le55000\\
1,&55000<k\le100000.
\end{cases}
\]

A跳过描述子训练前向与匹配计算，descriptor requires_grad=False、BN.eval、grad=None；参数/BN buffers不改变，AdamW不推进其状态、不施加weight decay。描述子已在启动时初始化并注册，不能阶段切换时重新随机初始化。

B/C恢复描述子训练和正常BN，检测及gray持续。共享骨干正常训练，灰度不detach。恢复优化器时所有参数仍必须存在，逐参数Adam步数不同是预期：完整结束共享/检测/gray为100000，描述子为50000。不能假定所有参数step相同。

学习率：
- k=1..2000：lr=3e-6+(3e-4−3e-6)×(k−1)/1999，首步为初始值，k=2000正好达到峰值。smoke同样按(k−1)/(warmup−1)计算。
- k=2001..50000：保持3e-4。
- k=50001..100000：cosine衰减至3e-6：
\[
lr(k)=3e-6+\frac{3e-4-3e-6}{2}
[1+\cos(\pi(k-50000)/50000)].
\]
- 所有活跃参数使用同一lr。阶段切换不重置AdamW、RNG、游标，不另做隐含warm-up；匹配系数渐进接入即本次接入策略。

AdamW betas(.9,.999)/eps1e-8，Conv weight decay1e-4、其他0。每micro loss除4，累积后全学生参数clip5一次、step一次，zero_grad(set_to_none=True)。保留原有全局裁剪；不能在本次偷偷换分组裁剪。

## 6. 噪声课程

设a(k)=0在k≤5000，a(k)=1在k≥40000，中间：

\[
a(k)=\frac{1-\cos(\pi(k-5000)/35000)}2,\quad
r_{max}(k)=\exp(a(k)\ln100).
\]

每视角独立从log-uniform[1,r_max(k)]采样。k=1..5000仍含ratio1的ELD噪声，不是clean；40000到100000维持完整[1,100]。匹配接入时不再同时改变噪声范围。阶段A的gray必须接触真实训练噪声。

不引入ratio256、全阶段clean预训练或额外随机曝光模型。数据请求由绝对sample_number和seed决定，暂停描述子不能改变采样结果。

## 7. 配置与运行状态

当前实现保留唯一正式配置configs/formal.yaml和一个同算法的压缩验证配置configs/smoke.yaml。不要保留calibration、旧short、原KL、冻结BN、旧阶段等并列可执行配置。

下表是必须对应到配置与调度的值；字段命名可以沿项目习惯简化，但只保留一套含义：

| 内容 | 正式 | smoke主训练 |
|---|---:|---:|
| 总更新 | 100000 | 600 |
| 检测+gray阶段 | 50000 | 300 |
| 匹配接入阶段 | 5000 | 100 |
| 联合阶段 | 45000 | 200 |
| lr warm-up | 2000 | 20 |
| lr峰值/最低 | 3e-4/3e-6 | 相同 |
| ratio开始扩展 | 5000 | 30 |
| ratio达到100 | 40000 | 240 |
| gray weight/epsilon | 10/.01 | 相同 |
| seed/batch/accum/clip | 42/4/4/5 | 相同 |

smoke同样先保持lr到检测阶段结束，再在匹配活跃阶段cosine衰减；只缩短边界和记录间隔。正式与smoke通过同一函数实现，无重复训练器、硬编码跳阶段、第二套loss或模型。

configs/formal.yaml已适配本表，并完成代码审查、数值/状态测试和同算法600更新smoke；详见准备报告。正式从头开始；smoke checkpoint不能作为正式初始化。恢复仅支持本次新schema、相同算法/config/数据身份，不要求兼容历史实验checkpoint。

## 8. 日志、诊断、保存与恢复

### 8.1 每次更新的必要记录

scalars.jsonl保留每次真实更新的一行；TensorBoard相同数值每20步写一次：
- global_step、phase、lr、matching/gray系数、sample_cursor、实际ratio min/mean/max及各noise范围计数；
- det/occupancy/mass-position、gray、匹配原始loss与加权贡献、total；未计算项记null；
- valid_cells/pixels、教师前景质量及位置放大倍数、匹配有效对应数（A未计算时null）；
- clip前global norm、clip factor、clip次数、更新耗时及峰值显存；
- 每个epoch等价曝光计数可作辅助，不能替代step。

run_manifest保存：完整解析配置、代码SHA清单、依赖/torch/CUDA版本、GPU型号、精度/TF32、seed、训练源清单hash、官方权重/标定hash、协议/cache/baseline身份、启动命令和时间。统计对齐绝对step；恢复追加，不覆盖旧曲线或重复计数。

不要每步把完整GPU输入搬到CPU存tensor hash。启动、边界、恢复抽检与故障样本保存输入/hash/seed；平时保留足以重建请求的cursor、固定seed和源清单。

### 8.2 固定诊断与梯度

每1000步在预先固定的16验证源、6桶上诊断；使用eval副本/no_grad，不改变训练BN或RNG。边界50000/55000/100000必须记录。
- 正式阈值和无阈值Top400/1024的教师1px/3px召回、点数与3px内条件定位误差；
- 条件熵、教师NMS位置相位Top1/Top5与dx/dy、质量分布；
- 按教师cell最大像素分数≥.005的显著/背景区域分报有无误差；计数与教师覆盖必报，空集合null；
- gray加权/普通MSE、边缘梯度误差/方向/能量、噪声下检测稳定性；按实际视角ratio和验证bucket分别记录，显示样本数。学生clean点保留率的参考集合变化需注明，不能单独判鲁棒性；
- B/C报告已知位置检索及实际自动点匹配，前者明确oracle属性。A描述子未训练，不用其随机匹配评价检测。

每5000步及阶段边界取一个完整4micro累积输入，记录各任务在S1/S2/S3与各头的加权梯度范数、方向、global clip，并保留Adam步数/历史状态检查。探针不能更新训练模型。A的匹配梯度缺失是结构性缺失，不报虚假的匹配质量0。

### 8.3 checkpoint与恢复

每1000步保存恢复checkpoint，边界50000/55000/100000强制保存；最近3个普通恢复点轮换，边界、best_det、best_geometry和最终点保留。验证分析摘要不删除。保存采用临时文件+原子替换。

checkpoint含完整student（BN buffers）、AdamW、global_step、phase/下一lr及系数、sample_cursor、Python/NumPy/torch CPU/CUDA RNG、配置/源/代码/资产身份、计数、验证摘要和best值。从k继续时下一更新必须为k+1，不重复数据或重做阶段。A暂停参数的Adam状态必须一致恢复。

同型号物理GPU之间迁移可使用显式`--allow-gpu-change`，必须只暴露一个CUDA设备并恢复保存的单设备RNG状态。该开关只允许`CUDA_VISIBLE_DEVICES`变化及`rawfeat/gpu_resume_revision.json`精确登记的本次恢复代码修订；GPU型号、配置、训练数学、依赖、数据与资产仍严格检查。原始manifest/checkpoint保持不变，恢复额外保存来源/当前身份与差异记录，之后checkpoint使用真实当前身份。老run后续恢复仍保留此开关，以校验原manifest到当前运行的迁移。

边界继续用当前last状态，不自动回滚best_det权重或重置step。A的best_det按完整固定协议五噪声教师1px召回选择，保存对应固定K/背景指标；它只供分析，不能将A随机描述子H-AUC当作best_geometry。稳定联合阶段开始后按完整主分数保存best_geometry；最终同时报告last和best。

OOM/非有限：停止并保存最后可靠完整状态与故障请求/日志，不重放已成功更新、不用CPU补训练、不静默改batch/精度。无法完整保存当前累积时恢复最近完整checkpoint，并明确实际丢失/重算计数。不得把部分参数或BN已改变的failure_state冒充更新前状态。

### 8.4 必须可分析的曲线

由可恢复原始JSONL生成独立PNG/PDF图：
- det/occupancy/position、match、gray及total的原始与加权loss；
- lr、匹配系数、noise cap、clip与耗时；
- 分噪声teacher1px/3px、固定K、背景、相位和H-AUC；
- gray加权与普通误差、边缘指标；
- 标出50000/55000边界，显示raw点；平滑曲线只作附加，不掩盖接入回落。

训练期间至少在验证/阶段边界刷新曲线与summary；完整结束生成最终报告，比较各阶段、各噪声、best/last及基线，保留逐对结果和失败。最终分析不能只读一条total loss曲线。

## 9. 固定验证、基线与部署

沿用协议rawfeat.coco.fixed.v2：
- manifest：manifests/coco_val2017_seed2027.json，256源/1536对；
- cache：outputs/checks/detection_followup_20261002/validation_cache_rngsplit；
- baseline：outputs/checks/detection_followup_20261002/baseline_rngsplit/summary.json；
- 当前cache identity：d950a2368a67a363bf28ed8fe91029974eac05ede17225020d5fe2c0c0b1757d。

clean256对单报；噪声[1,4,16,64,100]各256对，半数双侧当前ratio、半数单侧ratio1，方向均衡。主分数为五噪声H-AUC@5等权，不含clean。TopK是同mask/border/NMS后至多K点，不改变主提取阈值。

原图尺度：阈值.005、NMS4、border8、max1024；MNN，无额外ratio test。RANSAC阈值3px、confidence.999、max_iterations10000，固定随机设置；失败计入，不删除。同时报告H-AUC@1/3/5、正确匹配、精度、重复性、点数及条件定位误差。H-AUC@1不是teacher1px召回。

正式每2000步和阶段边界运行完整256源评估：A只运行完整检测/gray统计，不产生用于选几何模型的随机描述子主分数；B/C输出完整自动匹配/几何，best_geometry从稳定联合阶段开始选择。16源诊断不能冒充完整结果。完整评估与诊断使用eval，不用于BN统计校准。

MeanAD+SuperPoint保持已核对Raw-SLAM路径：RGGB双线性demosaic→减黑/限幅、除16383、uint16编码/解码→RGB共同mean/MAD、(x−mean+2MAD)/(4MAD+1e-8)限于[0,1]→RGB灰度→官方SuperPoint。MeanAD不进入学生、教师或gray训练输入。当前完整基线65.6030%。

代码清理优先不改已确认的数据/噪声/几何/基线数学。cache与baseline含源码fingerprint，文字/接口重构也可能改变身份：不得忽略校验或把旧结果改标签当新结果。必要变化保留旧资产，建立新的明确身份并做对应等价/完整验证；不能为避免重算静默放宽身份检查。

导出只对当前schema的B/C模型或最终模型使用eval与RepVGG融合，删除gray。核查融合前后logits/descriptor、阈值/NMS选点与匹配/几何；给定容差和离散近同分边界说明。报告参数/MACs和网络/完整提取延迟范围。保持当前整数坐标提取，不加入局部质心后处理。

## 10. 代码清理与审查要求

本次整个自有源码已按当前方案整理，删除旧路线；清理、简化、审查及修复记录见准备报告。以下保留本次执行的审查契约。

保留当前数据、模型、唯一loss、阶段调度、训练、诊断绘图、验证/基线、推理导出、必要资产准备工具和对应测试。删除原KL选择器、冻结/重估BN实验路线、旧calibration/short模式与配置、历史对照driver/报告生成器，以及docs/files等候选Python副本；将仍必要的诊断能力并入当前模块。正式与smoke只是同算法的两组时长，不是两种训练方案。

历史Markdown报告、outputs中的检查点/结果/配置记录、缓存、官方依赖与真实数据保持；这些是不可变证据，不作为训练入口。不要把旧Python再复制到仓库另一个可执行目录形成第二套代码；无需旧checkpoint兼容层。删除前建立路径/SHA清单并记录理由，检查引用/入口/测试，无关用户变动不能覆盖。当前仓库大量文件可能untracked，git diff不能代表全部实际改动。

按用户要求先应用[code-simplifier技能](</homes/rongjie/.codex/skills/code-simplifier/SKILL.md>)，范围覆盖本次适配与必要整体清理。保持Python风格；技能中的React/ES modules条款不适用于本项目。用户明确要求移除旧方案，不能用“保留历史功能”反过来阻止清理。

然后使用[open-code-review-delegate技能](</homes/rongjie/.agents/skills/open-code-review-delegate/SKILL.md>)：
1. ocr delegate preview --format json确定文件，ocr delegate rule --format json获取规则。
2. 主代理读取diff或untracked完整文件并逐项审查，OCR不调用外部LLM。
3. 用(path,status)建立覆盖清单，每项reviewed或有具体理由的skipped；明确被排除的官方/generated/historical资产。
4. 修复全部确认的正确性、训练数学、恢复、数据身份、效率和维护问题；复查修复及相关上下文，直到无未解决的实质发现。测试通过不能替代代码审查。
5. 输出findings、修复记录及total/reviewed/skipped/coverage；仅运行CLI不能声称审查通过。
6. 若--format不受支持，仅在unknown flag错误时按技能回退文本；缺失CLI可按技能安装，不能绕过审查。超大背景用≤8000字符的忠实摘要，完整方案仍人工读取。

在清理、简化和修复都完成后最终复查，再开展短训。短训发现问题则修复、补查；任何影响实现的新修改都应再次审查，不能沿用修改前的通过结论。

## 11. 新会话有限短训及交付

主smoke从随机初始化运行600次，A300/B100/C200；checkpoint、系数、ratio和lr压缩到§7定义。预检、边界pilot、恢复一致性检查及短训中的真实学生优化器更新总额**不超过1000**，全额记账，历史9209单列；不通过重启再花600绕过上限。CPU数学单测与只读前向不计实训更新，实际GPU学生更新必须记录。

先做不更新的调度/数据/梯度检查，再进行必要少量pilot（计入预算）。用同schema验证A暂停、B/C接入、300/400边界及中途保存恢复，参数/BN/Adam/cursor/RNG正确。有限训练报告所有loss曲线、阶段系数、分噪声诊断、grad/clip和耗时，并在固定32源/192对上验证完整管线；这是子集，不称完整1536结果。

短训验收看实现正确、更新有效、数值有限、恢复一致、暂停/接入符合定义、没有明显输出塌缩、日志足以分析。不能要求600步达到基线，也不能由低短训分数自动否定本次长阶段方案；基础故障必须修复后再交付。

交付：
- 当前代码与唯一方案一致的差异/清理清单，所有历史入口移除证据；
- 必须的测试与准确结果、完整OCR覆盖与最终修复复查报告；
- 新600步短训及所有额外更新账本、真实配置、identity、曲线与失败；
- 完整run_manifest、参数/BN/恢复/资产预检结果；
- 用户手动执行的准确preflight、从头全量训练、同run恢复、最终验证/导出与曲线汇总命令；注明GPU选择和输出目录、解释器及日志位置；
- 基于smoke分阶段实测吞吐估计完整耗时及验证/存盘额外成本，明确只是估算；
- 明确“全量未启动”，不创建自动任务或隐藏后台进程。

目标命令接口维持python -m rawfeat.cli preflight/train/validate/export；实际参数由新会话用--help、短训和恢复核验后填写。这里不提供未经新实现验证的可直接启动命令。preflight ready不代表代理可以启动全量；用户手动启动权始终保留。

## 12. 已有证据与来源

| 证据 | 结论与限制 |
|---|---|
| 原KL vs mass1000完整主分数.7472% vs28.5211% | 位置质量归一化有受控收益，选择当前唯一检测目标 |
| 正常mass1500 43.4613%，冻结43.5347% | 冻结设置无稳定额外优势，当前不采用 |
| 持续联合mass2000 51.8136%，巩固策略51.5805% | 250+50+200未胜出，差−.233pp区间跨0；不是随机初始化预训练 |
| 同身份MeanAD+SP65.6030% | 当前完整质量仍有差距 |
| gray加权/边缘部分改善，普通MSE/稳定性未改善 | 尚未确认gray因果收益、10最优或共享充分降噪 |
| 最近50项测试、输入/hash/恢复核查 | 历史执行记录，不代表新清理实现已经测试或审查 |

历史总账9209，所有此前训练已停止。原报告保留：[检测复核](detection_followup_report_20261002.md)、[BN排查](detection_bn_residual_report_20261002.md)、[阶段/gray实测](detection_gray_staged_report_20261002.md)、[成熟方法查阅](detection_methods_review_20261002.md)。旧§16–20、旧默认KL与禁止未来正式训练的文字只在归档中保留，不覆盖本文件或用户最新指令。
