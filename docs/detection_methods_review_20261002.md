# 检测剩余问题：成熟方法查阅与适配判断

日期：2026-10-02。本文是文献与源码证据报告，不是并列执行方案。当前唯一方案仍为 [统一设计](low_light_raw_feature_extractor_design.md)，研究后的优先级归入其§18。本次只查阅、分析和整理文档，未启动GPU任务、训练或新测试，未修改模型、损失、配置、检查点或缓存。此前32项测试通过属于上一轮结果。

## 1. 研究改变了哪些判断

四个问题都有已有方法可借鉴，但成熟方法在原任务上有效，不等于已经验证适合 packed Bayer、强噪声和本项目预算。当前建议是：**保留质量归一化位置项；先用训练数据处理BN统计与批次依赖；任务竞争优先采用检测先学、随后联合微调和显式配比；有无判断仍弱时采用分层平衡；精细定位仍弱时引入直接几何监督或检测专属相位路径。**

此前“两组各新增500步：正常匹配 vs S3.detach()”应降为有条件的因果诊断，不能作为研究之后不加判断的下一步。永久detach也不是首选最终训练方法。

| 剩余问题 | 最相关的已有做法 | RawFeat推荐 | 适配边界 |
|---|---|---|---|
| train/eval差距 | PreciseBN；RepSR后期使用总体统计训练 | 先固定权重重估训练域统计；必要时冻结统计适配 | 零更新重估能查统计问题，不能修复所有批次依赖；不使用验证输入校准 |
| 检测与匹配竞争 | SuperPoint检测预训练→联合微调；PCGrad/CAGrad | 先分阶段或减小固定匹配系数 | 原论文系数不能照搬到当前InfoNCE；detach只作诊断/过渡 |
| 前景/背景判断弱 | XFeat限制无点样本；Focal/OHEM/QFL | 保留软教师目标，对完整Bernoulli项按显著/背景cell分层平均 | 平衡方式是本项目适配；先确认分组梯度，不先把软标签硬化 |
| 精细定位弱 | ALIKE可微坐标+重投影；XFeat相位保留检测支路 | 先检查局部分数信息，再选一个几何目标或结构候选 | 简单soft-argmax可能变差；原始灰度unfold不能直接替代带噪Bayer表示 |

## 2. 本地证据及修正

完整1536对、相同修复后缓存身份下，原KL/质量归一化候选/MeanAD+SuperPoint的五噪声档等权H-AUC@5为0.7472%/28.5211%/65.6030%。候选有效，但clean单应矩阵角点误差中位数仍为3.2648px，基线0.4883px；候选仅2/256个clean案例≤1px，基线216/256。这里是几何精度，不能写成关键点1px召回。[实测报告](detection_followup_report_20261002.md)、[原始逐对结果目录](../outputs/checks/detection_followup_20261002/)

固定B检测单任务1000步，在相同样本/合批下train/eval教师1px召回约84.49%/51.03%，相位Top1约80.35%/34.76%。但该训练每次前向只有4对、8视角，并重复相同两组；累积4次只增大优化器batch，**不增大BN的统计batch**。高train召回可以依赖批次组成，因此只能说“这些特定批统计前向能够表达细位置”，不能据此排除部署表示或泛化问题。[B逐检查点诊断](../outputs/checks/detection_followup_20261002/B/learning.json)、[固定实验实现](../scripts/detection_followup.py)、[BN官方定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)

当前检测与描述子均使用S3，灰度只使用S1/S2。匹配在给定教师对应点采样描述子，不经过学生提取坐标，因此没有直接监督检测峰值的几何位置。所有参数一起clip5，再由AdamW更新。[模型](../rawfeat/model.py)、[损失](../rawfeat/losses.py)、[训练](../rawfeat/training.py)

累计7209更新是多个实验的合计，不能当作在线候选训练了7209步；在线候选实际止于1000，完整噪声范围稳定后仅约100步。学习不足仍是合理可能性，但继续训练应验证一个明确改进策略，而不是用时长替代原因分析。[实测报告](detection_followup_report_20261002.md)

## 3. BN：成熟、低成本且与当前部署兼容的第一步

《Rethinking “Batch” in BatchNorm》区分优化器batch与归一化batch，并研究固定权重后重估总体统计的PreciseBN。它有公开实现，适合在新增训练之前检查EMA统计滞后。对RawFeat先保留原检查点，在副本上改变BN统计，比较正常eval与不同批组成的batch-stat前向。[论文§3–4及附录](https://arxiv.org/html/2105.07576v1)、[fvcore实现](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/precise_bn.py)

具体要求：只使用train2017生成的输入；线上模型覆盖预先声明的训练/部署噪声分布，固定clean模型只用于固定集机制诊断。预先限定校准样本数与forward次数；冻结学习权重、no_grad；保留原buffers和原始评估结果；完成后用普通eval和当前融合公式评估。不得用正式1536对验证输入估计统计，也不得把固定clean统计复制给噪声模型。

当前fvcore采用激活数量加权的总体二阶矩合并，包含跨batch均值差。PyTorch的SWA `update_bn` 是另一种常见重估实现，平均各batch统计；异质/不同大小batch时两者不等价。单次重估的深层输入仍受上游batch归一化影响，因此重估不是保证恢复全部train质量的算子。[fvcore源码](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/precise_bn.py)、[PyTorch源码](https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py)

若重估后仍有明显批组成依赖，训练侧已有相近先例：RepSR在低层视觉的重参数化网络中，先正常BN训练，后期切到总体统计让权重适应部署前向。可据此选择有限“冻结统计、继续适配”候选，保持卷积融合。普通BN.eval只冻结统计，gamma/beta仍可训练；不能与冻结全部affine的FrozenBN混称。[RepSR论文](https://arxiv.org/abs/2205.05671)、[Detectron2 FrozenBN源码](https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py)

增大实际microbatch也可减轻统计噪声；增加累积次数无法达到这个作用。GroupNorm使用输入相关的样本内统计，不能按现有RepVGG公式折叠为固定卷积；Batch Renormalization保留总体统计推理，但要增加训练修正与调度。两者是后续替代项，当前不优先。[GroupNorm论文](https://arxiv.org/abs/1803.08494)、[Batch Renormalization论文](https://proceedings.neurips.cc/paper_files/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf)

## 4. 任务竞争：先采用同类任务的简单成熟路线

SuperPoint先学习检测器，再加入描述子联合学习，并显式设定检测/描述子损失配比。这比直接使用复杂多任务优化器更接近我们的任务。其描述子目标是稠密hinge，与当前逐对应平均、温度0.1的InfoNCE不同，不能照抄其λ=1e-4等数值。作者公开仓库没有训练代码，因此这里的阶段依据来自论文，不声称核验了官方训练脚本。[论文§6](https://arxiv.org/html/1712.07629v4)、[作者说明](https://github.com/magicleap/SuperPointPretrainedNetwork#additional-notes)

本项目推荐：未来从头路线采用检测优先，再渐进加入匹配；已有候选则不必为复刻阶段而丢弃当前1000步模型。BN口径稳定后，优先选择一个预先固定的较低匹配权重做有限继续验证，或在确需隔离因果时采用暂时detach。具体权重依据完整累积batch的梯度与裁剪情况选择，不在正式验证集上无限扫系数。gray10仍未标定。

`descriptor(S3.detach())`让描述子头继续学习，同时阻断匹配对骨干的直接梯度，但骨干不再按匹配需求改善。它还不能彻底隔离检测更新：描述子头梯度仍进入全局clip范数；AdamW还有历史矩。应记录完整累积后的共享/各头梯度、clip factor与频率，并依据相同优化器状态分析更新，不能只看首个microbatch余弦或重置优化器后宣称隔离成功。[detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)、[全局裁剪](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)、[AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

PCGrad对冲突梯度作投影，CAGrad在平均目标附近约束任务改善，都是有原论文和公开实现的方法。但方向冲突不自动等于实际质量受损；还需额外逐任务反向与累积处理。灰度是辅助目标，也不自然对应三个任务同等优先。先用阶段/固定配比，无法兼顾时再选择其中一种。[PCGrad论文](https://arxiv.org/abs/2001.06782)、[作者实现](https://github.com/tianheyu927/PCGrad)、[CAGrad论文](https://arxiv.org/abs/2110.14048)、[作者实现](https://github.com/Cranial-XIX/CAGrad)

GradNorm若以S3做三任务平衡，会遇到灰度梯度为零；需选择真正共享的S2或只平衡检测/匹配。不确定性加权也不能不加论证地套到已经重加权的KL、灰度与InfoNCE上；尤其学习权重时，KL与CE之间的教师熵常数会影响权重动力学。这些是推导出的适配限制，不是算法本身无效。[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)、[不确定性加权](https://arxiv.org/abs/1705.07115)

## 5. 前景/背景：先分层平衡完整软目标

质量归一化位置项已有实测支持。DKD也揭示标准蒸馏中条件分布项被教师概率抑制，并将两项解耦加权，是相近的成熟思路。但是DKD的目标/非目标分类、逐样本解耦，与我们dustbin/64位置分组及逐视角质量归一化不是同一公式。[DKD论文§3](https://arxiv.org/html/2203.08679v2)、[作者实现](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py)

位置项修正后，有无项仍按全部有效cell平均，容易被大量简单背景占据。XFeat同样使用65类cell头，并限制无关键点样本数量处理不平衡；Focal Loss和OHEM分别通过减少简单样本贡献、挖掘困难样本处理密集检测问题。它们提供机制先例，而不是证明我们应该把教师软分布全部改成硬点。[XFeat§3.3](https://arxiv.org/html/2404.19174v1#S3.SS3)、[Focal Loss](https://arxiv.org/abs/1708.02002)、[OHEM](https://arxiv.org/abs/1604.03540)

最低侵入候选是保留每cell教师质量`m_T`，按教师显著/其余区域分别平均**整个**Bernoulli KL，然后用固定系数组合，位置项保持当前mass口径。例如：

\[
L_{occ}^{bal}=a\,\operatorname{mean}_{c\in F}D_{Bern}(m_{T,c}\Vert m_{S,c})
+b\,\operatorname{mean}_{c\in B}D_{Bern}(m_{T,c}\Vert m_{S,c}).
\]

这是借鉴不平衡处理的本项目候选，尚未训练验证。先统计两组cell数量、梯度和误差再选固定`a,b`，保持总尺度可比较。显著分组可用教师展开后的cell最大像素分数≥0.005，保留原软目标；不能只用`m_T>0.5`，已有固定诊断中它只覆盖44.56%的教师NMS点。若某组为空，只计算存在组并明确计数；不把缺失诊断报作零误差。困难背景选择放在简单分层后，避免同时改多个机制。

不要把软BCE的正负项分别任意加权后仍声称概率含义不变。对目标`y`，`-αy log p-β(1-y)log(1-p)`的最优值是：

\[
p^*=\frac{\alpha y}{\alpha y+\beta(1-y)}.
\]

`α≠β`通常改变教师目标的最优预测；给整个cell的KL乘教师定义的正权重，则保留其逐cell最小值位置。这是数学推导，不是文献已经验证RawFeat的结论；共享有限模型的实际折中仍会改变。

QFL支持连续质量标签，官方tensor-target实现以`|y-p|^β`调制软BCE，是可选的困难软目标方法；原论文质量标签是IoU，移到教师关键点质量仍需适配。优先用简单分层，当前不同时搜索focal指数、硬标签和阈值。[Generalized Focal Loss论文](https://arxiv.org/abs/2006.04388)、[MMDetection官方实现](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/gfocal_loss.py)

还需修正早先“前向KL的最优解就是摊平”的表述：对一个确定教师分布且模型自由表达时，最小值在学生等于教师，并不必然平坦；多个不确定目标、表示不足或优化受限才可能得到平均化响应。已确认的是位置监督权重不足的收益证据，不能由KL方向独自断定全部根因。

## 6. 精细定位：成熟的直接监督与结构分工

ALIKE用NMS候选周围的可微局部坐标、双向重投影损失和dispersity peak loss训练准确位置；ALIKED沿用该检测机制，均有作者实现。RawFeat已有homography、有效区域和浮点描述子采样接口，可以保留mass KL，添加局部几何监督，而不必更换整网。需要定义有效候选、对应关系、边界和匹配半径；峰形约束只施加在可信候选，不能逼背景无条件产生尖峰。[ALIKE§III-B/C](https://arxiv.org/html/2112.02906v1#S3.SS2)、[ALIKE源码](https://github.com/Shiaoming/ALIKE/blob/master/soft_detect.py)、[ALIKED§III-C](https://arxiv.org/html/2304.03608v1#S3.SS3)

结构方面，XFeat官方模型的65类检测头直接读取灰度8×8展开，深骨干另供描述子/reliability，这是保留相位与任务分工的直接先例。RawFeat可借鉴为S1/S2到检测头的相位保留短路径，而非增大整个骨干。但原始packed Bayer含CFA和强噪声，极小灰度感受野不能直接照搬；应保留去噪上下文、明确倍率/相位次序，一次只验证一个结构。[XFeat官方forward](https://github.com/verlab/accelerated_features/blob/main/modules/model.py#L113-L142)

纯后处理soft-argmax不是可靠的通用修复。Keypt2Subpx的MegaDepth/MNN消融中，SuperPoint加简单soft-argmax，pose AUC@5从35.34%降至19.56%，加入可学习局部模块后才改善。这不预言RawFeat会下降，但否定“加亚像素就一定更准”。基线SuperPoint整数坐标已能取得clean几何误差中位数约0.49px，候选数像素误差不能主要归因于整数输出。[Keypt2Subpx Table 6](https://arxiv.org/html/2407.11668v1)、[作者实现](https://github.com/KimSinjeong/keypt2subpx)

先在冻结检查点做有界局部诊断：候选identity、阈值/NMS/Top-K保持，用原始未清零score窗口仅改变坐标并重采样描述子；分开记录教师位置窗口的oracle诊断与真正自动提取结果。原始像素概率是`m_cell*q(position|cell)`；不能展开64原始logits后跨cell softmax，因为每cell任意公共logit平移不改变原预测，却改变跨cell权重。用解码概率或log概率定义权重，并预先固定口径。概率再套`softmax(P/T)`可能因分数很小而趋于均匀，与`P/sum(P)`不同。这些限制直接来自当前分数定义。

DARK的Gaussian热图与Taylor精修、UDP的坐标处理是成熟的人体姿态方法；当前多兴趣点/dustbin分布不满足其整套热图假设。它们提示继续遵守坐标契约，不支持无证据改变已经测试的半像素约定。[DARK](https://arxiv.org/html/1910.06278v1)、[UDP](https://arxiv.org/abs/1911.07524)

## 7. 研究后的推进顺序

1. **零训练更新优先**：在副本检查线上BN模式差、相同图像不同批组成的敏感性；用训练数据固定预算重估统计；核查正常eval和部署融合等价性。固定B只作机制参照，不代替线上结论。
2. **稳定口径后检查任务更新**：完整累积batch的加权梯度、全局裁剪及AdamW历史状态；分解显著/背景有无误差和位置误差；做一个预定局部定位探针。这些读数用于选择策略，不以范数相等或熵更低作为目标。
3. **只选一个必要训练修正**：BN问题突出时先冻结统计适配；否则优先检测主导的固定较低匹配系数/阶段策略。两组各新增500步仍可作有限预算建议，但不应先机械启动detach对照，更不同时叠加新loss与新结构。
4. **按残留问题分支**：背景错误主导→完整Bernoulli项分层平衡；定位主导且局部窗口有信息→局部可微几何目标；相位表示始终不足→一个检测专属相位路径。简单路线无法兼顾任务时才选PCGrad/CAGrad。
5. **验收同时看实际几何与分项**：正常eval下1px/3px召回、固定K、背景、相位与定位误差；已知点检索作辅助；自动点匹配及同身份完整1536对H-AUC作联合质量。增加点数、改善train分数或降低loss不能单独判成功；反复查看的小子集需独立留出复核。

用户允许充分检查后必要的有限约2000步实验，该授权不改变；仍不得无依据重复或启动全量正式训练。本次研究新增优化器更新为0。研究提供有来源的策略选择，后续最小验证只回答适配是否成立，不以继续训练本身代替设计。
