# 2026-10-01 设计复核与检测/灰度标定实测（本次有限复核与对照执行完成）

> 文档地位：历史设计复核与有限训练实测，只作为证据记录。当前唯一执行方案是 [统一方案与检测排查路线](low_light_raw_feature_extractor_design.md)，本轮授权与顺序见其§17；本文的旧范围、旧协议或旧限制不覆盖当前用户指令。

所有GPU检查使用物理GPU1 RTX5090，CUDA_VISIBLE_DEVICES=1使其成为cuda:0；Python为/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python。所有新产物在本目录，历史检查点、缓存、基线和报告保留。未启动正式100000步。

## 1. 设计与实现复核

完整重读docs/low_light_raw_feature_extractor_design.md，独立检查数据、数值域、几何、网络、监督、损失、优化器、调度、验证、恢复和部署；逐节证据见review_checklist.json。当前未确认新的训练核心数学/坐标错误。确认梯度测量日志缺项：旧grad_det/match/gray只是第一个共享参数，缺完整共享范数、S3检测/匹配余弦、检测/灰度和匹配/灰度分别余弦。rawfeat/training.py新增明确*_shared字段和缺失余弦；原字段保留历史语义，不改变训练数学或三条调度。新随机试训第一步已实际执行并验证共享范数与各阶段平方和一致，S3灰度梯度为结构性零。

测试中的旧256倍率已改为100，针对性3项通过；最后完整28项测试通过，数据/坐标检查64个cell类别全部解码正确，36个代表缓存有限且DN→packed误差≤1.19e-7。ratio100两对反向有限，输入保留负值，0次优化器更新。固定验证1536对及全部基线1536条/预测数核实，逐对重算基线65.95153886%，clean独立、不加入主分数。预检ready=true只证明工程资产一致。

新实际恢复：连续2步vs1步+恢复，累计执行4次更新；RNG、游标32、调度一致。学生参数最大差2.16e-5、优化器2.43e-6、前向最大7.38e-6；PyTorch本地文档明确CUDA grid_sample反向非确定性，不宣称逐位相等。历史检查点真实融合最大logits差1.049e-5、描述子5.443e-6，保存重载完全一致。

## 2. 历史状态与无训练诊断

检查点确为2000步，配置ratio_max=16；日志最大实际倍率15.953、上限16。没有ratio100/256训练历史。旧ratio256仅历史压力，未进入当前比较。

对同一旧检查点及固定前4源图像扫描0.005/0.002/0.001，其他规则完全相同；正式阈值保持0.005。clean每视角33.625/364.875/642.125点；教师3px召回2.172%/13.91%/17.84%；正确匹配每对11/101.25/165.5；匹配精度55.65%/44.45%/42.40%。top400/top1024对照召回14.83%/20.27%，正确匹配107/202，精度44.80%/38.42%。重复性、定位误差、RANSAC角点误差和各删除阶段逐对数据见offline_scan_v2/scan.json。

clean正式阈值筛选链平均：307200候选→35453.625过阈值→34621.25过边界→740.75过有效区域→33.625过NMS；数量上限删除0。无效区域高响应被正确排除，不能把这些候选算成真实检测点。有效区域已收缩8像素，边界与有效域不重复计数。

空间证据：条件64位置熵平均0.9968（教师0.7229），5%分位约0.989；教师位置类别在学生同cell中排名中位数的视角均值25.625，top1仅2.663%，top5为12.080%。不是严格平坦；argmax存在类别偏好（class7占17.25%，教师同相位约2.30%）。±8px局部峰均值偏移(+0.0665,+0.0158)，未观察固定方向偏移；64类解码与连续坐标warp检查通过。教师位置及±3/±8局部峰见response_*.json，代表图见spatial_detail/response_example.png。

逐cell dustbin误差MAE0.0852，教师显著cell为0.2402；KL精确分解前景/背景0.12626、条件位置加权0.12002，合计0.24628。平均dustbin接近不能证明区域背景识别正确。低分与弱空间辨别同时存在；降阈值明显增正确匹配，但精度下降，不能单独解决。ratio100正式正确匹配2.25/对、精度11.25%；降到0.002为38/23.05%，0.001为55/18.55%，强噪声新增点稳定性仍弱。

## 3. 固定样本学习检查与灰度对照

固定8对train2017样本（独立seed31415），缓存仅用于独立诊断，clean/ratio1共享增强和几何，ratio1噪声固定。完整网络/三项损失/FP32/4对×累积4/AdamW/clip5/初始化seed42，20步预热3e-6→3e-4后保持。初始化原始损失在1/3/10三个权重实验完全一致。

clean-gray10和ratio1-gray10各300步：召回7.265%/7.492%，正确匹配22.75/23.375；检测梯度及实际检测头/S3更新均非零。前期dustbin压低整体分数后150→300检测回升，不能以300步未高召回判实现错误。clean仅续至1000：召回23.73%，正确匹配62.125；train模式33.10%/98.125，存在后期模式差异，尚未充分拟合。

ratio1-gray1/3/10各300步：召回6.142%/6.352%/7.492%，正确匹配20.25/25.125/23.375；精度24.84%/30.23%/23.08%；共同96对留出验证子集主分数0.388%/2.140%/1.901%。64/100均很弱。只延长有竞争力的3和10至总计1000：召回21.76%/22.40%，正确匹配66.75/61.75，精度32.75%/33.06%，重复性22.23%/22.71%。共同96对子集分数2.431%/1.996%；强噪声无明确改善，不能据此定正式权重。

晚期共享梯度中位数(det/match/weightedgray)：lambda1=.1380/.4364/.00678；3=.11735/.43260/.01493；10=.12415/.35477/.03233。S1/S2/S3全部同一参数集合比较；S3灰度结构性无路径。持续检测/匹配负余弦见stage2_grad_summary和各scalars；灰度/检测多数正向，灰度不是晚期压倒检测的证据。不能要求梯度范数相等，也不能据total下降定权重。匹配竞争已观察到，但无改变三项损失的因果消融，不称唯一主因。

## 4. 随机在线与强噪声实测

lambda3/10各600步，从同一随机初始化seed42开始；同一全COCO train2017流，每次重新在线增强/InvISP/几何/噪声，4×4有效batch16，50步预热至3e-4后保持，固定log-uniform[1,100]诊断分布。它是独立的强噪声分布诊断，不是正式课程压缩或正式训练。每200步共同前16源图像/96对验证，最后两者都执行完整256源图像/1536对当前v2验证，clean单报，五个含噪档等权。

9600对数据请求及对应点随机种子逐项一致；来自118287张训练源图像中的9210张，未与验证源重合。19200次倍率采样中1852次≥64，最小1.00008、最大99.95691。数据控制证据见random_data_control.json。

| Gray | 200步子集主分数 | 400步子集主分数 | 600步子集主分数 | 600步完整主分数 | 完整clean H-AUC@5 | 裁剪比例 |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 2.316% | 0.456% | 0.754% | 0.2197% | 0.3465% | 14.33% |
| 10 | 1.848% | 0.764% | 1.176% | 0.2575% | 0.1365% | 13.00% |

子集分数不可替代完整主分数；第一阶段点数过多但位置信号弱，背景拟合后点数下降，loss下降没有带来同步质量改善。

完整协议分桶结果（正确匹配为每对均值；精度为逐对等权均值）：

| Bucket | gray3 H-AUC@5 | gray10 H-AUC@5 | gray3正确匹配 | gray10正确匹配 | gray3精度 | gray10精度 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.3465% | 0.1365% | 3.891 | 3.613 | 24.19% | 25.37% |
| 1 | 0.2622% | 0.0733% | 3.898 | 3.672 | 24.55% | 26.09% |
| 4 | 0.2609% | 0.3641% | 3.848 | 3.590 | 23.66% | 25.36% |
| 16 | 0.0699% | 0.4792% | 3.820 | 3.496 | 24.45% | 24.46% |
| 64 | 0.2854% | 0.0000% | 3.746 | 3.262 | 23.97% | 23.04% |
| 100 | 0.2201% | 0.3712% | 3.285 | 3.051 | 21.24% | 20.89% |

基线与上述模型均使用当前v2协议，基线主分数65.9515%。未使用旧v1学生1.39%或ratio256结果作比较。所有学生数字是有限试训结果，不是最终正式模型性能。

按256张源图像整体重采样10000次（同源各倍率一起采样），gray3−gray10主分数差−0.0378个百分点，95%区间[−0.2201,+0.1376]个百分点；正确匹配+0.3055个/对，95%区间[+0.0555,+0.5656]，但重复性和精度区间均包含0。仅源图像变异进入该区间，未覆盖训练种子、其他噪声实现或正式课程变异。少量正确匹配改善不能替代定位/主指标改善。

随机晚期20次探针的共享范数中位数(det/match/weightedgray)：gray3=.1373/1.4682/.01069；gray10=.1318/1.4472/.02447。S1中gray10加权灰度.01612与检测.01422相近，S2分别.01916/.03021；S3灰度为结构性零，不能仅凭全骨干范数说灰度在各层都不重要。两者检测/匹配负余弦比例S1/S2/S3均55%/30%/55%；随机S2中位数正向，S1/S3弱负向，固定样本冲突更强。观察到了竞争，尚未证明它是唯一或主要因果来源。

随机600步模型在同样4对图像上进行新阈值诊断（正式阈值保持.005）：gray3 clean正确匹配5.5→112→175（.005/.002/.001），gray10为5.5→113.75→177；ratio100分别3.75→74.75→103与5.5→75.5→101.5。top1024教师召回两者均约19–20%，仍有明显位置排序不足。各对角点误差、重复性、定位误差和筛选删除数保存在offline_random_g*_600/scan.json。

原检查点的补充复测完全重现clean 33.625点、2.17196%教师召回、教师位置分数全体中位数.00124318、已知位置双向检索93.1885%。新模型clean已知位置检索约84–85%，ratio100约62–63%，都绕过检测且仅有限候选；不能称为实际提取器匹配成功率。

## 5. 标定结果、原因分层与正式训练条件

**灰度权重未完成可用于正式定值的标定。** 原值10→保留暂存10，没有采用新的正式权重；10不是已证实合适的值，也没有确认最终权重。1在固定短试不占优；3在固定/随机部分正确匹配稍好，但完整主分数、重复性、定位和精度没有一致改善。不能把这些结果写成“降低灰度解决检测问题”，也不能以稳定性、loss下降或一次梯度宣布10已标定。

已确认的实现/记录问题：训练共享梯度日志缺项；噪声单元测试仍用旧256倍率；设计文档实施状态过时。分别补齐日志、改为100并通过针对检查、校正状态。未确认新的训练核心数学、坐标解码或几何warp错误。新诊断脚本自己的过严数值断言和配置记录问题另列，不混为项目训练错误。

已观察的质量问题：低响应、cell前景/背景判断不准、64位置概率扩散及相位偏好、有限正确匹配和错误几何估计。平均dustbin不能代表逐cell准确；响应不是严格平坦；降低阈值有明显收益，却仍不能恢复教师定位排序。固定样本有改善，尚未良好拟合；随机模型更弱。

本次检查不支持：旧检查点“因为曾训练ratio256而差”；固定方向的cell解码/warp偏移；检测参数没有梯度或优化器不更新；严重train/eval差异是旧检查点主要解释；仅分数阈值或仅降低灰度即可解决问题。这些排查覆盖当前测试条件，不宣称数学上排除所有图像/运行情况。

仍未知：匹配竞争对检测不足的具体因果份额；随机数据样本效率、优化时间和当前容量是否足以在完整课程学出位置响应；初始全[1,100]诊断分布的难度与正式先ratio1再渐增的差异；最合适的灰度权重及其全课程表现。没有新增未经确认的方法、损失或配置路线，也没有以局部假设修改正式实现。

**工程资产具备条件，经验参数/质量复核未具备可确认的正式启动条件。** 最终预检ready=true、28测试通过和全部资产一致；权重标定尚未成立、定位不足未闭环，所以本次不认定正式参数准备已完成。未启动100000步，也没有机械追加2000步：当前对照未形成新权重或清晰改善假设，不以“允许2000”为追加理由。单种子、固定8对、300/1000步、随机600步不能外推正式100k噪声课程、其他相机/真实Raw、最终SLAM或最终模型部署质量。

## 6. 实际命令、预算和可复核产物

在仓库根目录执行，GPU命令前缀为：

```bash
PY=/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python
export CUDA_VISIBLE_DEVICES=1
```

完整逐项命令见 [executed_commands.txt](../outputs/checks/design_calibration_20261001/executed_commands.txt)，随机试训另外设置OMP_NUM_THREADS=8、MKL_NUM_THREADS=8。主要实际入口：

```bash
"$PY" -m pytest -q tests
"$PY" -m rawfeat.cli preflight --config configs/formal.yaml
"$PY" -m scripts.calibrate_diagnostics scan --output outputs/checks/design_calibration_20261001/offline_scan_v2
"$PY" -m scripts.calibrate_diagnostics build-fixed --output outputs/checks/design_calibration_20261001/fixed_pairs
"$PY" -m scripts.calibrate_diagnostics fit --bucket 1 --gray 3 --steps 300 --output outputs/checks/design_calibration_20261001/fixed_ratio1_g3
"$PY" -m scripts.calibrate_diagnostics fit --bucket 1 --gray 3 --steps 1000 --resume outputs/checks/design_calibration_20261001/fixed_ratio1_g3/step_0300.pt --output outputs/checks/design_calibration_20261001/fixed_ratio1_g3_extended
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" -m rawfeat.cli train --config outputs/checks/design_calibration_20261001/random_fullrange_g3.yaml
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" -m rawfeat.cli train --config outputs/checks/design_calibration_20261001/random_fullrange_g10.yaml
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" -m rawfeat.cli validate --checkpoint outputs/checks/design_calibration_20261001/random_fullrange_g3/latest.pt --output outputs/checks/design_calibration_20261001/val_random_g3_600_full --max-images 256 --baseline-summary outputs/baseline/summary.json
"$PY" -m scripts.summarize_gray_comparison --left outputs/checks/design_calibration_20261001/val_random_g3_600_full --right outputs/checks/design_calibration_20261001/val_random_g10_600_full --output outputs/checks/design_calibration_20261001/gray_full_paired_comparison.json
```

| 运行 | 实际更新预算 | 输入/调度 | 保存结果 |
|---|---:|---|---|
| 旧检查点扫描/相位/KL/已知位置检索 | 0 | 固定4对当前v2输入 | offline_scan_v2、spatial_detail、historical_diagnostic_recheck |
| ratio100极值反向 | 0 | 两对，[1,100]/[100,100]，完整三损失 | extreme100.json |
| 固定clean gray10 | 300+700=1000 | 8对冻结样本，20预热 | fixed_clean_g10及extended |
| 固定ratio1 gray1 | 300 | 同初始模型/数据/20预热 | fixed_ratio1_g1 |
| 固定ratio1 gray3/10 | 每种300+700=1000 | 同模型/数据/20预热 | fixed_ratio1_g3/g10及extended |
| 随机全[1,100] gray3/10 | 每种600 | 同在线9600对请求/50预热/峰值保持 | random_fullrange_g3/g10，保留200/400/600 |
| 实际恢复 | 共4次 | 连续2与1+1恢复 | execution_checks |
| 成本profile | 1 | 完整4×4在线更新 | random_update_profile |
| 完整验证 | 每种1536对，0更新 | 当前v2，256源，固定.005 | val_random_g3/g10_600_full |

合计新增4505次优化器更新；最长单个实验总预算1000步，未重新执行历史2000步或正式100k。随机试训单次更新中位耗时约4.10/3.49秒，峰值分配约1.74GiB；耗时受并行只读诊断/验证影响，不能作权重效率差异结论。每次loss与梯度有限检查都实际执行，无训练failure.json。

固定试验实际参数以各目录actual_diagnostic_config.json为准：原config.json曾保留正式参考schedule，top-level诊断字段和实际日志使用20预热；保留原记录并添加校正记录，逐条学习率验证通过，代码已改为直接记录实际调度。未执行参考中的100k。首次扫描的描述子6e-8浮点差异触发过严bitwise断言，改为1e-6并在独立目录完成；实际恢复初始2e-5断言在2.1614e-5停止，检查全部状态后使用已完成状态验证输出≤7.38e-6，未重复训练。失败日志保留。两次辅助shell路径拼写错误已修正，未影响实验。

最终检查：`pytest -q tests`为28通过/4个第三方弃用警告，`pytest -q tests/test_sensor.py`为3通过，`compileall -q rawfeat scripts tests`与`bash -n scripts/run_required_checks.sh`通过；最后只读正式preflight仍为相同v2身份ready=true。历史完整检查点SHA和缓存/基线身份的末次保护核验见final_integrity.json。

![固定样本学习曲线](/homes/rongjie/projects/RawFeat/outputs/checks/design_calibration_20261001/fixed_learning.png)

![旧检查点代表空间响应](/homes/rongjie/projects/RawFeat/outputs/checks/design_calibration_20261001/spatial_detail/response_example.png)

图仅辅助查看；结论以逐对/逐cell统计为依据。
