# V2 HPatches 独立评估：2026-10-28

正式发布的V2结果为27000、28000、29000、30000四个checkpoint，见[权重与完整重跑命令](../weights/v2/README.md)和[结果归档](results/v2/README.md)。本文保留评估期间的历史快照与本地队列记录；step0仅为V1初始化参考，10000步和recovery不属于V2发布权重。

## 已保存的 checkpoint

在训练继续运行时，于北京时间2026-10-28 03:25左右复制并校验以下文件，固定路径为`outputs/checks/v2_hpatches_20261028/checkpoints/`。每个文件的源路径、SHA-256、大小、实际step和数据游标记录在同目录上一级的`checkpoint_inventory.json`中。

| 文件 | 实际step | 说明 |
| --- | ---: | --- |
| checkpoint_step_000000.pt | 0 | V1/96000初始化；student和BN与发布V1逐项精确一致 |
| checkpoint_step_010000.pt | 10000 | ratio课程达到225的边界权重 |
| checkpoint_step_027000.pt | 27000 | 近期训练权重 |
| checkpoint_step_028000.pt | 28000 | 近期训练权重 |
| checkpoint_step_029000.pt | 29000 | 本次捕获时的latest |
| recovery.pt | 1000 | 原故障恢复文件，保留用于审计；本次全量比较使用上述五个正式checkpoint |

原训练目录当时`latest.pt`指向29000，`best_geometry.pt`指向0。复制快照可避免训练自动清理27k等旧文件影响评估。30k在启动首轮评估时尚未生成；首轮五个权重完成后，另行复制30k最终权重并追加独立评估，见本文末尾。

## 协议和输入

- 全量HPatches：116个完整图像序列、696张图、580对图像，每个序列使用1→2…6全部五对，单图480×640、FP32推理。
- 沿用现有合成Raw评估：clean及ratio1/4/16/64/100，主分数仅平均五档噪声H-AUC@5；150/225扩展单独报告，不与旧主分数混合。
- 所有checkpoint读取相同新缓存、同一manifest、裁剪/几何及固定噪声种子。原有全量缓存的pipeline身份不匹配，因此没有绕过校验或改写原缓存；此次缓存为`outputs/checks/v2_hpatches_20261028/cache`。
- 使用现有`hpatches-evaluate`和`hpatches-v2-extra`入口，MeanAD+SuperPoint基线也保留。新增汇总对V2与V1做配对的序列bootstrap，10000次、seed20261028，分别报告光照/视角组和全体；其区间不混同于RawFeat相对SuperPoint的区间。
- 独立评估结果不修改训练中的优化器、配置或`best_geometry.pt`。共享GPU的计时只能反映本次环境，不作为独占GPU延迟结论。

## 已完成的短测

固定使用原有smoke序列`i_ajuntament/i_autannes/v_abstract/v_adam`，包含2个光照序列和2个视角序列，共20对图像。V1初始化及V2/29000均已完成clean、旧五噪声档与150/225评估；完整记录在`outputs/checks/v2_hpatches_20261028/smoke/`。

| H-AUC@5 | V1初始化 | V2/29000 |
| --- | ---: | ---: |
| 原五噪声主分数 | 35.823% | 36.026% |
| clean | 50.695% | 46.652% |
| ratio150 | 15.753% | 15.818% |
| ratio225 | 0.058% | 4.813% |

这是4序列小样本，不是完整HPatches结果。旧五档的V2−V1差值为+0.202个百分点，序列bootstrap 95%区间为[-6.741,+5.985]个百分点；150/225两档平均差为+2.410个百分点，区间为[-0.656,+5.171]。均未排除0，不能据此认定整体泛化提升。

## 全量队列

2026-10-28 03:31（北京时间）在GPU1启动独立后台进程。执行顺序：新缓存补齐→V1初始化→V2/29000→10000→27000→28000，每个checkpoint都跑原协议和150/225扩展。先完成V1/29000，便于尽早看到主要比较；其余权重随后串行处理。当前训练继续运行在GPU0。

运行信息为`outputs/checks/v2_hpatches_20261028/queue_process.json`，当前阶段与失败状态为`progress.json`，总日志为`queue.log`；每阶段还有独立日志。各checkpoint完整结果在`full/step_*/legacy`及`full/step_*/extra`，每完成一组，自动更新`full/comparison.json`和`full/comparison.md`。

已启动的准确命令：

```bash
cd /homes/rongjie/projects/RawFeat
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python -u \
  .planning/2026-10-17-v2-training-optimization-design/hpatches_20261028/run_full.py
```

不要在原进程仍运行时重复启动。队列会用文件锁阻止重复运行；已有完整、相同checkpoint哈希的结果可跳过，未完成输出目录不会被自动覆盖。完整五个权重预计需要约1–2小时，实际受共享GPU负载及磁盘影响，以阶段完成状态为准。

检查阶段及结果：

```bash
cat outputs/checks/v2_hpatches_20261028/progress.json
cat outputs/checks/v2_hpatches_20261028/full/comparison.md
```

最终需要`state: complete`，十个评估阶段都成功，并且每个legacy/extra的`summary.json`为`full_protocol: true`、`pair_count_per_condition: 580`。首轮五个权重已经全部达到这些要求，原完成状态保存在`progress_initial_five_complete.json`；`progress.json`随后用于追加的30k最终权重评估。

## 本轮已执行检查

`run_smoke.sh`通过`bash -n`并实际在CUDA运行完四个评估任务；两个Python汇总/队列脚本通过AST语法检查，汇总脚本已在真实smoke输出运行成功。全部六个权重已复制并核对SHA-256，实际step与文件名一致；step0 student/BN与V1/96000逐项相同。未修改`rawfeat/`训练或评估源码。

## 首轮全量结果：五个 checkpoint 已完成

首轮队列于北京时间2026-10-28 05:54全部成功结束，耗时约2小时23分。2026-10-28 14:46复核确认：十个legacy/extra评估均为full protocol，每档580对；各权重SHA-256与冻结清单相符，使用同一manifest和缓存身份，MeanAD+SuperPoint主分数均为46.8948%。每个权重的legacy文件有6960条唯一逐对记录，extra文件有2320条，H-AUC@5均为有限值。

| HPatches H-AUC@5 | 原五噪声平均 | clean | ratio150 | ratio225 | 150/225两档平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1初始化 / step0 | 48.608% | 64.131% | 23.948% | 18.123% | 21.035% |
| V2 / 10000 | 47.584% | 63.929% | 24.904% | 18.184% | 21.544% |
| V2 / 27000 | 47.645% | 63.002% | 25.781% | 19.167% | 22.474% |
| V2 / 28000 | 47.661% | 63.931% | 23.952% | 20.889% | 22.420% |
| V2 / 29000 | 47.565% | 63.869% | 25.402% | 19.537% | 22.470% |

旧五噪声档平均相对V1下降约0.95–1.04个百分点。后期三个V2权重的150/225两档平均增加约1.38–1.44个百分点；28k的225单档点估计增加2.766个百分点。当前体现的是强噪声收益与原档性能回退的取舍，尚不支持“所有噪声范围整体提升”的结论。这里没有用HPatches结果修改训练选模规则。

以116个序列为簇的配对bootstrap，10000次，差值和区间单位均为百分点：

| V2 step | 原五噪声平均差及95%区间 | 150/225两档平均差及95%区间 |
| --- | --- | --- |
| 10000 | -1.024 [-1.960, -0.105] | +0.509 [-0.829, +1.827] |
| 27000 | -0.963 [-1.892, -0.027] | +1.438 [+0.044, +2.840] |
| 28000 | -0.947 [-1.937, +0.014] | +1.385 [+0.093, +2.667] |
| 29000 | -1.044 [-2.018, -0.087] | +1.434 [-0.053, +2.866] |

这些是多checkpoint、分组比较中的探索性区间，未做多重比较校正；不据此宣称某个权重在整体泛化上确定最优。光照/视角分组和逐噪声结果保存在`full/comparison.json`及各模型原始报告。

## 训练结束与最终30000步补测

训练已于北京时间2026-10-28 06:28正常完成30000次成功更新，cursor480000。最终检查点为`outputs/formal_v2_30k/checkpoint_step_030000.pt`，模型权重有限，88组Adam计数均30000；`latest.pt`已指向该文件。

最终固定验证综合分数为62.240%，初始化为62.657%；COCO原五噪声为68.073%，初始化为68.085%；COCO clean为87.592%，初始化为87.947%。最终权重已满足旧域回退门槛，但综合分数仍低于初始化，因此`best_geometry.pt`保持step0。

首轮HPatches队列没有包含后来生成的30k权重，随后将其复制并核验后追加到相同缓存、相同完整协议的评估中。补测已于北京时间2026-10-28 15:11成功结束，最终30k结果如下。

```bash
cd /homes/rongjie/projects/RawFeat
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python -u \
  .planning/2026-10-17-v2-training-optimization-design/hpatches_20261028/run_full.py --steps 30000
```

补测状态写入当前`progress.json`、`queue_process.json`，总日志为`queue_final30000.log`；首轮完成状态及进程记录另存带`initial_five_complete`后缀的文件。最终30k结果已保存在`full/step_030000/`并加入`full/comparison.json/md`，当前状态为`complete`、退出码0。

## 最终30000步 HPatches 结果

2026-10-28 22:27复核通过：legacy和extra均为完整116序列、580对图像/档，分别含6960和2320条唯一且H-AUC@5有限的逐对记录；checkpoint SHA-256与冻结的30k权重一致，manifest与缓存身份和V1参考完全相同。完整五档基线仍为46.8948%。

| H-AUC@5 | V1初始化 | V2 / 30000 | 差值（百分点） |
| --- | ---: | ---: | ---: |
| 原五噪声平均 | 48.608% | 47.117% | -1.492 |
| clean | 64.131% | 62.906% | -1.225 |
| ratio150 | 23.948% | 26.448% | +2.501 |
| ratio225 | 18.123% | 18.581% | +0.458 |
| 150/225两档平均 | 21.035% | 22.515% | +1.479 |

以116个序列为簇的V2−V1配对bootstrap：原五噪声平均差为-1.492个百分点，95%区间[-2.381,-0.628]；150/225两档平均差为+1.479个百分点，95%区间[+0.180,+2.778]。区间未做多重比较校正，分组完整结果保存在`full/comparison.json`。

这次完整HPatches合成Raw评估显示：最终V2在150/225强噪声平均上有收益，主要来自ratio150；原五档平均和clean回退，尚未全面优于V1。ratio225点估计最高的已测checkpoint为28k（20.889%），最终30k未保持该单档水平；没有根据测试集事后修改训练选择的best或发布权重。

训练、首轮五个checkpoint和最终30k补测均已完成，没有剩余训练或评估进程需要等待。最终交付包括六个正式checkpoint的完整HPatches比较、逐对结果和配对序列置信区间。
