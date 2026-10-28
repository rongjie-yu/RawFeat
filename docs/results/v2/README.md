# V2发布结果

V2发布结果固定为27000、28000、29000、30000步；权重及SHA-256见[weights/v2](../../../weights/v2/README.md)和[manifest.json](../../../weights/v2/manifest.json)。step0仅为V1初始化参考，其student和BN与发布V1/96000逐项精确一致。

| H-AUC@5 | V1初始化 | V2/27000 | V2/28000 | V2/29000 | V2/30000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HPatches原五噪声平均 | 48.608% | 47.645% | 47.661% | 47.565% | 47.117% |
| clean | 64.131% | 63.002% | 63.931% | 63.869% | 62.906% |
| ratio150 | 23.948% | 25.781% | 23.952% | 25.402% | 26.448% |
| ratio225 | 18.123% | 19.167% | 20.889% | 19.537% | 18.581% |
| 150/225平均 | 21.035% | 22.474% | 22.420% | 22.470% | 22.515% |

完整HPatches协议为116序列、696图像、每档580对。五档平均只含1/4/16/64/100，强噪声150/225另算。全部模型使用同一manifest、缓存及噪声种子；MeanAD+SuperPoint五档基线为46.8948%。最终30k相对V1的五档差为-1.492个百分点，序列配对95%区间[-2.381,-0.628]；强噪两档平均差+1.479个百分点，区间[+0.180,+2.778]，未做多重比较校正。结果支持噪声范围之间的取舍，尚未全面优于V1。

## 文件

| 路径 | 内容 |
| --- | --- |
| [training_summary.json](training_summary.json) | 30000成功更新、480000对、6:4实际配比和有限性检查 |
| [resolved_config.json](resolved_config.json) | 原训练实际配置 |
| [run_manifest.json](run_manifest.json) | 初次启动身份；恢复记录另存`resume_identity_*.json` |
| [source_provenance.json](source_provenance.json) | 数据索引哈希、图像/场景数量和留出场景 |
| [validation_manifest.json](validation_manifest.json) | 原固定验证456源、3648对的输入分配及种子；运行索引由`v2-prepare`重建 |
| [initial_validation.json](initial_validation.json) | V1初始化在V2固定协议上的结果 |
| `fixed_validation/` | 发布权重对应的固定验证/诊断记录；27k和29k没有完整几何验证，28k和30k为完整协议 |
| [hpatches_cache_identity.json](hpatches_cache_identity.json) | 公用HPatches缓存身份 |
| [hpatches/comparison.json](hpatches/comparison.json) | 四个V2结果与V1参考的汇总及配对区间 |
| `hpatches/step_*/legacy/` | 原六条件摘要、序列统计及原运行身份 |
| `hpatches/step_*/extra/` | 150/225摘要和序列统计 |

完整逐对JSONL、HTML、图像、逐图计时和缓存保留在原本地`outputs/checks/v2_hpatches_20261028/`。这里归档可复核指标和配对bootstrap的紧凑数据；原摘要保留实际运行路径和报告来源。

## 复核归档统计

以下只读取已发布的序列统计，无需重新运行GPU评估：

```bash
python scripts/compare_hpatches_checkpoints.py \
  --root docs/results/v2/hpatches \
  --output outputs/checks/v2_archived_comparison
```

完整重跑命令见[权重说明](../../../weights/v2/README.md#重跑hpatches)。原训练环境、数据根路径、文件元数据和资产身份属于历史记录；精确续训按严格身份合同执行，新环境的数据索引按训练手册重新生成。
