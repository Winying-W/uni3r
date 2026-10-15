# 数据划分（2026-10-13 定稿，同日修订：验证集 40→20 个物理场景，训练集加入 ScanNet test split）

划分单位是**物理场景**（sceneXXXX）：同一房间的所有扫描（_00/_01/...）必须落在同一侧。

| 文件 | 用途 | 扫描 / 物理场景 | 来源与规则 |
|---|---|---:|---|
| `train_scannet.txt` | 训练 | 1241 | ScanNet 官方训练集 − LSM 测试集的 9 个物理场景 − 验证集（1141），加上 ScanNet 官方 test split 的 100 个扫描（0707–0806，无标注、不在任何评测集中，Uni3R 也未用其训练） |
| `train_scannetpp.txt` | 训练 | 219 场景 | 本机 `scannetpp_processed`（CUT3R 预处理子集）全部；官方划分未核对，不用于评测 |
| `val_scannet.txt` | 验证：调超参、阈值、选 checkpoint | 41 / 20 | 从官方训练集（去掉 LSM 测试场景后）用 `random.Random(0)` 打乱后取前 20 个物理场景；实例 GT 用 SIU3R train 的对应扫描 |

训练总规模 1241 + 219 = **1460**，约为 Uni3R 论文 1565 个场景的 93%。差距主要来自：我们不训练 ScanNet 官方 val（它是测试集），
而 Uni3R 的加载器训练过其中大部分扫描。
| `test_lsm_scans.txt` | 测试 A：重建 + 语义，Uni3R 协议，和 Uni3R 比 | 50 / 21 | LSM 发布的 `scannet_test`（2-view 实际用 40 个场景） |
| `test_siu3r_val.txt` | 测试 B：实例，SIU3R 协议，和 SIU3R 比 | 312 / 142 | SIU3R val = ScanNet 官方 val（已核对完全一致），`val_pair.json` 共 1860 组 |
| `test_siu3r_val_clean.txt` | 测试 B 的子集：Uni3R 官方权重未训练过的场景 | 29 / 11 | 官方 val 中，物理场景不在 Uni3R 加载器训练范围（排序后前 1463 个扫描）内 |

规则：
- 测试集只在最终报告时跑；所有调参只看 `val_scannet.txt`。
- 整个 ScanNet 官方 val 和 LSM 测试场景不进入训练和验证（脚本已断言无物理场景重叠）。
- 已知且需在论文中说明：Uni3R 官方权重训练时见过 ScanNet 官方 val 的大部分扫描（仅 RGB-D，自监督，无实例标签）；
  SIU3R 官方权重训练时见过 LSM 测试集中 9 个物理场景（0687/0688/0691/0692/0694/0698/0703/0705/0706）。
- `scannet_processed_heldout/` 中的 56 个扫描都属于测试场景，按本划分本来就要排除。

生成脚本逻辑记录在对话中；如需重建：官方清单在 `author_dataset_audit/scannet_official_{train,val}.txt`。
