# Winying-W/uni3r 合并记录

2026-10-08，将 https://github.com/Winying-W/uni3r 的 `main`
提交 `7a7f80205fb075228ff75aac43635b96de901110` 合入当前目录。
共同基线为 `4a5dd00b6737a95afd4c6b1760cad166cf17d3d4`。

## 合并内容

- 导入 `reproduce/` 下的实例诊断代码、研究和开发计划、另一台服务器的复现报告。
- 导入快速位姿对齐选项、检查点加载校验、内存映射加载和 PyTorch3D 延迟导入。
- 解决 `test.py`、`uni3r/loss.py`、`uni3r/lseg.py` 的冲突。
- 保留本机环境脚本、作者数据核查、分片评测和可合并指标累计量。
- 单数据集 `metrics.json` 同时保留远端使用的顶层指标及本地 `metrics.TestDataset`。
  `--max_batches 0` 仍表示完整评测；分片不能被结果比较脚本误报为完整测试集。

`instance_B/`、`uni3r/instance/` 及本地生成的报告、初始化权重、诊断结果全部保留。
备份清单中除主动合并的 5 个源码文件外，其余 **301 个文件逐项校验未变**。
四个子模块的原有工作区/暂存状态也保持一致，包括 rasterization 子模块已有的暂存删除。
数据链接仍为 `data -> ../data`，本机 checkpoints 和环境缓存继续使用现有目录。

## 验证

- 远端实例诊断数学测试：4 项通过。
- 本地实例模块测试：51 项中 50 项通过，1 项按原测试条件跳过。
- GPU 5 上完成一个真实 ScanNet 样本：官方 2-view 权重、100 步位姿对齐，
  开启远端 fast alignment，同时使用本地分片入口；正常退出。
- 检查两种 JSON 指标格式一致、累计状态中的样本索引为 `[0]`，并确认报告工具拒绝将 smoke 结果当作完整评测。
- 用合并后代码重新汇总已有 320 个样本的四个分片，结果与原全量指标 JSON 完全一致。
- 无未解决的 Git 冲突；Python 语法检查与 `git diff --check` 通过。

日志：`logs/merge_instance_tests.log`、`logs/merge_winying_gpu.log`、
`logs/merge_winying_shards_recheck.log`。
新增验证结果：`test_results/merge_winying_smoke/` 和
`test_results/merge_winying_shards_recheck.json`，原结果目录保留。

## 本机运行与 Git 状态

本机评测继续使用已配置的入口：

```bash
cd /mnt/workspace/Jensen/project/3DGS/Uni3R
CUDA_VISIBLE_DEVICES=5 bash scripts/test.sh --max_batches 1
```

`reproduce/run_local.sh`、`run_oracle_lifting.sh` 和历史报告保留另一台服务器的路径与环境信息。
若在此机使用其 Python 实例诊断入口，应先 `source scripts/env.sh`，并显式传入本机
`--checkpoint`、`--data`、`--output`；相应实例标注数据需要实际存在。
远端 Git 中未包含的原服务器权重、数据集和可视化文件不属于此次 Git 拉取内容。

本地 `main` 已前进到远端提交；本地修改和新增文件仍留在工作区，没有额外提交或推送。
新增远端名称为 `winying`，原 `origin` 保留。

合并前备份：
`/mnt/workspace/Jensen/project/3DGS/merge_backups/uni3r_before_winying_20261008T102957Z/`。
其中包含 `local_files.tar`、文件哈希清单、主仓库/子模块补丁与索引、保留校验结果和合并结果。
另保留本地 stash 对象 `391551a041387759efe8b6eea3bea99c0cfd1b5b`；该 stash 已应用，勿重复应用。

此前作者版训练数据的访问/版本问题仍独立存在，此次代码合并不表示训练数据目标已完成。
