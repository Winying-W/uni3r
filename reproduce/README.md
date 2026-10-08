# Uni3R：开发机 L20 复现

## 实例分割实验与开发规划

- [两/三人并行开发计划](INSTANCE_TEAM_PLAN.md)：冻结 Uni3R、缓存特征、独立训练实例学生及接回主线的接口和验收标准。
- [GT instance lifting 实验结果](ORACLE_INSTANCE_RESULTS.md)：已完成 3 场景、18 组双视角输入；这是标签与几何诊断，尚未训练实例学生。
- [方法与相关工作评审](INSTANCE_METHOD_REVIEW.md)。

GT 实验入口为 `prepare_oracle_data.py` 和 `run_oracle_lifting.sh`，数学检查可运行
`python reproduce/test_oracle_math.py`。脚本沿用下述本机环境路径；在其他机器运行前需配置对应的数据、权重和 Python 环境。
Git 仓库包含代码、配置及文字报告；模型权重、数据集、编译缓存和 `outputs/` 下的可视化/点云不随代码上传。

## 本机环境

数据、权重、Python 环境、日志和评测结果保存在：
`/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R`。
使用现有开发机 L20，不需要申请云端 GPU。

## 已有环境直接运行

```bash
cd /file_system/vepfs/intern/haozhe.jia/projects/FastWAM/3DGS/Uni3R
nvidia-smi
# 按实际空闲显存选择 GPU；下面的 5 只是示例。
UNI3R_GPU=5 UNI3R_VIEWS="2 8 16" UNI3R_FAST_ALIGNMENT=True bash reproduce/run_local.sh
```

各视角按顺序运行。单独运行可设置 `UNI3R_VIEWS=8`；添加
`UNI3R_MAX_BATCHES=1` 仅做冒烟测试，不能作为论文复现结果。
所有模型均已通过单样本完整推理、渲染、100 步位姿对齐和指标计算。

运行器保存 `runs/local-时间戳/`，其中包含 `packages.txt`、`code.patch`、
`lseg_vendor.patch`，以及每个视角的 `eval.log` 和 `metrics.json`。
`active_runs.json` 记录三项正式评测的结果路径和退出状态。
三套全量结果已完成，汇总见 `RESULTS.md`。

## 参数与指标

使用官方固定视角权重、256×256 分辨率、batch size 1、数据集 seed 777、
AMP 关闭、100 步位姿对齐。`fast_pose_alignment=True` 只跳过中间步骤
不参与 RGB L1 位姿优化目标的语义特征计算；最终指标保持完整。
同样本与原始路径的所有 RGB、语义和深度指标精确相同，证据见
`alignment_validation.json`。设为 `False` 可恢复中间语义计算。

2 视角为 40 场景、320 样本；8 和 16 视角的发布代码各读取 49 场景。
论文第 4.1 节描述 40 个未见场景，因此多视角结果只能作为发布代码的
实际复现，不能直接声称严格复现论文的同一场景协议。

`metrics.json` 的 `smoke_run=false` 且 `num_batches=total_batches` 才是
完整评测。PSNR、SSIM、LPIPS、mIoU 和准确率沿用原始代码聚合方式。
MeanIoU 在整个评测中没有出现的类别返回 -1；单样本 mIoU 可能为负。
深度指标来自目标视角，不能直接与论文表 1 的源视角深度比较。
`model_forward_seconds_mean` 包含原始模型的 distiller 前向，且在共享
L20 上测量，不能直接与论文 A100 推理速度比较。

```bash
python3 reproduce/compare_metrics.py /完整路径/2views/metrics.json \
  /完整路径/8views/metrics.json /完整路径/16views/metrics.json
```

## 环境与下载

Python 3.10 独立 venv 位于 EFS `venv/`，继承现有 fastw 环境的
PyTorch 2.7.1+cu128。额外依赖安装在 venv 内，未修改 fastw。
CUDA 12.9、L20 compute capability 8.9、gsplat 1.5.3；扩展构建缓存位于
本仓库 `reproduce/build_cache/`。每次运行保存实际包版本。

首次准备可运行 `bash reproduce/setup_local.sh`。当前数据和权重均已齐全，
无需重新下载。独立分片下载脚本为 `fetch_checkpoints.py`，合并脚本为
`assemble_checkpoint.py --views 2`（或 8、16）。不要同时运行多个资产下载器。
HF 直连超时的已验证替代路径是本机 HTTP 代理 37890 解析地址后直连 CDN；
脚本支持断点续传并校验 SHA256。权重校验清单位于 EFS `checkpoints/`。

为兼容当前 PyTorch，修正了模型加载 device 参数、可信权重的
`weights_only=False` 和训练机 LSeg 路径；LSeg 完整权重加载时不重复
下载会被覆盖的 timm 初始权重。训练专用 PyTorch3D 改为延迟导入。
未训练或微调模型，未降低分辨率、输入视角数或位姿对齐步数。

最新运行状态见 `STATUS.md`。历史记录 `HISTORY.md` 中的云端任务均已取消。
