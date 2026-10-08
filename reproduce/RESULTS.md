# Uni3R ScanNet：开发机 L20 完整评测

官方 2／8／16 视角权重均已完成发布测试加载器的全量评测，三项进程均正常退出。
下面是本机实际结果，不声称与论文分数完全一致。

| 输入视角 | 场景／样本 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | mIoU ↑ | Accuracy ↑ |
|---|---|---:|---:|---:|---:|---:|
| 2 | 40／320 | 25.370596 | 0.867149 | 0.141197 | 0.546484 | 0.819008 |
| 8 | 49／49 | 24.122183 | 0.829760 | 0.250930 | 0.521274 | 0.806720 |
| 16 | 49／49 | 23.069988 | 0.806486 | 0.298608 | 0.550141 | 0.834954 |

## 论文对比

参考：https://arxiv.org/html/2508.03643，固定视角表 1／2。

| 视角 | 指标 | 本机 | 论文 | 本机 − 论文 |
|---|---|---:|---:|---:|
| 2 | psnr | 25.370596 | 25.530000 | -0.159404 |
| 2 | ssim | 0.867149 | 0.872700 | -0.005551 |
| 2 | lpips | 0.141197 | 0.138000 | +0.003197 |
| 2 | miou | 0.546484 | 0.558400 | -0.011916 |
| 2 | accuracy | 0.819008 | 0.826800 | -0.007792 |
| 8 | psnr | 24.122183 | 24.710000 | -0.587817 |
| 8 | ssim | 0.829760 | 0.851000 | -0.021240 |
| 8 | miou | 0.521274 | 0.554000 | -0.032726 |
| 8 | accuracy | 0.806720 | 0.807000 | -0.000280 |
| 16 | psnr | 23.069988 | 23.320000 | -0.250012 |
| 16 | ssim | 0.806486 | 0.816000 | -0.009514 |
| 16 | miou | 0.550141 | 0.558000 | -0.007859 |
| 16 | accuracy | 0.834954 | 0.835000 | -0.000046 |

论文第 4.1 节写 40 个未见 ScanNet 场景；发布代码的 2 视角使用选定的 40 场景，
8／16 视角则读取 49 场景。本次完整保留发布加载器，因此多视角比较存在场景协议差异。
原安装说明使用 PyTorch 2.1.0，本次为 2.7.1；gsplat 在原说明中未固定版本。分数差异原因尚未被逐项归因。

深度指标保存在原始 JSON 中，来自目标视角；未与论文表 1 的源视角深度混比。
mIoU 沿用发布代码的 MeanIoU 聚合。推理时间包含原模型 distiller 前向，
运行在共享 L20 上，不与论文 A100 的推理速度作直接比较。

## 数据、环境与运行方法

所有资产和运行日志位于 `/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R`。ScanNet 压缩包及官方权重分片均通过 SHA256 校验，
合并权重校验清单位于 `checkpoints/{2,8,16}views.manifest.json`。
Python 3.10 venv 继承 fastw 的 torch 2.7.1+cu128；CUDA 12.9，gsplat 1.5.3。
三次正式运行的代码、包版本和 LSeg 补丁快照一致，见 `provenance_audit.json`。

配置：256×256、batch size 1、seed 777、AMP 关闭、100 步位姿对齐。
加速只跳过中间步骤不参与 RGB L1 优化目标的语义计算，最终渲染和所有指标保持完整。
与原始路径的同样本全部指标精确一致，见 `alignment_validation.json`。
额外尝试的 RGB-only 光栅化未用于正式结果。没有训练或微调权重。

```bash
cd /file_system/vepfs/intern/haozhe.jia/projects/FastWAM/3DGS/Uni3R
# 先用 nvidia-smi 选择空闲显存足够的开发机 L20；GPU 5 仅为示例。
UNI3R_GPU=5 UNI3R_VIEWS="2 8 16" UNI3R_FAST_ALIGNMENT=True bash reproduce/run_local.sh
```

详细环境准备与单视角命令见 `reproduce/README.md`。

## 原始结果

- 2 视角：`/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R/runs/local-20261005T190036Z/2views/metrics.json`
  SHA256：`09e3f262d745c9495d9604ef0f53ae6e34498f6e70b8a31bd1a73874b490b31e`；同目录 `eval.log` 包含完整日志。
- 8 视角：`/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R/runs/local-20261005T190318Z/8views/metrics.json`
  SHA256：`7e23d44895466f93dfbec0be6ffcd4b8b3b7a9e3e0fc53fec727b5dfc71ec850`；同目录 `eval.log` 包含完整日志。
- 16 视角：`/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R/runs/local-20261005T190709Z/16views/metrics.json`
  SHA256：`a7f3ebd5e86d098790a131b576f42c19b51218dddcdba32832a550fcb2d66da0`；同目录 `eval.log` 包含完整日志。
