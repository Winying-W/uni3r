# Uni3R 官方 2-view 权重在 SIU3R ScanNet val 协议下的基线

## 数据
- SIU3R 发布的 ScanNet val 全部 312 个扫描（= ScanNet 官方 val），`val_pair.json` 共 1860 组；
  `data/siu3r_val_full/`，经 `scripts/siu3r_protocol/verify_scans.py` 逐文件比对压缩包：312/312 通过。
- 每组 2 个输入视角 + 6 个目标视角（`target_ids`，含 2 个输入视角本身），与 SIU3R 一致。

## 协议（SIU3R 官方代码 `external/SIU3R` @ 8ea8016）
直接调用 SIU3R 自己的代码，不重新实现：
- 读数据与 GT 掩码：`src/data/components/scannet_dataset.py`（val、`seg_task=panoptic`）；
- 写结果与 GT 文件：`src/visualizer.py`（`visualize_recon_image`、`save_seg_ids`、`save_gt_seg_masks`）；
- 打分：`src/evaluator.py`，配置与 `configs/main.yaml` + `src/config.py` 的 panoptic 分支一致
  （20 类，stuff = wall/floor，thing = 其余 18 类）。
- 指标：目标视角 PSNR/SSIM/LPIPS；深度 AbsRel/RMSE（逐图最小二乘尺度+平移后）；
  输入视角（2D-only）和目标视角（3D-aware）的 20 类语义 mIoU；实例 mAP、全景 PQ。

## Uni3R 接入方式（`scripts/siu3r_protocol/export_uni3r.py`）
- 输入：SIU3R 的 256×256 图像和内参原样输入，不做 Uni3R 的裁剪，保证渲染与 GT 逐像素对齐。
- 目标相机：SIU3R 的 GT 相对位姿（相对第一个输入相机，公制），平移除以输入基线，换到 Uni3R 的坐标约定。
- 语义：渲染出的特征经 LSeg 与 20 个类名（`otherfurniture` 写作 "other furniture"）做开放词汇解码，
  每个像素都给出预测，不用 GT 掩掉未标注区域。
- 实例：Uni3R 没有实例输出，实例 ID 全为 0，mAP/PQ 不评（SIU3R 论文对 LSM 同样处理）。
- 评测时去掉 distiller（`UNI3R_EVAL_SKIP_DISTILLER=1`），它只服务于训练损失，已验证不影响输出。

## 两个变体
| 变体 | 目标位姿 | 说明 |
|---|---|---|
| `strict` | GT 位姿直接渲染 | 严格 SIU3R 协议；SIU3R 自己也不做测试时对齐。和 SIU3R 论文表直接可比 |
| `align100` | 每个目标视角用 Uni3R 官方的 100 步 RGB L1 位姿优化 | Uni3R 自己论文的评测方式；用到了目标视角 GT 图像，必须单独标注 |

为什么需要两个：诊断（`diag_context_render.py`）显示 Uni3R 预测几何与 GT 坐标系有漂移，
第二个输入视角不对齐时 15–23 dB，对齐后 20–27 dB。只报 strict 会低估 Uni3R 的渲染能力，
只报 align100 则对 SIU3R 不公平。
对齐时只渲染 RGB（特征通道换成 1 维占位），与带特征渲染的结果一致（3 组：PSNR 23.5195 vs 23.5174），速度约 2 倍。

## 结果（2026-10-13，1860 组 / 312 扫描，全部完成，0 跳过）

| 方法 | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入视角 mIoU↑ | 新视角 mIoU↑ | mAP / PQ |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Uni3R 2-view，strict（本次）** | **0.0625** | **0.181** | 22.09 | 0.703 | 0.247 | 0.361 | 0.361 | 无实例输出 |
| Uni3R 2-view，align100（本次，用到目标 GT 图像） | 0.0604 | 0.175 | 27.55 | 0.852 | 0.184 | 0.367 | 0.367 | 无实例输出 |
| SIU3R（论文表 1） | 0.0742 | 0.208 | 25.96 | 0.822 | 0.184 | 0.592 | 0.592 | 0.282/0.661（输入），0.271/0.650（新视角） |
| LSM（SIU3R 重训，论文表 1） | 0.0747 | 0.219 | 21.88 | 0.734 | 0.304 | 0.275 | 0.271 | — |
| LSeg（论文表 1，仅 2D） | — | — | — | — | — | 0.260 | — | — |

新视角逐类 IoU（strict，类别 1–20）：wall .62, floor .755, cabinet .494, bed .601, chair .541, sofa .473, table .455,
door .437, window .277, bookshelf .388, picture .016, counter .229, desk .0, curtain .331, refrigerator .525,
shower curtain .0, toilet .563, sink .506, bathtub .015, otherfurniture .002。

要点：
- 几何（深度）Uni3R 已优于 SIU3R；渲染（strict）与 LSM 相当、比 SIU3R 低约 3.9 dB，主要是预测坐标系漂移（align100 可补回）。
- 语义：Uni3R 的 LSeg 零样本开放词汇 0.36，远高于 LSM/LSeg（~0.27），但低于用 GT 语义监督的 SIU3R（0.59）。
  desk / shower curtain / picture / bathtub / otherfurniture 接近 0，是 LSeg 词汇与 ScanNet 标注定义不一致的典型类别。
- 实例 mAP/PQ：Uni3R 为空，这正是我们要补的部分。
- 注意 Uni3R 官方权重训练时见过大部分 val 扫描的 RGB-D（自监督），SIU3R 没有；深度/渲染的优势需谨慎解读，
  可在 `data/splits/test_siu3r_val_clean.txt`（29 扫描）上复查。

## 复现
```bash
bash scripts/siu3r_protocol/run_shard.sh {strict|align100} <shards> <index> <gpu>   # 可断点续跑
python scripts/siu3r_protocol/evaluate_siu3r.py --siu3r_repo ../external/SIU3R \
  --path test_results/siu3r_protocol/uni3r_2views_{strict|align100} --no_instance
```
环境：`source logs/siu3r_check/env_eval.sh`（本容器的 LD_LIBRARY_PATH 指向 py3.12 的 torch，需要剥掉）。
