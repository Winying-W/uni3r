# Uni3R 本机复现

> 2026-10-08 数据要求纠正：用户要求与作者一致的训练数据。
> 下文的 ScanNet 公开镜像和 ScanNet++ CUT3R 子集只完成了运行验证，
> **不满足作者版训练数据要求，数据复现尚未完成**。
> `scripts/train_local.sh` 已改为默认读取单独的 `data/author_processed/`，
> 缺少作者版数据时会退出；不再默认使用这两套替代训练数据。
> 原始来源核查、作者处理参数及尚缺信息见 [AUTHOR_DATASETS.md](AUTHOR_DATASETS.md)。
> 官方测试集和已经完成的官方权重评测不受此影响。

代码目录：`/mnt/workspace/Jensen/project/3DGS/Uni3R`。
数据目录：`/mnt/workspace/Jensen/project/3DGS/data`，仓库内 `data` 是其符号链接。
环境：`/root/miniconda3/envs/uni3r`，Python 3.10 / PyTorch 2.1.0+cu121 / NumPy 1.26.4。
本机为 A100；扩展使用系统 CUDA 12.8 编译，架构为 SM80。

## 已完成的评测与训练验证

2026-10-08，官方 2-view 权重、256×256、每样本 100 步位姿优化，
完成 320 个样本。四个分片的样本索引覆盖检查通过，指标按累计量合并。

| 指标 | 实测 |
|---|---:|
| PSNR | 25.3715 |
| SSIM | 0.86714 |
| LPIPS | 0.14120 |
| mIoU | 0.28365 |
| Accuracy | 0.81976 |
| Depth AbsRel | 0.04852 |
| Depth RMSE | 0.15369 |

这是当前代码和锁定依赖的实测结果，不声明与论文全部表格数值一致。
mIoU 使用仓库 `TestLoss` 和 torchmetrics 1.6.1 的逐图逐类 MeanIoU 口径，
不是全数据集累计混淆矩阵的 mIoU；不存在的类别按该版本默认规则处理。
深度 Delta 指标的阈值为源码中的 `1.03`、`1.03²`、`1.03³`。
JSON 中的 forward 时间不包含 100 步位姿优化，且测量期间 GPU 有其他任务。

训练验证使用原训练循环和 AdamW，对两套真实数据各更新一次：损失有限，
GS head 参数最大变化 `0.00015094`，峰值 CUDA 分配约 27.5 GB。
没有执行从头训练 100 epochs。

## 使用

```bash
cd /mnt/workspace/Jensen/project/3DGS/Uni3R
source scripts/env.sh

# 一个真实样本，保留官方 100 步位姿优化
CUDA_VISIBLE_DEVICES=5 bash scripts/test.sh --max_batches 1

# 320 个样本的完整 2-view 评测
CUDA_VISIBLE_DEVICES=5 bash scripts/test.sh

# 四张卡分别评测互不重叠的样本，合并累计统计量
GPU_IDS=0,1,4,5 bash scripts/evaluate_sharded.sh 2

# ScanNet 和 ScanNet++ 各一个训练批次，真实反向传播及 AdamW 更新
CUDA_VISIBLE_DEVICES=5 python scripts/train_smoke.py

# 完整训练入口；默认单卡、batch=1，可覆盖参数
CUDA_VISIBLE_DEVICES=5 bash scripts/train_local.sh
# 原论文脚本的 8 卡、每卡 batch=4 配置：
# NPROC_PER_NODE=8 BATCH_SIZE=4 bash scripts/train_local.sh
```

执行前选择有足够显存的 GPU。本次单样本评测峰值分配约 12.7 GB。
完整训练入口默认 100 epochs；训练 smoke test 不等于完成 100 epochs 训练。
本次复现以官方 **2-view checkpoint 的评测**为主；8/16-view 入口已支持本机路径，
相应权重需要另行使用下面的下载命令准备。

结果保存在 `test_results/`，日志在 `logs/`：

- `test_results/smoke_2views/metrics.json`：首次单样本验证，不能代表全测试集成绩。
- `test_results/2views_full/metrics.json`：四个完整分片结束后生成的全量评测。
- `test_results/train_smoke.json`：两套数据的训练验证。
- `requirements-reproduction.lock.txt`：本机实际安装版本快照。

## 数据来源与范围

所有数据下载到指定的 `data` 目录，保留源文件清单和哈希校验结果。

| 用途 | Hugging Face 来源 | 本机目录 |
|---|---|---|
| 作者指定的 ScanNet 评测集 | [Journey9ni/LSM](https://huggingface.co/datasets/Journey9ni/LSM) / `scannet_test.tar` | `data/scannet_test` |
| ScanNet RGB-D 训练镜像 | [ZzZZCHS/processed_scannet](https://huggingface.co/datasets/ZzZZCHS/processed_scannet) / `scannet_2d.tar.gz` | `data/scannet/archives`、`data/scannet_processed` |
| ScanNet 逐场景相机标定 | [maksimko123/scannet](https://huggingface.co/datasets/maksimko123/scannet) / ZIP 内 `intrinsic.txt` | `data/scannet/calibration` |
| ScanNet++ 预处理 RGB-D 子集 | [vctvct321/scannetpp-cut3r-processed](https://huggingface.co/datasets/vctvct321/scannetpp-cut3r-processed) | `data/scannetpp/archives`、`data/scannetpp_processed` |

ScanNet 共转换 1,613 个场景、132,418 帧，其中 1,492 个场景满足原始加载器
的 `>32` 帧条件。ScanNet++ 共 227 个场景包，转换保留 219 个场景、32,745 张 DSLR 图像。
8 个场景的 DSLR 帧数不超过 128，按 Uni3R 原始加载器的要求排除。
仅使用 DSLR 帧；保留原 RGB/深度像素、毫米深度单位和 camera-to-world 位姿。
相机数组从 `scene_metadata.npz` 转为逐帧 `camera/*.npz`。

ScanNet 镜像已将图像缩放到 640×480，但不包含内参。
脚本从另一公开镜像读取逐场景 RGB 内参及一张对应原图的真实尺寸，再缩放内参。
因此该训练数据使用 **RGB 相机内参**，区别于作者 README 的 `.sens` 导出后使用
`intrinsic_depth.txt` 的流程。此公开镜像的帧采样、场景范围和 ScanNet++ 子集均不应
宣称与论文完整训练集一致。完整评测使用作者指定测试集，不使用上述训练转换结果。

`data/hf_training_download_result.json` 记录下载是否全部成功；
`data/scannet_conversion.json`、`data/scannetpp_conversion.json` 记录实际场景和帧数。
`scene0533_00` 的缺失标定通过 [the-Lin/scannet_hf](https://huggingface.co/datasets/the-Lin/scannet_hf)
的分卷 ZIP 补齐，来源单独记录在该场景的 `supplement_source.json`。
原始加载器对 ScanNet 还会排除不超过 32 帧的场景，并将排序后的末 150 个场景用于验证。
整套数据实测：2-view 训练为 1,346 个有效场景、221,692 个样本，验证为
146 个有效场景、28,192 个样本；2/8/16-view 的实际样本读取均已通过。
`data/verification.json` 已通过逐场景三类文件集合一致性检查，以及每场景
首、中、末相机记录的尺寸、有限性、焦距和齐次矩阵检查。

重新准备数据：

```bash
source scripts/env.sh
python scripts/download_reproduction.py --views 2
# 若需要多视角官方权重：
# python scripts/download_reproduction.py --views 8 16
python scripts/download_hf_training_data.py
python scripts/fetch_scannet_intrinsics.py
python scripts/prepare_scannet_hf.py
python scripts/prepare_scannetpp_hf.py
```

测试 tar 的内部路径是 `data/scannet_test`，初次解压需在项目父目录执行：

```bash
python - <<'PY'
from pathlib import Path
import tarfile
data = Path('../data').resolve()
with tarfile.open(data / 'scannet_test.tar') as archive:
    archive.extractall(data.parent, filter='data')
PY
```

## 环境与修复

依赖清单在 `requirements-reproduction.txt`。本分支实际路径使用 VGGT 和 gsplat，
不依赖旧 PointTransformer 路径中的 flash-attn、spconv、torch-scatter 等包。
README 未列出的 `simple-knn` 已编译；PyTorch3D 使用 v0.7.7；CroCo CUDA 扩展已编译。
`submodules/diff-gaussian-rasterization` 在任务开始前已有暂存删除，本次保留；当前
渲染代码使用 `gsplat`，不需要该扩展。

重建同一环境可运行 `bash scripts/install_environment.sh`。精确源码版本为
Uni3R `4a5dd00b6737a95afd4c6b1760cad166cf17d3d4`，CLIP
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`，simple-knn
`60f461f4a56b7967e5d8045bf92f8c33f36976d0`，PyTorch3D
`89653419d0973396f3eff1a381ba09a07fffc2ed`。

源码修复：

- `test.py` 使用关键字传递 `device`，修正原来误传给 `use_pretrained_lseg` 的位置参数。
- 排除 CLIP / PyTorch3D 等安装源码目录的路径注入，避免遮蔽已编译 wheel。
- 从完整 LSeg checkpoint 严格加载全部 730 个权重时，跳过马上会被覆盖的 ImageNet
  ViT 预训练下载；`submodules/lang_seg/modules/models/lseg_net.py` 有对应小幅修改。
- 评测增加样本上限、无重叠分片、JSON 输出及可合并指标累计量；修复原计时列表被覆盖。
- 原 `scripts/test*.sh` 已改为使用本机数据路径和环境。

权重：Uni3R 和 VGGT 来自其官方 Hugging Face 仓库，按 LFS SHA-256 验证；
LSeg `demo_e200.ckpt` 来自 [Gofinge/LSeg](https://huggingface.co/Gofinge/LSeg)，
已通过其 LFS 哈希及模型严格加载验证。CLIP 官方权重已复制到 `checkpoints/clip` 并核对完整 SHA-256；
VGG16 从本机缓存复制并核对 `397923af` SHA-256 前缀。

初始依赖安装曾受默认代理和 PyPI 直连速度影响，最后通过阿里云 PyPI 镜像完成。
安装日志在 `logs/install_uv_aliyun.log`；编译日志在 `logs/build_extensions.log`
和 `logs/build_curope.log`。CPU/Python 静态检查可在受限环境执行；
GPU 运行和联网下载需要能访问宿主 GPU/网络的执行环境。
