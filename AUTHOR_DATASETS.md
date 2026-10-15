# 与作者一致的训练数据：核查状态

核查日期：2026-10-08。Uni3R 源码：`4a5dd00b6737a95afd4c6b1760cad166cf17d3d4`。

**尚未完成。** 原先的公开训练镜像只能证明代码可以运行，不能视为作者训练集。
必须同时核实原始 release、场景清单、帧范围、预处理和训练划分；仅目录格式相同不够。
证据下载至 `../data/author_dataset_audit/`，其中不保存访问 token。

## 作者已公开的要求

依据：[数据文档](data_process/data.md)、[ScanNet 导出程序](data_process/scannet/export_data/export.py)、
[ScanNet 处理程序](data_process/scannet/scannet_processor.py)、
[ScanNet++ 渲染配置](data_process/scannetpp/common/configs/render.yml) 和
[渲染实现](data_process/scannetpp/common/render.py)。

| 项目 | 可核实的作者流程 |
|---|---|
| ScanNet release | 文档明确写 ScanNet **v2** |
| ScanNet 输入 | 原始 `scans/<scene>/<scene>.sens`，包含 RGB、深度、位姿、内参 |
| ScanNet 导出 | `OptimizedSensorData`，每 **10** 帧导出，480×640，PIL nearest RGB 缩放、JPEG quality=95；深度 nearest |
| ScanNet 后处理 | `intrinsic_depth.txt`；剔除非有限数据；作者 `BaseSceneProcessor` 输出 color/depth/pose |
| ScanNet loader 划分 | 场景排序后末 150 个用于 val，其余 train；各自过滤 ≤32 帧场景。并非直接读取官方 1201/312 split |
| ScanNet++ 输入 | 原始 DSLR `resized_images`、`resized_anon_masks`、COLMAP，以及 `scans/mesh_aligned_0.05.ply` |
| ScanNet++ 场景选择 | 原始 release 的 `splits/nvs_sem_train.txt` + `splits/nvs_sem_val.txt` |
| ScanNet++ 渲染 | 作者 renderpy 流程；fisheye 去畸变；默认 downscale_factor=3；near=0.05、far=20.0；毫米 uint16 深度；c2w 位姿 |
| ScanNet++ loader | 读取处理根目录内所有场景的 DSLR，排除 ≤128 帧场景 |
| 评测集 | 作者链接的 `Journey9ni/LSM/scannet_test.tar` 已下载并验证，可继续使用 |

作者[论文 v4](https://arxiv.org/html/2508.03643v4)报告两套数据合计 **1,565** 个训练场景，
但所核查的论文、数据文档和配置没有明确给出 ScanNet++ release 编号和完整训练场景清单。
不能仅凭引用 2023 年论文就断定是 v1，也不能把 v2 的最新完整数据直接称为作者版。
必须补齐 release/场景证据，并解释其与当前源码划分、过滤后的数量是否一致。
官方 ScanNet train/val 清单已留档作核对参考，不能擅自替换作者 loader 的划分。
进一步检查了官方 `2views.pth` 的 pickle 元数据（只解析，不执行反序列化）：
保存的 train/test dataset 参数与公开训练脚本一致，但未提供 release 和场景清单。
提取记录见 `../data/author_dataset_audit/checkpoint_metadata_strings.json`。

## Hugging Face 原始来源调查

以下只是候选原始镜像，**尚未验证内容与作者 release 相同**。
使用本机已有 HF 登录对实际文件发起 Range 请求，以下均返回 403；
详情见 `../data/author_dataset_audit/access_checks.json`。

| 候选镜像 | 元数据和访问情况 |
|---|---|
| [zhuhu00/scannet_sens](https://huggingface.co/datasets/zhuhu00/scannet_sens) | 28 个包，约 1.144 TB；manual gated；本机登录无文件访问权限 |
| [yujia0913/scannet_origin_compressed](https://huggingface.co/datasets/yujia0913/scannet_origin_compressed) | 分卷原始候选；manual gated；本机登录无文件访问权限 |
| [zhuhu00/scannetpp_compressed](https://huggingface.co/datasets/zhuhu00/scannetpp_compressed) | 41 个包，约 1.337 TB；auto gated；本机登录无文件访问权限 |

已检查的其他公开数据不足以直接复现：

- `ZzZZCHS/processed_scannet`：当前使用的抽帧 RGB-D 镜像；缺少原始标定，先前使用 RGB 内参补齐，与作者不同。
- `vctvct321/scannetpp-cut3r-processed`：227 个包的其他项目预处理子集；当前保留 219 个场景，与作者 release/处理流程不等价。
- `yifei-liu/scannetv2`：检查第一个批次的 ZIP 索引，仅有 color/pose/intrinsic，无 depth 或 `.sens`。
- `yifei-liu/scannetppv2`：检查第一个批次，主要为 DSLR resized_undistorted_images/COLMAP，无作者所需原始图像、实际 mask 和 mesh 的完整组合；且 v2 是否为作者版本未确认。
- `ziwang1105/yuxi-scannetpp`：ZIP 索引中为网格语义/手机元数据等，缺少 DSLR RGB。
- `haotongl/scannetpp_zipnerf`：所查样本为 iPhone/ZipNeRF 预处理结果。

403 是数据仓库访问控制，不是自动审批拒绝。需要已有授权的官方下载配置，
或对上述候选仓库已获准的 HF 账号，再检查包内容、版本及场景清单。
不要将 token 写入文档、命令日志或聊天；可提供本机配置路径。

## 准备目录和原作者处理命令

现有替代数据保留作 smoke test，避免覆盖或丢失已经完成的下载。
以下命令是待原始数据到位后的处理入口，**本次尚未执行，也不代表场景范围已确认**。

```text
data/author_raw/scannet/scans/<scene>/<scene>.sens
data/author_raw/scannetpp/data/<scene>/{dslr,scans}/...
data/author_raw/scannetpp/splits/{nvs_sem_train,nvs_sem_val}.txt
data/author_extracted/scannet/...
data/author_staging/{scannet,scannetpp}/...
data/author_processed/{scannet,scannetpp}/...  # 核验完成后才发布到这里
```

ScanNet：

```bash
cd /mnt/workspace/Jensen/project/3DGS/Uni3R
source scripts/env.sh
python data_process/scannet/export_data/export.py \
  --input_dir "$DATA_ROOT/author_raw/scannet" \
  --output_dir "$DATA_ROOT/author_extracted/scannet" --num_workers 4
# 作者并行处理实现实际需要 GPU；选择空闲 GPU，避免全卡同时使用。
CUDA_VISIBLE_DEVICES=5 python -m data_process.scannet.scannet_processor \
  --root_dir "$DATA_ROOT/author_extracted/scannet" \
  --save_dir "$DATA_ROOT/author_staging/scannet" --device cuda --num_workers 1
```

ScanNet++：本机路径配置已写入 `scripts/render_author_scannetpp.yml`，保持作者渲染参数。
独立 renderpy 环境已安装，并通过下述几何验证；原始数据仍未到位。

```bash
cd /mnt/workspace/Jensen/project/3DGS/Uni3R
source scripts/env_render.sh
cd /mnt/workspace/Jensen/project/3DGS/Uni3R/data_process/scannetpp
python -m common.render ../../scripts/render_author_scannetpp.yml
```

原始数据、release/场景证据齐备后，需要核验所有输入包、三类逐帧输出集合、
内参和深度单位、训练/验证场景 ID、有效样本总量，并用原 loader 读取真实样本。
核验前不把 staging 数据发布到 `author_processed`。

`scripts/train_local.sh` 默认读取 `author_processed`，目录未就绪时立即退出。
`scripts/train.sh` 也转到该入口，保留原始 8 卡、每卡 batch=4 的默认值。
`scripts/train_smoke.py` 仍是原来的替代数据运行验证；其成功不表示作者数据匹配。

## 已验证的原始渲染环境

环境位于 `.cache/renderpy-env`：Python 3.9.25、NumPy 1.26.4、
OpenCV Python 4.10.0，版本清单见 `requirements-renderpy.txt`。
使用作者指定的 [liu115/renderpy](https://github.com/liu115/renderpy)，
固定本次取得的源码为 `57a9308d8d92b26b9714d8c7fa6b6103971eccd6`，
源码及子模块未修改；作者文档没有固定其历史 commit。

系统包本来存在依赖冲突，因此没有执行系统安装或修复。
OpenCV C++ 4.10.0 构建在 `.cache/render-native`；GL/EGL 使用本机运行库，
编译头文件取自已缓存的 GLVND 1.7.0 和 GLU 9.0.3 包。
Boost 1.83 仅下载并解包作编译头文件，不修改系统包。
构建来源、哈希、依赖清单记录在 `test_results/author_renderer_environment.json`。
`bash scripts/build_renderpy_local.sh` 可利用本机已准备的原生依赖重建扩展。

```bash
source scripts/env_render.sh
python scripts/verify_author_renderer.py
```

验证实际调用未修改的 `common.render.process_scene`，使用两帧合成平面，
分别检查 2 m 和 1.75 m 的深度、c2w 位姿、缩放后的内参以及 RGB/mask/depth 输出。
输出为 2000 mm 和 1750 mm，逐像素最大深度误差均为 0 mm。
结果：`test_results/author_renderer_smoke.json`；日志：`logs/author_renderer_smoke.log`。
实际后端为 EGL 1.5、OpenGL 4.6.0 / NVIDIA 570.133.20，EGL 枚举设备 0。
作者 renderer 自动选择第一个能够初始化的 EGL 设备，不能假定其受
`CUDA_VISIBLE_DEVICES` 控制；完整预处理前需据此安排运行资源。

**这只是原始预处理程序的环境与几何验证，使用的不是 ScanNet++ 数据；
不能据此认定数据版本、场景范围或完整数据预处理已经完成。**
