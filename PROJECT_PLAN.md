# 项目规划：基于 Uni3R 的无位姿前馈 3D 实例分割（开放词汇）

更新：2026-10-15。本文档汇总目前已确定的方案、数据、协议、已有结果和下一步，作为团队共同参照。
旧的 `reproduce/INSTANCE_TEAM_PLAN.md`、`instance_A/`、`instance_B/` 是另一条路线（冻结 + GT 监督 query 解码器），
**不再作为方向依据**，只当工具库取用。

---

## 1. 目标与核心想法

- **任务**：输入 2 张无位姿图像，前馈输出 3D Gaussians，同时支持**实例分割 + 开放词汇语义**。
- **基座**：尽量沿用 Uni3R 官方权重（先只做 **2-view**）。
- **想法一：几何与语义/实例解耦**（受 UniForward 启发，但往前走一步）
- **想法二：教师蒸馏**（沿用 Uni3R 用 LSeg 蒸馏的思路，换成实例级、考虑多视角一致性的教师），训练一个学生分支接入 Uni3R。训练**不需要实例 GT**。

### 1.1 对 Uni3R 现状的理解（代码核实）

```
VGGT aggregator tokens ─┬─ point_head ───────────→ Gaussian 中心（几何，已独立）
                        ├─ camera_head ──────────→ 相机位姿
                        └─ dpt_head → gs_attr_proj(123 通道) = scale3+rot4+opacity1+SH48+gs_feats64+color3
gs_feats(64) += feature_reduction(LSeg(img))   ← 推理时也跑 LSeg，语义是 LSeg 上的残差
loss = L1 + LPIPS + 0.2·cos(渲染特征, LSeg(目标视角)) + VGGT 几何蒸馏
```

- 几何分支（point_head/camera_head）已经独立；外观和语义共享同一个 dpt_head，只在最后的 1×1 卷积分开。
- 推理依赖 LSeg（残差设计），不是"训完学生扔掉老师"。换成重型实例教师时**不能沿用残差设计**，学生必须独立输出。

### 1.2 UniForward 是怎么做的（读原文 arXiv 2506.09378 核实）

- MASt3R backbone → 几何分支（Gaussian 中心 + pose head）和属性分支（一个 DPT，外观 head 和语义 head 分开）。
- semantic head 的输入加了原图和 LSeg 特征（shortcut）。
- **全部端到端训练，没有 stop-grad**，语义梯度会流进共享 backbone。它的"解耦"只在 decoder/head 层面。
- 消融：完整 PSNR 26.147 / mIoU 0.347；w/o Sem. Head 25.455 / 0.317；w/o Shortcut 24.321 / 0.325。
- 结论：把 UniForward 原样搬到 Uni3R，实质上只剩"拆开 gs_attr_proj"，作为贡献不够。

### 1.3 我们的设计

1. **三路 decoder 解耦**：几何（point_head、camera_head）/ 外观（原 dpt_head）/ 语义+实例（新 DPT，用原 dpt_head 权重初始化）。
2. **梯度可控（核心消融，UniForward 没做）**：语义/实例损失流入共享 aggregator 的梯度乘 λ ∈ {0, 0.1, 1}，
   同时报几何和分割指标，回答"实例监督对几何是帮助还是干扰"。
3. **可选**：实例分支输入加入 detach 后的 point_head 3D 坐标。
4. **学生两个头**：
   - 语义头：CLIP 对齐特征（开放词汇）；
   - 实例头：对比学习嵌入，推理时聚类得到物体，mask 内平均语义特征与文本匹配。
5. **关键可行性点**：教师给的实例 ID 跨视角不一致没关系。实例头用"视角内对比损失"（对 ID 排列不敏感），
   损失在**渲染出的目标视角**上算，跨视角一致性由共享的 3D Gaussian 保证。多视角置信度用于损失加权/过滤。

### 1.4 训练阶段

| 阶段 | 几何 | 训练内容 | 目的 |
|---|---|---|---|
| 1 | 冻结 | 只训新的语义/实例分支 | 快速验证"教师 + 对比蒸馏 + 渲染监督"能学出实例；省算力 |
| 2 | 解冻（aggregator 后几层或 LoRA、dpt_head、gs_attr_proj，lr≈1e-5） | 联合训练，λ 消融 | 论证解耦的价值 |

**必须有的对照组**：原版 Uni3R 在**同样数据、同样步数**下继续训练（因为我们的训练数据不是作者那份）。

### 1.5 定位与竞品

- **SIU3R**（NeurIPS 2025）是最直接竞品：无位姿、前馈、实例/全景，用 GT 监督、闭集。
- 我们的差异化：**开放词汇 + 不需要实例 GT 的蒸馏 + 解耦与梯度分析**。
- 不推荐用 SIU3R 本身当教师（它用 GT 训练过，间接引入 GT）。

### 1.6 算力

8 × A100（共用）。代码按 1–8 卡可运行写，带梯度累积和断点续训；教师输出、distiller 输出离线缓存。

---

## 2. 数据

### 2.1 各论文的数据与"验证集"

| 论文 | 训练 | 报告结果用的 | 有无独立验证集 |
|---|---|---|---|
| Uni3R | ScanNet（排序后去掉最后 150 个）+ ScanNet++，论文称 1565 个场景 | LSM 测试集（LSM 论文构建，HF `Journey9ni/LSM/scannet_test.tar`），8 类语义 | 无（"最后 150 个"恰好包含 LSM 测试集） |
| SIU3R | ScanNet 官方训练集（1201 扫描） | ScanNet 官方 val（= SIU3R val） | 无（val 即测试集） |
| UniForward | ScanNet++ 官方训练集 | ScanNet++ 官方测试集 | 无 |

→ 我们**自己从训练集划验证集**，所有调参只看验证集。

### 2.2 最终划分（`data/splits/`，划分单位是物理场景 sceneXXXX）

| 用途 | 文件 | 规模 | 说明 |
|---|---|---:|---|
| 训练 | `train_scannet.txt` | 1241 扫描 | ScanNet 官方训练集 − LSM 测试集 9 个物理场景 − 验证集（1141），+ ScanNet 官方 test split 100 个扫描（0707–0806，无标注） |
| 训练 | `train_scannetpp.txt` | 219 场景 | 本地 ScanNet++ DSLR 子集（CUT3R 预处理） |
| **训练合计** | | **1460** | ≈ Uni3R 论文 1565 的 93% |
| 验证 | `val_scannet.txt` | 41 扫描 / 20 物理场景 | 官方训练集中 `random.Random(0)` 抽取；实例 GT 用 SIU3R train 对应扫描 |
| **测试（主）** | `test_siu3r_val.txt` | 312 扫描 / 1860 组视角对 | SIU3R val = ScanNet 官方 val，SIU3R 协议，所有指标都在这上面报 |
| 测试子集 | `test_siu3r_val_clean.txt` | 29 扫描 / 11 物理场景 | Uni3R 官方权重未训练过的场景（附录用） |
| 补充（暂缓） | `test_lsm_scans.txt` | 50 扫描 / 21 物理场景 | LSM 测试集，Uni3R 协议，只用于证明"改造后没变差" |

已脚本断言：训练、验证、各测试集之间无物理场景重叠。整个 ScanNet 官方 val 不参与训练。

**论文中需要说明的泄漏**：
- Uni3R 官方权重训练时见过 ScanNet 官方 val 大部分扫描（281/312，只有 RGB-D 自监督，无实例标签）；
- SIU3R 官方权重训练时见过 LSM 测试集中 9 个物理场景（0687/0688/0691/0692/0694/0698/0703/0705/0706）。

### 2.3 数据来源与格式

| 数据 | 来源 | 格式 |
|---|---|---|
| ScanNet 训练 | HF 公开镜像（`ZzZZCHS/processed_scannet`），**非作者版**：内参用 RGB 相机内参缩放补齐 | `color/*.png` 640×480、`depth/*.png` 16 位毫米、`pose/*.npz`（3×3 K + 4×4 c2w） |
| ScanNet++ DSLR | HF `vctvct321/scannetpp-cut3r-processed`，219 场景（另 8 个按 ≤128 帧规则排除） | `rgb_resized_undistorted/*.JPG`、`render_depth/*.png`、`camera/*.npz` |
| SIU3R val / train GT | HF `insomnia7/SIU3R`（公开），1000 train + 312 val | 256×256：`color`、`depth`（毫米）、`extrinsic`（c2w）、`intrinsic.txt`、`semantic`（20 类）、`instance`/`panoptic`（RGB 编码，`id=R+256G+65536B=1000·类别+实例号`，跨帧一致）、`iou.pt` |
| ScanNet++ iPhone 扩充（暂缓，不迁移） | HF `HarrisonPENG/scannetpp`（公开），902 场景，217 GB | `images` 920×690、`depth`、`refined_ins_ids`（跨帧一致实例 ID）、`scene_iphone_metadata.npz`（位姿/内参）；含 `splits/`。约 620 个场景不在官方 train/val 清单（v2 新增），可扩充训练；`nvs_sem_val` 50 个场景须剔除 |

作者版原始数据需向 ScanNet/ScanNet++ 官方申请，不是阻塞项。
ScanNet 已经没有可补的扫描；扩充只能靠 ScanNet++。

---

## 3. 评测协议（SIU3R 协议，主测试集）

- 直接调用 SIU3R 官方代码（`external/SIU3R` @ 8ea8016）：数据读取 `scannet_dataset.py`、文件输出 `visualizer.py`、打分 `evaluator.py`。
- 每组 2 个输入视角 + 6 个目标视角（含输入视角本身）。
- 指标：目标视角 PSNR/SSIM/LPIPS；深度 AbsRel/RMSE（逐图最小二乘尺度+平移）；
  输入视角（2D-only）与新视角（3D-aware）的 20 类语义 mIoU；实例 mAP、全景 PQ（全局 ID，跨视角换 ID 扣分）。
- stuff = wall/floor，thing = 其余 18 类。我们的方法用 20 个类名作文本查询实现开放词汇分类。
- 两个变体：`strict`（GT 位姿直接渲染，与 SIU3R 论文可比）/ `align100`（Uni3R 自带 100 步目标位姿优化，用到目标 GT 图，仅参考）。
- 代码：`scripts/siu3r_protocol/`（`export_uni3r.py`、`evaluate_siu3r.py`、`run_shard.sh`、`verify_scans.py`）。

---

## 4. 已有结果

### 4.1 分布检查：SIU3R 数据能否直接用于 Uni3R（`test_results/siu3r_check/REPORT.md`）

Uni3R 官方 2-view，Uni3R 自己的评测流程，31 个干净扫描 184 样本：

| 数据 | PSNR | SSIM | LPIPS | AbsRel |
|---|---:|---:|---:|---:|
| LSM 测试集（历史全量） | 25.37 | 0.867 | 0.141 | 0.0485 |
| SIU3R val 子集 | 23.54 | 0.818 | 0.183 | 0.0628 |
| 其中帧间隔 ≤20（与 LSM 相当） | 27.88 | — | 0.139 | — |

→ 可行；总分偏低来自约 1/4 的大基线（>45 帧）样本，不是分布问题。

### 4.2 Uni3R 基线（SIU3R 协议，1860 组全量，`test_results/siu3r_protocol/`）

| 方法 | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | mIoU（输入/新视角）↑ | mAP / PQ |
|---|---:|---:|---:|---:|---:|---:|---|
| **Uni3R strict** | **0.0625** | **0.181** | 22.09 | 0.703 | 0.247 | 0.361 / 0.361 | 无 |
| Uni3R align100（参考） | 0.0604 | 0.175 | 27.55 | 0.852 | 0.184 | 0.367 / 0.367 | 无 |
| SIU3R（论文） | 0.0742 | 0.208 | 25.96 | 0.822 | 0.184 | 0.592 / 0.592 | 0.282/0.661（输入），0.271/0.650（新视角） |
| LSM（SIU3R 重训） | 0.0747 | 0.219 | 21.88 | 0.734 | 0.304 | 0.275 / 0.271 | — |

要点：深度已优于 SIU3R（注意泄漏）；渲染 strict 低约 3.9 dB，主因是预测坐标系漂移；零样本语义 0.36 高于 LSM/LSeg 但低于
GT 监督的 SIU3R；desk/picture/shower curtain/bathtub/otherfurniture 接近 0；**实例为空，这是我们要补的**，目标是超过 SIU3R 的 0.28/0.66。

### 4.3 教师对比（验证集 41 扫描、205 组 / 410 帧，`test_results/teacher_bench/`）

| 教师 | 单帧 AP | 单帧 AP50 | AR100 | 双视角 AP | 双视角 AP50 | mCov | 跨视角 ID 一致率 | 类别 AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM2.1 自动分割 | 0.141 | 0.256 | 0.236 | 0.023 | 0.083 | 0.376 | 0 | — |
| **SAM2.1 视频传播** | 0.131 | 0.258 | 0.245 | **0.124** | **0.253** | 0.414 | **97.6%** | — |
| Grounded-SAM（阈值 0.3） | 0.189 | 0.397 | 0.267 | 0.021 | 0.086 | 0.401 | 0 | 0.342 |
| **Grounded-SAM（阈值 0.2）** | **0.210** | **0.449** | **0.307** | 0.024 | 0.100 | **0.475** | 0 | **0.366** |
| MaskClustering（SAM2.1 掩码 + GT 深度/位姿） | 0.116 | 0.233 | 0.222 | 0.083 | 0.179 | 0.339 | 97.2% | — |

结论：单帧质量和开放词汇 Grounded-SAM 最好；跨视角一致性 SAM2 视频传播最好；MaskClustering 受限于
用 SAM2 替代 CropFormer（零件级掩码）、256×256 帧和深度融合点云，这次不如 SAM2 视频。
单帧 AR 普遍 0.22–0.31，现成模型的"物体"粒度与 ScanNet 标注不完全一致，需要多视角融合/后处理。

---

## 5. 下一步

1. **教师组合**（各约 1 小时）：
   - Grounded-SAM 掩码作为 SAM2 视频传播的种子；
   - Grounded-SAM 掩码作为 MaskClustering 的 2D 输入。
2. **训练加载器改造**：`Scannet`/`Scannetpp` 读 `data/splits/` 清单（现在仍是"最后 150 个作验证集"的旧规则）。
3. ~~ScanNet++ iPhone 扩充~~：暂缓，正式训练确有需要再做（见第 6 节）。
4. **离线缓存**：教师掩码 + CLIP 特征、VGGT distiller 输出。
5. **阶段 1 代码**：三路 decoder、语义头 + 实例嵌入头、对比蒸馏损失、λ 梯度缩放接口；先冻结几何训练。
6. 阶段 2：解冻几何，λ 消融，同数据同步数的原版 Uni3R 对照。

---

## 6. 迁移到 daxiang-ying 开发机

连接：`ssh -p 2222 ying.wu@115.191.53.180`（公钥已加）。

| 项目 | 情况 |
|---|---|
| GPU | 8 × L20 46GB（共用，已占 14–38 GB/卡）→ 训练需调小 batch / 梯度检查点 |
| CPU / 内存 | 180 核 / 865 GB |
| 存储 | home 在 vepfs `/file_system/vepfs/intern/ying.wu`，共享盘剩 3.2 TB |
| **系统盘 `/` 已满（331 MB）** | `TMPDIR`、pip/uv 缓存、HF 缓存都必须指到 vepfs |
| 软件 | CUDA 12.9、git、rsync；无 conda/uv |
| **外网** | HuggingFace、hf-mirror、GitHub 不通；pytorch.org、阿里云 pip 镜像可用 → 权重和数据只能从本机传 |
| 机间带宽 | 约 25 MB/s（100 GB ≈ 70 分钟） |

迁移清单（目标目录 `~/3dgs/` 待确认；ScanNet++ iPhone 不迁移）：
- **带**：代码（Uni3R 仓库+子模块、`external/{SIU3R,MaskClustering,sam2}`、`scripts/`、`data/splits`、`test_results`）；
  权重（Uni3R 2-view、LSeg、CLIP、VGGT、SAM2.1-L、GroundingDINO-B）；数据（`scannet_processed`、`scannetpp_processed`、
  `siu3r_val_full`、`siu3r_trainval_gt` 压缩包）。
- **不带**：8/16-view 权重、原始压缩包与下载缓存、`scannet_processed_heldout`、`siu3r_instance`、LSM 测试集、
  作者数据核查材料、instance_A/B 旧产物、环境本身（到对方用脚本重装）。
- **ScanNet++ iPhone（216 GB）**：**本次不迁移**（2026-10-15 决定）。训练集已达论文规模 93%，属于锦上添花；且为 iPhone 数据，与现有 DSLR 风格不同。
  以后需要时在本机按需取子集（如 300 场景），预处理到 ~256 分辨率、去掉 `refined_ins_ids`、剔除 `nvs_sem_val` 后再传（估计全量 15–25 GB，300 场景 5–8 GB）。本机已有 18 个场景，暂不继续下载。
- 集群上 `haozhe.jia` 目录有早期 Uni3R 复现资产，若可读可复用部分权重/环境。

---

## 7. 环境与工程注意事项（当前 DSW 容器）

- 容器的 `LD_LIBRARY_PATH` 指向 Python 3.12 版 torch，所有环境导入 torch 前要剥掉：`source logs/siu3r_check/env_eval.sh`。
- 评测环境 `envs/uni3r_eval`（torch 2.1.0 + 已编译的 simple-knn/pytorch3d/curope + SIU3R 评测依赖）。
- 评测时可设 `UNI3R_EVAL_SKIP_DISTILLER=1` 去掉 VGGT distiller，省约 5 GB 显存（已验证输出一致）。
- **CPFS 大量小文件并发写会卡死**：解压/安装要限并发；不要用 `du` 统计大目录。
- **本地盘 `/opt` 会随容器重启清空**（已丢失一次教师环境与中间产物，分数已存回仓库）；只放临时文件。
- 后台下载/解压任务重启前要按进程树彻底清理，避免多个进程同时处理同一压缩包。
- 教师相关：CropFormer 官方权重已下架；MaskClustering 的 `pytorch3d.ball_query` 用 `scripts/teacher_bench/mc_shim/` 的
  torch 等价实现（已与暴力解核对一致，内存分块）；open3d 需要 `libusb-1.0.so.0`。
