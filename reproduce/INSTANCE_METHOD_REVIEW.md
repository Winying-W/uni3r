# Uni3R → Feed-forward Instance 3DGS：六轮审查与建议方案

调研日期：2026-10-07。对应本地基线提交 `4a5dd00b6737a95afd4c6b1760cad166cf17d3d4` 及已有复现补丁。

## 结论

推荐先做 **冻结 Uni3R 重建路径 + 独立多视角实例解码器 + 跨视角一致的实例集合蒸馏**。先以 GT instance 验证学生，再以离线 teacher 伪标签训练单独的学生实验。部署只保留 Uni3R 重建路径和实例分支。

这是一条实现风险可控的研究路线，但尚不能承诺分数或“最优”。尤其 **object queries、pixel-aligned mask → Gaussian 实例分组并非空白：SIU3R 已经实现了非常接近的设计**。仅换 Uni3R 底座不足以建立论文贡献。需要验证：在保持重建不变的条件下，可靠的多视角伪标签蒸馏是否能替代人工实例标注，并在稀疏输入、几何误差和标签缺失下取得可测收益。

不把去除语义训练历史、从头训练 geometry-only、joint fine-tuning、unposed teacher 全部设成第一阶段要求。

## 六轮审查的结论与修正

| 审查轮次 | 核查结论 | 对方案的修正 |
|---|---|---|
| 1：UniForward 的“解耦”是什么 | 论文 §3.2 分为位置/位姿分支与外观+语义分支；§3.4 仍联合优化 photometric、pose、semantic losses | 学习其独立预测路径，不能援引它证明语义梯度完全隔离或冻结必然优于联合训练 |
| 2：本地 Uni3R 能否拆分 | Gaussian centers 来自 `vggt.point_head`；独立 `dpt_head` 与 123 通道头产生属性；LSeg 残差仅用于 `gs_feats` | 保留所有重建参数与路径，在共享 tokens 后新建独立实例解码器；先不裁原属性头通道 |
| 3：几何是否已足够可靠 | 现有全量评测含 100 步位姿对齐；深度另做逐图 scale+shift 拟合；不是原生几何质量的完整证明 | 先做无位姿优化基线、单一场景坐标对齐后的几何评估、GT-mask lifting 诊断 |
| 4：teacher 是否直接可用 | SAM2Object/MV3DIS 需要 RGB-D、相机、点云/mesh、superpoints；当前资产只有测试数据 | 先补正式训练集与实例验证标注；GT 几何用于离线 teacher 是合理主方案，预测几何 teacher 作为扩展 |
| 5：创新是否成立 | SIU3R 已有共享 queries、跨视角 masks、Gaussian lifting、mask rendering；GroupForward 已有分阶段训练与伪标签路径 | queries、冻结、teacher 均是合理构件，不能单独宣称首创；必须对比 SIU3R-style 与 embedding+clustering |
| 6：计算与评测是否闭合 | 16×256²≈105 万 Gaussians；全量 attention 不合适；semantic mIoU 不能证明实例能力 | 降采样 memory、分块 membership、class-agnostic 实例 AP、场景统一匹配、可见区域协议、独立计时 |

## 1. 文献查证

### UniForward：可以借鉴，但需要准确表述

读过论文正文及补充材料。其结构为 MASt3R backbone → 两条 DPT 分支：

- Geometry branch：Gaussian position + relative pose。
- Attribute branch：appearance head + semantic head。外观与语义共享该分支，再由不同 heads 预测。
- Semantic head 的 shortcut 包含原图及原图 semantic feature，不只是离线 loss 上使用语义。
- 总损失含 photometric、pose、semantic；论文没有把“整个共享 backbone 不接收语义梯度”作为设计保证。
- “w/o semantic head”消融的 PSNR 为 25.455，完整模型为 26.147；它不是严格的“去掉所有语义训练”的单因素证据，不能据此把语义一律视作几何污染。

论文入口：[UniForward](https://arxiv.org/html/2506.09378)。本次通过论文链接、GitHub 仓库及 README 搜索未定位到可确认的官方实现；不能声称已核对其代码。结构判断以论文为准。

### 最关键的近邻：SIU3R

[SIU3R，NeurIPS 2025 Spotlight](https://arxiv.org/html/2507.02705) §3.2 用跨视角共享 learnable queries 预测 masks；Algorithm 1 将 mask/query 结果映射到 pixel-aligned Gaussians；§3.3 又通过 Gaussian rasterization 聚合多视角 mask 信息。支持实例、语义、全景和文本指代任务。

实际核对了官方代码：

- [`src/models/model.py`](https://github.com/WU-CVGL/SIU3R/blob/main/src/models/model.py)：独立 adapter、VideoMask2Former、Gaussian heads；`post_process_gaussians` 将分割 mask 赋给 `gaussians.instance_labels`。
- [`src/models/mask2former/video_seg_decoder.py`](https://github.com/WU-CVGL/SIU3R/blob/main/src/models/mask2former/video_seg_decoder.py)：Hungarian matcher 的输入为 `[B,Q,V,H,W]`，跨帧展平后计算一个对象匹配，而不是每帧独立匹配。
- 官方提供 COCO 预训练 segmentation head；该 head 与 MASt3R/adapter 协同训练，不能不做适配就当成 VGGT 特征的即插即用权重。

我们的基础版与之高度接近。正确研究定位应是**冻结重建底座条件下的实例蒸馏及其鲁棒性/成本**，而不是“首次 object-centric Gaussian”。它也是必须纳入的直接基线。

### 其余直接相关工作

| 工作 | 已核查内容 | 对本项目的用途 |
|---|---|---|
| [GroupForward，2026-08 预印本](https://arxiv.org/html/2608.17535) | 紧凑 Gaussian instance embeddings、HDBSCAN、direct+render losses、三阶段 curriculum；也讨论 SAM2+几何生成一致伪标签 | 同底座 embedding+clustering 基线；冻结策略有依据，但不是新贡献。本次未找到可确认的官方仓库 |
| [EPS3D，ICML 2026（官方仓库标注）](https://arxiv.org/html/2606.08980) | 2D teacher 蒸馏 Gaussian semantic/instance features，semantic-instance mutual enhancement，实例聚类 | 证明 teacher→feed-forward perception 可行；官方代码有冻结重建属性 head 的实用版本 |
| [IGGT](https://arxiv.org/html/2510.22706) | VGGT-style tokens、geometry/instance heads、3D-consistent contrastive learning、聚类，InsScene-15K 数据 | 可用的数据/基线/离线 teacher 候选；说明联合训练也可能有益，不应绝对否定 |
| [SAM2Object，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_SAM2Object_Consolidating_View_Consistency_via_SAM2_for_Zero-Shot_3D_Instance_CVPR_2025_paper.html) | 本次读官方摘要、README 与 `graphclustering/sam2object_base.py`；SAM2 tracking、几何投影与合并，依赖 mesh/superpoints | 视频连续、几何完整时适合离线标签生成；本次未阅读全文 |
| [MV3DIS，CVPR 2026](https://arxiv.org/html/2604.08916) | 3D-guided mask matching、depth consistency、superpoint 合并；读正文、README 与 `mv3dis_main.py` | 推荐作为第一候选伪标签流水线，和 SAM2Object 在相同场景比较后选定 |

EPS3D 官方 [`eps3d_encoder.py`](https://github.com/Runsong123/EPS3D/blob/main/src/model/encoder/eps3d_encoder.py) 明确冻结 `gaussian_param_head`，并建立独立 semantic/instance heads；backbone 是否冻结另由配置决定。不能把“属性 head 冻结”误写成所有几何一定不会变化。

MV3DIS 当前公开仓库改用 Grounded-SAM2 并报告 ScanNetV2 class-agnostic AP/AP50/AP25=41.9/62.5/77.2。这些是 README 的完整几何、多视角 teacher 结果，不是本项目稀疏 student 的目标分数，也不应冒充论文原版或本地复现结果。若使用其检测词表，必须记录词表与类别先验。

## 2. 对 Uni3R 权重和计算图的核查

在 CPU 上以 mmap 读取了本地 `2views.pth`、`8views.pth`、`16views.pth` 的实际配置和 state dict，未训练、未改写权重。结构记录见 `INSTANCE_RESEARCH_SOURCES.json`。

权重根目录：`/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R/checkpoints/`。

| 模块前缀 | 职责 | state dict 元素量（约） | V0 处理 |
|---|---|---:|---|
| `vggt.aggregator` | DINO patch embedding、intrinsic embedding、frame/global attention | 908.17 M | 全冻结、eval |
| `vggt.camera_head` | 相机预测 | 216.17 M | 全冻结 |
| `vggt.point_head` | Gaussian centers 所用的 `world_points` | 32.65 M | 全冻结 |
| `vggt.depth_head` | depth/confidence | 32.65 M | 全冻结；用于诊断/分支几何信息 |
| `vggt.track_head` | tracking，可选路径 | 65.94 M | 不运行；部署可不加载 |
| `dpt_head` | Gaussian attribute 的 dense feature decoder | 32.95 M | 原件冻结；复制一份给实例分支 |
| `gs_attr_proj` | 256→256→123 channels | 0.62 M | 全冻结，第一版保留 123 通道 |
| `gaussian_head` | 属性拆分、activation、scale/covariance、SH | 0.008 M | 保留重建逻辑；其 `gaussian_proj` 在 VGGT 路径不用 |
| `lseg_feature_extractor` | LSeg teacher/semantic residual | 495.14 M | 新实例推理 wrapper 不加载、不运行 |
| `feature_reduction/expansion`、`tokenizer` | 原语义特征处理 | 很小 | 新实例路径不使用 |
| `distiller` | 518² frozen VGGT geometry teacher | 1,256.54 M（2-view ckpt 含该前缀） | 重建冻结后无需在线运行；8/16-view ckpt 未存该前缀，但当前构造器仍会加载它 |

关键事实：

1. `dpt_head` 不是 Gaussian centers 的唯一几何 decoder；centers 来自 `outputs['world_points']`。冻结“DPT”一项不足以冻结重建。
2. `gs_attr_proj` 的 123 通道按 `[3,4,1,48,64,3]` 拆分：scales、rotations、opacity、SH、semantic features、RGB residual。重建使用前 56 通道及末尾 3 通道；64 个 semantic channels 位于 `[56:120]`。不能简单把输出截成前 59 维。
3. `gaussian_head` 只把 LSeg residual 加到 `gs_feats`，不用于 centers/scales/rotation/opacity/SH。现有 `feat_residual=False` 配置并未阻止代码执行这一加法，必须改执行路径，不能只改配置。
4. `Uni3R.__init__` 写死 `self.config.freeze_dust3r=False`。新 wrapper 必须显式 `requires_grad_(False)`；`eval()` 本身不会冻结梯度。
5. `train()` 会递归影响子模块模式，需覆盖训练模式管理，保证冻结 base 始终 eval。
6. 复制 DPT 后务必重新设置新分支 `requires_grad_(True)`；深拷贝冻结模块会继承冻结标志。新分支可以接 detach tokens，但不能把实例 decoder 或 mask renderer 一起包进 no_grad。

保留 checkpoint 的语义训练历史是允许的。可声明“实例损失不更新重建参数”，不能声明“特征中没有语义”。从 VGGT 初始化重训无语义 Uni3R 是独立消融，不是实现本方案的前置条件。

## 3. 推荐学生架构

```text
RGB views + known intrinsics，no input extrinsics
                   │
       frozen vggt.aggregator
                   │
       tokens at layers 4/11/17/23
          ┌────────┴──────────────────────┐
          │                               │ detach
 frozen camera/point/depth heads     instance_dpt
 frozen dpt_head + gs_attr_proj      （复制原 dpt_head，独立训练）
          │                               │
      Gaussians                     multi-scale adapter
          │                       + detached XYZ/confidence
          │                               │
          │                      shared object queries
          │                      6-layer mask decoder
          │                               │
          └───────────── pixel/Gaussian membership + objectness
                                          │
                                instance IDs + confidence
```

### 特征与权重接入

- 从 `vggt_w_intr_emb.py` 已返回的 `outputs['tokens']['aggregated_tokens_list']` 接入。DPT 默认取第 `[4,11,17,23]` 个层输出。
- 每个 patch feature 为 2048 维（frame/global features 拼接），256×256 输入、patch size 16 时每视角 16×16 spatial tokens；排除 camera/register tokens。当前 forward 已移除附加 intrinsic token，不能误当作像素 token。
- 推荐复制本地 `dpt_head` 为 `instance_dpt`，让实例分支具有独立的 dense decoding capacity，避免仅在最终重建 feature 上加一个过小的 MLP。
- 新 decoder 最终输出 64 维 per-pixel mask features；query hidden dim=256，mask embedding dim=64，初始 query 数=100，6 层、8 heads。它们是起始超参数，不是论文最优值。
- query memory 使用 1/8、1/16 等降采样 dense feature；可把 1/8 memory 上限设为 8,192 tokens，保留 view/spatial 信息。不能让百万 Gaussians 全部进入多层 cross-attention。
- 3D positional encoding 使用 detached predicted centers，经场景中位中心与鲁棒半径归一化；保留 2D 位置、view/ray 信息作为补充。没有 GT XYZ 输入学生；置信度不是校准概率，先作为特征或固定过滤信号，不直接当作可靠性概率。
- 两视角最小测试使用 `2views.pth`；正式 8-view 实验使用 `8views.pth`，先单独建立对应训练和基线。两套 base 不同，实例 head 不应假定可无适配互换。
- 16-view 作为扩展。本地固定视角权重不等于已验证 arbitrary-view；官方 arbitrary-view 分支可后续单独复现。

### 实例预测

以像素 i 对应的 Gaussian 为单位，预测 `s_ik = sigmoid(z_i^T e_k / sqrt(d))`，以及 query objectness `o_k`。这里 `s_ik` 是独立 mask score，不能假称它已经是各对象间和为 1 的分类概率。

推理时先按 objectness 筛有效 queries，再以 `o_k*s_ik` 选择归属并应用在验证集确定的阈值；低置信结果进入 unknown/background。墙、地面等 stuff 不应强迫分到物体 query。scene 内实例 ID 不要求跨独立 forward 保持固定；跨输入子集比较必须允许 query permutation。

一颗 Gaussian 只对应一个输入像素，允许直接用源像素 teacher mask 监督它。这样不必把 GT mesh 的坐标强行 nearest-neighbor 到存在误差的预测点云。跨视角的一致 ID 来自 teacher 与场景级匹配。

训练临时计算 per-Gaussian features/membership 是必要的，因此不能声称“完全不在 Gaussian 上计算实例表示”。部署可仅保存每个 Gaussian 的 instance ID、置信度和 object query，不保存 N×Q 全矩阵，也不需要 HDBSCAN。

## 4. Teacher、标签与训练数据

### 首先补齐数据

当前已配置资产是约 1 GB 的 Uni3R 测试集，50 个目录，发布 loader 分别取 40/49 场景。没有在这些资产中发现可用的官方训练集或 instance-id maps。

`uni3r/datasets/testdata.py` 中 `instance` 是图片文件名；训练 loader 的该字段也是字符串。`labelmap` 是语义标签。它们都不能用作实例 GT。

新建独立 `InstanceDataset`，使用明确的 ScanNet train/val 场景名单，不复用当前 `Scannet._load_data()` 的“排序后最后 150 个当 val”规则。尽量按物理场景防止不同扫描序列泄漏；记录官方 base 的已知训练来源，不能保证未公开的 base 训练清单与新测试集绝无交集。现有 Uni3R 论文测试集不用于学生训练或阈值调参。

### 三条监督轨道

1. **Oracle sanity**：GT instance ID masks + 同样的 frozen Uni3R，验证学生和数据/匹配实现。该实验只能叫有监督验证。
2. **主 pseudo-label 轨道**：在训练场景用 RGB-D/pose/mesh/superpoints + MV3DIS 生成全局一致实例，再投影成 per-view maps。用 SAM2Object 做同场景质量和成本对照，选择更可靠者。teacher 可以离线使用更密集视角，学生只看采样的 2/8 views，但要公开这项 privileged training information。
3. **扩展轨道**：用 VGGT 预测几何重建点云和 superpoints，再运行匹配；或者评估 IGGT 输出作为另一种离线 teacher。需处理尺度、法向/邻域质量、遮挡与跟踪错误，不直接假设与 GT 几何同质。

GT depth/pose 用于离线训练监督并不违反“推理输入无外参”；无 GT-pose teacher 是更强的数据需求目标，用户当前任务不必同时解决。若主方法声称没有人工实例监督，主 pseudo-label 学生实验不得从已在 GT instances 上训练的学生权重继续训练；两条轨道分开初始化。

Teacher 缓存至少保存：`scene_id, frame_id, global_instance_id, mask, confidence, valid/ignore mask, visibility, camera/crop metadata`。仅把在学生输入中可观察的 teacher objects 当作匹配目标。未标注/遮挡/低可信像素标 ignore，不能全部当 background。

## 5. Loss：先直接集合蒸馏，再加入可信的渲染约束

### 第一阶段

一个场景的所有输入视角共用一次 Hungarian matching：

`cost(q,k) = 5*weighted_BCE(s_q,T_k) + 5*weighted_Dice(s_q,T_k) + 2*objectness_cost(q)`。

mask cost 在所有有效视角与采样像素上共同计算，按实例归一化，避免大物体和可见次数多的物体统治训练。完整 GT 时未匹配 queries 做 no-object；伪标签缺失较多时，降低 no-object 权重并屏蔽未知区域，不能把 teacher 漏检直接转成大量强负样本。

训练目标从 `L_direct + 2*L_objectness` 开始。Dice/BCE 的 5/5、objectness 的 2 为成熟 set-prediction 实现的起始值；归一化后应重新检查梯度大小。matching 与 loss 权重可共享配置，避免不一致。

### 第二阶段：membership rendering

把 membership score 当作 Gaussian 附加通道渲染：`R_vk(p) = sum_i T_vi(p) * alpha_vi(p) * s_ik`。

- 所有 centers/covariance/opacity/camera detach；只让梯度更新实例通道。
- 从已观察 source views 开始检查投影和 alpha coverage，再扩展到训练时 held-out target views。
- 只在 teacher label 有效且 frozen geometry 有可靠覆盖的位置监督；低覆盖不是物体不存在。若用 `R/alpha` 做前景归一化，必须限制最小 coverage，并明确这是条件于已覆盖表面的 mask，不是完整画面概率。
- target camera 必须和 Gaussian 处于同一 gauge。训练时可由 context GT pose/depth 做一个场景级变换；不能给每个 target 单独优化位姿掩盖几何错误。部署不提供 GT geometry。
- 起始 `L = L_direct + 2*L_obj + 0.5*L_render`，先确认可见性/坐标正确再启用，0.5 是待调起点。
- 不假设“共享 query”自动保证几何一致。source masks 对了但 novel-view masks 错，可能是 geometry、occlusion、重复 Gaussians 或 assignment 的问题，需用 oracle lifting 诊断。

可研究但不应先宣称新颖性的扩展是 **coverage-aware scene-set distillation**：同一训练场景采样两个有重叠的输入子集，以 teacher 全局 ID 对齐各自匹配的 query，只在共同可见、可信表面约束 membership；允许一个子集中不存在的物体被忽略，避免要求固定 query 编号。其是否超越普通 multi-view set supervision 必须做消融，并进一步查重。

## 6. 冻结、解冻与训练预算

### 默认版本：不解冻重建

- 冻结表中所有原有 reconstruction modules，base 始终 eval/no_grad。
- 训练 `instance_dpt`、adapter、XYZ encoder、query decoder、mask/objectness heads。
- 初始 AdamW：新模块 LR=1e-4，复制的 `instance_dpt` LR=1e-5，weight decay=0.05，gradient clip=1.0；500-step warmup 后 cosine。batch size 1，以梯度累积到 4 个场景。均为试验起点。
- 初始 256²，2-view 功能验证；8-view 主实验。先单场景/少场景过拟合，再 50 个训练场景与独立验证小集做 10k-step 先导，确认有效再扩展数据和训练步数。
- 约 33 M 的独立 DPT 加轻量 decoder 比复制整个 0.9B aggregator 更合适。预期适合单张 48GB 级 GPU 的冻结训练；8/16-view 峰值显存和吞吐尚未实测，不保证当前共享 L20 的空闲显存足够。
- 不保存所有训练样本的全分辨率 256-channel features，磁盘开销大。可缓存固定样本的四层 patch tokens 用于小规模调试；正式多视角采样在线跑冻结 base。

### 如果冻结版本欠拟合

先依次查 teacher recall/ID consistency、boundary resolution、query 数上限、sample diversity、instance decoder capacity；不要把所有问题归因于 backbone 不可训练。

如确需更强 adaptation，可在冻结 tokens 后增加 2–4 层**仅属于实例分支**的 cross-view adapter blocks。它们与重建分支不共享可训练参数，因此仍保留重建不变的保证。

只有作为可选上界实验，才开放原 `vggt.aggregator.frame_blocks/global_blocks` 最后 2 组，LR=1e-6，其他 patch embedding、前面 blocks、camera/point/depth heads、原 `dpt_head/gs_attr_proj` 继续冻结。注意：即使这些 heads 冻结，输入 tokens 变化仍会改变几何，所以必须同时保持 RGB/geometry 约束，并监控与 frozen reference 的 outputs。该版本不再声称实例训练完全不影响重建。

不默认 joint fine-tuning；不默认从头 geometry-only。只有在对应消融证明值得投入时再开展。

## 7. 几何能力：目前知道什么，还不知道什么

现有发布协议全量结果：

| Views | PSNR | SSIM | LPIPS | 对齐后的目标视角 AbsRel |
|---|---:|---:|---:|---:|
| 2 | 25.3706 | 0.8671 | 0.1412 | 0.04852 |
| 8 | 24.1222 | 0.8298 | 0.2509 | 0.04461 |
| 16 | 23.0700 | 0.8065 | 0.2986 | 0.04809 |

这些证明官方模型重建链路可运行，不足以单独证明适合准确 3D 实例分割：

- 100 步 pose alignment 是测试渲染相机优化，不是训练 Gaussian，但完整计时/评分链路仍包含测试优化。
- `DepthEstimationMetric` 启用逐图 least-squares scale+shift，含 20% trimming；`Delta1` 阈值实际为 **1.03**，不是常见的 **1.25**。
- 8/16-view 是 49 场景，2-view 是 40；不能把不同 view 的 PSNR 变化直接解释为多视角增加导致几何下降。
- PSNR 无法直接衡量表面分层、重复 Gaussians、物体边界位置误差和细物体缺失。

优先做两个诊断：

1. `pose_align_steps=0`，不做测试 RGB 梯度优化；检查预测 camera、world_points 与原渲染坐标体系。以同一场景变换评价 raw/scene-aligned geometry，单独保留论文协议结果。
2. 把 GT instance masks 通过 source-pixel 索引直接赋给预测 Gaussians，再渲染/映射到评价表面。该 oracle-label lifting 是“标签足够准时几何会怎样”的诊断参考，并非严格数学上界。如果仍严重破碎，先处理几何或任务范围，不能指望换 decoder 完全修复。

## 8. 指标、基线与继续投入门槛

### 指标与协议

- **主要任务**：class-agnostic object instance segmentation；先不增加语义类别/文本 grounding。墙、地面归 stuff/背景，不强制变成物体。
- **主指标**：3D class-agnostic AP@[0.50:0.05:0.95]、AP50、AP25。采用明确的 ScanNet thing-instance 定义、最小实例尺寸、忽略标签规则。
- **稀疏观察**：主要报告由输入可见性定义的 observed-surface/visible-instance 评价，所有方法用相同 GT 定义的 mask；完整 scene benchmark 另报，不能把不可见区域删除后的 AP 当官方 full-scene AP。
- **2D/跨视角**：context 与 novel-view mask AP 分开；场景级一次匹配后的 multi-view IoU、split/merge rate、boundary F-score。不能逐视角重新匹配来掩盖 ID switching。
- **几何**：depth AbsRel/RMSE，明确 1.03 和 1.25 两种 threshold；单一 Sim(3)/尺度对齐下的 point accuracy/completeness/F-score。若使用 GT 对齐，只用于评价坐标变换，不纳入模型能力。
- **重建保持**：对应 frozen base 的 Gaussian 参数差、PSNR/SSIM/LPIPS 差。同 dtype、同路径的参数应一致；更改 kernel/channel layout 后可能有微小数值差，不可轻率承诺 bitwise RGB 相同。
- **效率**：RGB→Gaussians+instances 总耗时，含所有必需推理计算；另报 renderer 时间、峰值显存、teacher 离线耗时。不使用含 distiller 的旧时间冒充优化后速度，也不隐藏 HDBSCAN baseline 的聚类耗时。
- 未预测类别时不报告完整 semantic panoptic PQ；semantic mIoU 不是实例指标。

### 必须有的基线矩阵

| 实验 | 冻结的 Uni3R | 监督 | 解码方式 | 回答问题 |
|---|---|---|---|---|
| B0 | 同一权重 | GT labels 直接 lifting，无训练 | 源像素→Gaussian | 几何诊断参考 |
| B1 | 同一权重 | GT instance | 8D embedding + HDBSCAN | 同底座实例特征基线 |
| B2 | 同一权重 | GT instance | SIU3R-style shared queries | query 方案本身是否有效 |
| B3 | 同一权重 | 同一份 teacher labels | embedding + HDBSCAN | teacher 固定后比较表示 |
| B4 | 同一权重 | 同一份 teacher labels | queries，direct masks | 蒸馏学生基础版 |
| B5 | 同一权重 | 同上 | B4 + visibility-aware render loss | 几何约束收益 |
| B6 | 同一权重 | 同上 | B5 + 可选子集一致性 | 所提研究点是否有独立收益 |

另列 SIU3R 官方模型、IGGT/EPS3D 等外部模型作为系统级比较，但不能把不同数据、输入、监督、分辨率的分数直接当控制变量实验。UniForward 是结构参考而非现成的实例 baseline。

### 实用门槛（建议值，非已有结果）

1. 裁去 teacher 路径后检查 Gaussian 参数相等；纯冻结训练后仍满足。若不满足，先修模式/参数共享/坐标代码。
2. 在简单的单场景 GT sanity 上，匹配后 visible mask Dice 能到约 0.90，并且同一对象跨视角 ID 稳定。达不到时先排查工程，不急着大规模训练。
3. 小规模未见场景上同时记录 oracle-label lifting 和 teacher labels 质量；无可靠标签覆盖时先改 teacher/data，不堆 loss。
4. query 版本在同监督、同 base 上至少表现出稳定的准确率或效率收益；可把 AP50 提高 2–3 个百分点作为投入下一阶段的内部目标，必须用多 seed/场景 bootstrap 排除噪声，不能作为保证。
5. B5/B6 无稳定增益则删掉，不把未经验证的复杂性写成方法贡献。

## 9. 实现落点与阶段产物

推荐新增独立 wrapper 和训练入口，保留已有复现脚本可回归：

```text
uni3r/instance/frozen_base.py       # 仅加载必要 checkpoint prefixes，输出 Gaussians/tokens
uni3r/instance/decoder.py           # instance_dpt、adapter、shared queries、membership
uni3r/instance/criterion.py         # 场景级 matching、ignore/confidence、render loss
uni3r/instance/teacher_cache.py     # 离线实例标签、坐标和裁剪信息读取
uni3r/datasets/instance_dataset.py  # 官方 split 与跨视角全局 ID
train_instance.py
eval_instance.py
```

这是建议的文件布局，**本次未实现这些模型文件，也未运行新训练**。

阶段顺序：

1. 准备训练/验证实例数据；完成无测试位姿优化的 geometry sanity 与 teacher-free wrapper 一致性。
2. 两视角、小规模 GT instance 过拟合与未见场景验证；B1/B2 对照。
3. 选择 offline teacher；伪标签学生独立初始化；B3/B4。
4. 八视角主实验与 B5；检查 sparse-to-dense、mask coverage、teacher noise。
5. 有证据后做 B6、partial adaptation、arbitrary-view 或 predicted-geometry teacher；不把所有扩展同时开启。

实现可行性：基础 wrapper/branch 高；可靠 teacher 数据流水线中等；稀疏 noisy geometry 下高质量 3D segmentation 中等且需要先验诊断；无需人工 pose 的完整 teacher 工程成本更高；仅靠 queries+冻结形成论文创新的把握低。

## 10. 证据范围

本次实际阅读 UniForward、Uni3R、SIU3R、GroupForward、IGGT、EPS3D、MV3DIS 论文内容；核对本地 Uni3R 源码/实际 checkpoint metadata，及 SIU3R/EPS3D/MV3DIS/SAM2Object 的相关实现。SAM2Object 论文仅核对官方摘要，方法细节主要依据代码与 README。UniForward 未定位官方代码，普通网页搜索部分结果不相关，已排除；因此不声称完成穷尽式最新文献/新颖性审查。

链接、下载内容 hash、checkpoint 结构见 [`INSTANCE_RESEARCH_SOURCES.json`](INSTANCE_RESEARCH_SOURCES.json)。既有实际评测见 [`RESULTS.md`](RESULTS.md)。本次仅新增调研文档，没有改训练/推理代码、没有启动 GPU 训练或重新评测。
