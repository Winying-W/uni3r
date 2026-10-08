# 实例分支独立训练与两/三人并行开发计划

日期：2026-10-08。第一阶段仅使用官方 `2views.pth`，不同时训练 2/4/8-view 模型，不更新 Uni3R 重建权重。

## 当前结论与实验依据

可以两到三个人并行开发。推荐路线是：冻结主线 → 导出真实特征缓存 → 独立训练实例学生 → 接回在线推理。学生训练和最终部署必须使用同一份 Uni3R 权重及相同预处理；接回时只替换特征获取方式，不需要重新训练整个模型。

GT lifting 实验已经完成：3 个场景、18 组双视角输入，共 72 次新视角渲染和 36 次源视角互渲染。原两视角坐标约定下：

| 诊断 | 实例平均 IoU |
|---|---:|
| 两个输入视角的 Gaussians 渲染到新视角 | 84.97% |
| 仅用另一张输入图的 Gaussians 互渲染 | 42.10% |
| 互渲染限制到 GT 判断可覆盖的区域 | 59.57% |

这些结果使用 GT 实例标签及评测相机，不是学生模型的分割成绩，也不是完全无位姿部署结果。它们支持继续验证学生可学习性，但覆盖、遮挡和几何误差仍然明显，因此第一版先监督输入视角实例，不施加强跨视角渲染约束。完整协议、三维指标及局限见 [GT 实验报告](ORACLE_INSTANCE_RESULTS.md)。

沿用最终 Uni3R 权重不等于清除其语义训练历史。第一阶段能保证的是实例损失不更新原重建参数；无需先重训 geometry-only base。

## 1. 能否单独训练学生，再接回主线？

可以。这里的“单独”指独立优化实例分支；它的训练输入仍来自固定的 Uni3R 特征。

```text
离线训练：
RGB pair → frozen Uni3R → 特征/几何缓存 → instance student → mask losses
                       不反传              只更新这里

接回推理：
RGB pair → frozen Uni3R ┬→ Gaussian reconstruction
                       └→ 相同特征 → 已训练 instance student → Gaussian IDs
```

学生不是先在 SAM 特征或任意 RGB 分割特征上训练、然后直接嫁接。缓存输入与在线输入的 checkpoint、层号、分辨率、裁剪、归一化、顺序必须一致。网络形状相同不代表特征分布相同。

复制的 `instance_dpt` 是学生的一部分，可以独立训练；原 `dpt_head` 留在冻结重建路径中。第一版不需要训练一个全新的 Uni3R。

## 2. 三人分工（推荐）

| 人员 | 负责的模块 | 主要产物 | 不应同时改动的部分 |
|---|---|---|---|
| A：重建与接入 | 原模型基线、GT lifting 诊断、frozen base wrapper、特征导出、最终组合推理 | `frozen_base.py`、`export_features.py`、reference fixtures、`infer_instance.py` | B 的 decoder/loss 与 C 的标签定义 |
| B：学生训练 | instance DPT、adapter、query decoder、Hungarian loss、optimizer、学生 checkpoint | `student.py`、`criterion.py`、`train_instance.py`、`student.pth` | 原 Uni3R encoder/geometry 参数和 C 的场景划分 |
| C：数据与评价 | 正式 train/val、GT instance 解码/裁剪、teacher 缓存、实例评价、对照实验 | `instance_dataset.py`、`teacher_cache.py`、`eval_instance.py`、数据清单 | A 的相机约定与 B 的 query 编号 |

最先由 A/B/C 一起确定 `contracts.py` 和一组小样本 fixture。之后 B 可以使用缓存训练，A 做在线提取/几何回归，C 准备更多数据和 teacher，不必串行等待所有模块。

若只有两个人：A 负责 frozen base、特征缓存、接回主线及几何评测；B 负责实例数据读取、学生训练和实例指标。先使用 GT 标签，推迟复杂 teacher 流水线，避免 B 同时背负模型与原始 ScanNet/mesh 处理。

## 3. 必须先冻结的数据接口

建议统一成以下逻辑结构，具体 Python dataclass/字典由接口提交实现：

```python
FrozenSceneBatch = {
    "images": Tensor[B, V, 3, 256, 256],       # RGB [-1, 1]，与原 Uni3R 一致
    "intrinsics": Tensor[B, V, 3, 3],         # 像素单位，完成同步裁剪后
    "patch_tokens": {
        4: Tensor[B, V, 256, 2048],
        11: Tensor[B, V, 256, 2048],
        17: Tensor[B, V, 256, 2048],
        23: Tensor[B, V, 256, 2048],
    },                                       # 已剔除 camera/register/intrinsic tokens
    "means": Tensor[B, V, 256, 256, 3],       # 原模型共同坐标系
    "geometry_confidence": Tensor[B, V, 256, 256],
    "gaussians": {...},                      # 原模型完整属性，A 管理其打包方式
    "metadata": {...},
}

InstanceTargets = {
    "instance_ids": LongTensor[B, V, 256, 256],
    "valid_mask": BoolTensor[B, V, 256, 256],
    "confidence": Tensor[B, V, 256, 256],
    "scene_id": ...,
    "frame_ids": ...,
    "label_source": "gt" or "pseudo",
}

InstancePrediction = {
    "mask_logits": Tensor[B, Q, V, 256, 256],
    "objectness_logits": Tensor[B, Q],
    "query_features": Tensor[B, Q, 256],
}
```

注意：

- 这里的 shape 是明确的目标接口，不是宣称当前仓库已经有这些类。
- 四层 patch tokens 每层 2048 维。原 DPT 可通过适配其层索引读取四层特征；拷贝模块后相应调整 `intermediate_layer_idx`/`patch_start_idx`，不要补出伪特征或误吞特殊 token。
- Gaussian 展平顺序固定为 `index = view*H*W + y*W + x`，与 source-pixel mask 完全一致。若以后做 Gaussian filtering，必须返回 retained indices。
- K 个 teacher 实例可以在当前场景内重新编号，但所有输入视角必须共用同一个映射。不能在每张图各自从 1 编号。
- 没有物体标签的可信 stuff 背景，与缺失/不可信标签的 ignore 区域分开保存。
- 训练输入不包含 GT XYZ/GT 外参。它们可以用于离线标签、诊断或训练 renderer 的坐标变换，不能悄悄进入学生 encoder。
- 模型输出的 XYZ/confidence、camera、Gaussian attributes 都 detach。confidence 不先当成经过校准的概率。

每份缓存携带：base checkpoint SHA256、源码/补丁版本、feature contract version、层索引、输入尺寸、RGB normalization、intrinsic convention、frame 顺序、裁剪参数、缓存 dtype。

## 4. A 的工作：主线怎么改

1. 保留原 Uni3R 作为 reference，记录相同输入下 centers/scales/rotation/opacity/SH 和渲染结果。
2. 新建外层 `FrozenUni3R` wrapper，加载同一 `2views.pth`。从 aggregator 后导出四层 tokens，并导出 Gaussians、XYZ、confidence。
3. 原 `vggt.aggregator/camera_head/point_head/depth_head/dpt_head/gs_attr_proj` 全部冻结。显式设置 `requires_grad_(False)`，并保证外层 `.train()` 不会把 base 改回 train 模式。
4. 第一版可以保留原 forward 验证正确性；随后跳过 LSeg、feature reduction/expansion、distiller 的无用计算。移除步骤必须做 reference output 回归。
5. 原属性头保留 123 通道，减少改变卷积数值路径和 checkpoint 映射的风险。几何位置来自 `vggt.point_head`，不能只冻结外面的 DPT。
6. 输出一份真实的 feature fixture 给 B/C，再输出小规模训练缓存。
7. B 交付 student checkpoint 后，组合为 `FullModel(base, student)`；base 输出同时交给 renderer 和 student，再把 query membership 转成 Gaussian instance IDs。

接回主线时不需要重新训练整个模型，也不需要把学生权重混进原 `2views.pth`。建议分别保存、共同加载，并验证版本兼容。

## 5. B 的工作：学生怎么单独训练

第一版学生组成：

- `instance_dpt`：复制原 `dpt_head` 权重，独立实例，不共享可训练参数。
- 多尺度 adapter、XYZ positional encoder。
- 跨视角共享的 100 个 object queries，hidden dim 256、6 层 decoder。
- 64 维 mask features 与 query mask embeddings 的点积输出 membership logits。
- objectness head；先不加语义类别、语言 grounding 或额外聚类。

Query attention 先读取下采样的多尺度特征（例如 1/8、1/16），不要对全部高分辨率像素做全局 self-attention。高分辨率 mask 通过轻量 mask feature 与 query 点积恢复。保留背景/未分配状态：低 objectness query 丢弃，所有 query membership 都低的 Gaussian 设为未分配，不能强制把墙面和未知区域塞进某个物体。阈值在验证集固定，测试集不调参。

第一版只在 source-pixel masks 上监督，跨所有输入视角共同计算一次 Hungarian matching。初始权重 BCE/Dice/objectness = 5/5/2。伪标签的 ignore、confidence、未匹配 query 的负监督权重由标签质量决定。

起始优化：新模块 LR=1e-4；复制的 DPT LR=1e-5；AdamW、clip=1、batch size 1、累积 4 场景。先 GT supervision，确认有效后从相同未进行 GT-instance 训练的初始化建立独立 pseudo-label 实验。

验证顺序：

1. 真实缓存单个场景过拟合，确认同一对象在两个视角被同一个 query 覆盖。
2. 多训练场景与独立验证场景，确认不是背场景。
3. 相同标签下训练 embedding+HDBSCAN 基线及 SIU3R-style query 基线。
4. source supervision 有效后再增加 frozen-Gaussian membership rendering loss。渲染算子对 mask channels 可导，对几何不可导。

学生 checkpoint 只存学生 state dict 和必要 optimizer/config/provenance，不重复存原始几 GB 的重建权重。

## 6. C 的工作：数据怎么准备

当前已下载的三个重合场景属于验证诊断数据，不能接着作为正式学生训练集，再用它们报告泛化。

第一阶段从明确的 ScanNet/SIU3R train split 准备少量训练场景和独立 val 场景，提供 RGB、内参和 GT instance maps；跨扫描序列按物理场景检查泄漏。记录 base 已知训练来源，不能承诺未知 base 数据清单绝无重叠。

Instance RGB PNG 必须按 `R + 256*G + 65536*B` 解码。SIU3R 的完整 segment ID 含 `class*1000 + instance_id`，只取类别会变成语义分割，只取余数又可能把不同类别对象误合并。

所有 resize/crop 同步作用于 RGB、depth、instance、valid mask、intrinsics。ID 图只用 nearest interpolation。

GT 版本有效后再引入 MV3DIS/SAM2Object。该阶段需要原始/处理后的几何资产和 dense views，不能把现有 SIU3R 的几个 RGB-D frames 当作完整 teacher 数据流水线已经就绪。

第一阶段评价：固定 GT ID 的 lifting 诊断、学生的 scene-level set matching、cross-view IoU、mask AP；有完整 3D 标签/正确的 observed-surface 协议后再报告 3D AP。参考几何相机优化、场景对齐和真实部署指标分开。

## 7. 并行推进次序

| 里程碑 | A | B | C | 共同验收 |
|---|---|---|---|---|
| M0：接口 | reference outputs、一个真实 feature fixture | 用 fixture 跑通 student forward/loss | 核对同一 fixture 的 GT IDs/crop | frame、pixel、Gaussian 一一对应 |
| M1：单场景 | 导出更多固定样本；base 回归 | 单场景 GT overfit | 标签可视化、跨视角对应检查 | 匹配后 Dice 约 0.90 的简单样例；重建不变 |
| M2：小规模 | 在线 wrapper 与缓存对照 | 冻结学生训练、多场景验证 | 独立 train/val 和 baseline evaluation | 未见场景有有效实例结果 |
| M3：接回 | RGB→base→student 联合推理 | student checkpoint 与 config | 固定测试清单回归 | 缓存/在线输出一致，无 teacher 依赖 |
| M4：蒸馏 | 支持 renderer 的几何与相机接口 | 独立 pseudo-label student | teacher 缓存、质量/覆盖分析 | 同监督、同 base 的控制实验 |
| M5：论文实验 | 可选 8-view/adapter 扩展 | loss/表示消融 | 多 seed、效率、外部 baseline | 明确性能收益及限制 |

先把 M0–M3 完成；不把 8-view、预测几何 teacher、joint fine-tuning 都塞进第一次集成。

## 8. 缓存的作用与限制

- 两视角四层 patch tokens 的 fp16 缓存约 8 MiB/样本，fp32 约 16 MiB，还需 RGB/XYZ 等。1 万个 fp16 样本仅 tokens 就约 78 GiB。
- 最初 fixture 使用 fp32，排除量化差异；大缓存若改 fp16，在线路径也应采用相同转换，并比较对输出的影响。
- 缓存对应固定 frame pairing/crop，不能任意改变几何裁剪或颜色增强后仍使用旧 tokens。
- B 可从缓存直接训练且不占用原 Uni3R 前向显存。A/C 仍可在独立设备或时段准备数据和评测。
- 扩展到 8-view 时，特征随跨视角融合改变，且官方 base 权重可能更换；必须重新导出缓存并适配学生，不是把 2-view cache 拼成 8-view。

## 9. 接回主线的硬性检查

1. 相同输入、相同 checkpoint、相同 dtype/crop，缓存与在线 tokens 一致或处于预先约定的数值误差内。
2. 相同 student checkpoint 的缓存/在线 mask logits、objectness 一致；不要只看阈值后的 mask。
3. 加入学生前后，原 Gaussian 参数保持一致。RGB renderer 的 kernel 若发生变化，单独记录数值差。
4. 训练一步后检查 base 没有 gradients/参数变化，student 确有有效梯度。
5. 推理无需 GT masks、GT depth、GT pose、SAM 或 Hungarian matching；Hungarian 仅训练/评价使用。
6. 所有输出和评测使用一致的 view/y/x 顺序；scene 内物体编号允许 permutation，但不能每张图重新编号掩盖 ID 错误。
7. 缓存、学生和 base 的版本信息随 checkpoint 交付，加载时发现不兼容立即报错。

## 10. 代码协作边界

建议各自独立分支/worktree，先合并接口与 fixtures，再并行实现；不要三个人同时编辑 `uni3r.py` 或一份大型训练脚本。

```text
uni3r/instance/contracts.py        # 首次共同确定，此后版本化
uni3r/instance/frozen_base.py      # A
uni3r/instance/student.py          # B
uni3r/instance/criterion.py        # B
uni3r/instance/data.py             # C
uni3r/instance/metrics.py          # C
scripts/export_instance_cache.py  # A
train_instance.py                 # B
infer_instance.py                 # A
eval_instance.py                  # C
```

这些是规划中的产物，本次没有启动实例学生训练。GT lifting 诊断已完成 3 场景、18 组输入，代码位于 `reproduce/oracle_instance_lifting.py`，结果见 `reproduce/ORACLE_INSTANCE_RESULTS.md`。新视角 IoU 84.97%，但单源互渲染仅 42.10%，且厘米级三维误差依赖对齐；因此下一步先证明学生能学习和泛化，暂不把跨视角渲染当强监督。

## 11. 第一批开工任务与研究边界

| 人员 | 第一批任务 | 完成标志 |
|---|---|---|
| A | 用固定 `2views.pth` 导出一组真实 tokens、预测 XYZ、Gaussian 对应索引及原输出参照 | B 可直接读取样本；特征带版本与预处理记录 |
| B | 根据共同接口实现学生 forward、跨视角统一匹配和 loss | 真实单场景能够过拟合，同一物体跨视角对应同一 query |
| C | 检查该样本的实例标签，准备另外的训练场景与独立验证场景 | 跨视角 ID 正确，数据清单与 ignore 定义明确 |

三个人先共同核对一个真实样本，然后分头推进。B 不必等全部缓存导完，A 不必等学生训练完，C 也不必先搭好完整 Teacher 才交付 GT 数据。

第一阶段交付物是可运行的 2-view 实例重建系统：分别加载官方 base checkpoint 和学生 checkpoint，输入 RGB 与内参，输出原 Gaussians 及物体归属。验收包含未见场景实例结果、缓存/在线输出一致性、原重建保持，以及推理没有 Teacher/GT 依赖。

Query 分支本身已有 SIU3R 等近邻工作，不能仅凭“冻结 Uni3R + queries”宣称新的方法贡献。后续重点验证：相同 base、数据与标签条件下，实例一致性与蒸馏设计是否优于 embedding + clustering 和 SIU3R-style query 基线。相关调研见 [方法评审](INSTANCE_METHOD_REVIEW.md)。
