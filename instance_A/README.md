# A：冻结重建、真实缓存与组合推理

本目录承接 `reproduce/INSTANCE_TEAM_PLAN.md` 的 A 工作。第一批只完成 A 的 M0 交付；不改 B 的模型/损失或 C 的标签与场景划分。

## 代码

- `uni3r/instance/frozen_base.py`：`FrozenUni3R(base, provenance)`，完整执行原 Uni3R forward。全部 base 参数冻结，父模块 `.train()` 后 base 仍为 eval；异常时移除临时 hook。
- `scripts/export_instance_cache.py`：官方 test 样本导出、完整 checkpoint SHA256、当前源码清单、原输出与 RGB 回归、可选 B 实际接口与梯度隔离检查。
- `instance_A/verify_fixture.py`：无需加载 base 的 CPU 交付验证。
- `instance_A/tests/test_frozen_boundary.py`：冻结模式保护、异常清理、错误输入和误开启 base 梯度检查。

## 运行

在 Uni3R 根目录，输出目录必须尚不存在，防止覆盖已有实验：

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=5 python scripts/export_instance_cache.py \
  --output instance_A/results/my_m0_run --sample-index 0 --check-student
python instance_A/verify_fixture.py instance_A/results/my_m0_run
python -m unittest discover -s instance_A/tests -v
```

根据当前空闲显存选择 GPU。脚本完整执行 LSeg/distiller/123-channel 属性头，不做删头优化。fixture 全部浮点字段为原 fp32 路径；autocast 和 TF32 关闭。导出器当前专用于官方 test 诊断样本；正式训练缓存等待 C 提供已确认的 train/val 及一致预处理。

## B/C 如何读取

```python
import torch
from uni3r.instance.contracts import validate_frozen_scene_batch

batch = torch.load("instance_A/results/my_m0_run/fixture.pt",
                   map_location="cpu", weights_only=True)
validate_frozen_scene_batch(batch)
# batch 可以按同一结构搬到 student 所在设备后调用 student(batch)。
# 本交付没有 InstanceTargets，不能把 semantic maps 当作实例标签。
```

`fixture.pt` 包含四层 patch-only tokens、RGB、裁剪后像素内参、预测 XYZ/confidence、完整原输出和 metadata；`reference.pt` 是直接调用原模型的输出与预测源相机下 RGB。每次运行还生成 `metadata.json`、`source_manifest.json`、输入和 reference RGB PNG、成功时的 `report.json`。只有 `report.status=A_M0_REAL_FIXTURE_VERIFIED` 表示完整检查通过，未完成目录不得当作已验收结果。

## A 的 Gaussian schema：`A.raw_views.v1`

`batch['gaussians']['views']` 保留原模型的两元素列表，每个元素的 batch 维也保留：

| 字段 | 形状 | 解释 |
|---|---|---|
| means | B,H,W,3 | 与 `batch['means'][:,view]` 完全相同的预测 XYZ |
| scales | B,H,W,3 | 原激活、邻域距离缩放及 clamp 后的 scale |
| rotations | B,H,W,4 | 原归一化后的四元数值，保持原顺序 |
| covs | B,H,W,3,3 | 原 `build_covariance(scales, rotations)` 的输出 |
| opacities | B,H,W,1 | 原 sigmoid opacity |
| sh_coeffs | B,H,W,16,3 | 原 RGB residual 和 SH mask 处理后的系数 |
| gs_feats | B,H,W,64 | 原 LSeg residual 合并后的特征 |

其余 `depth/extr/intr/distiller_*` 字段原样保留，具体 shape 在 report 中。`extr` 是预测 world-to-camera；这些坐标不经 GT 对齐，不承诺物理米制。point confidence 是 `1+exp(raw)`，不是校准概率。

`source_indices[b] = arange(131072)`；第 `view*256*256+y*256+x` 个 Gaussian 来自相应输入像素，没有筛选。消费端按 view 顺序拼接，再按 y/x 展平，不能颠倒 view/batch。

四层索引为 4/11/17/23（从 0 开始）。先由原 `VGGT_IntrEmb.forward` 去掉尾部 intrinsic token，再去掉前 5 个 camera/register tokens，得到每层 `[B,2,256,2048]`。

注意原代码已有约定差异：`build_covariance` 按 **xyzw** 解释旋转，而原 gsplat 渲染路径直接传入同一四元数（gsplat 接口约定 wxyz）。A 在此交付中保留原数值和原渲染路径，并不将原 `covs` 和 gsplat 内部协方差宣称为同一语义。若后续修正，必须单独做版本化基线实验，不能在学生接入中静默换序。

## 范围与下一步

测试样本标记 `M0_DIAGNOSTIC_ONLY_NOT_FORMAL_TRAIN_OR_VAL`，只证明特征接口和重建回归。`--check-student` 用 B 默认 100-query/6-layer 模型及已验证 DPT 初始化；其他学生头未训练，不报告实例效果。梯度隔离用无标签诊断目标，更新仅保留在进程内，结束即丢弃。

M0 的联合验收仍需要 C 检查真实 instance IDs/crop，B 接正式 loss；这不等于 M1 单场景过拟合完成。之后 A 配合 C 导出正式小规模缓存，再在 B 交付学生 checkpoint 后实现 `infer_instance.py`。当前作者同版本完整训练数据仍有访问/版本确认依赖，不能用这份 test fixture 代替。

现有权重：`checkpoints/output/2views.pth`（11.10 GiB），VGGT `checkpoints/torch/hub/checkpoints/model.pt`，LSeg `checkpoints/pretrained_models/demo_e200.ckpt`，CLIP `checkpoints/clip/ViT-B-32.pt`。B 的 `dpt_init_from_2views.pt` 是初始化；其两步 synthetic 诊断权重不是正式学生。当前未找到官方 8/16-view 权重。
