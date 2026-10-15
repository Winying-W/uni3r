# A 首批交付：真实 fixture 与冻结重建回归

日期：2026-10-08。状态：**A_M0_REAL_FIXTURE_VERIFIED**。

A 的 M0 产物已经完成；联合 M0 的真实实例标签/loss 验收仍需 B/C。M1 过拟合、M2 泛化、M3 已训练学生接回尚未完成。

## 可直接交给 B/C 的文件

- `instance_A/results/M0_20261008_01/fixture.pt`：约 100 MiB，fp32 真实特征/几何/原 Gaussian 输出。
- 同目录 `reference.pt`：约 81 MiB，直接调用原模型所得输出与源相机 RGB。
- 同目录 `metadata.json`：输入帧、裁剪与 K、特征及 Gaussian 约定、版本。
- 同目录 `report.json`：数值回归、梯度隔离、文件 SHA256 与运行环境。
- 同目录 `source_manifest.json`：源码逐文件哈希；被修改的工作区不只靠 Git HEAD 追溯。
- `instance_A/README.md`：调用方式、schema、范围和后续接入要求。
- `logs/instance_A_M0_01.log`：完整实际 GPU 运行日志。

真实样本为官方 `scannet_test/scene0686_01`，输入顺序 `000000.jpg`、`000020.jpg`。原图 512×386，作者 loader 缩放至 339×256，再裁 `[42,0,298,256]` 为 256×256。完整逐步操作、原始与处理后内参、原始 RGB/NPZ 哈希均在 metadata 中。

该样本仅用于诊断，不用于正式训练或泛化报告。没有导出 InstanceTargets，也没有把语义图冒充实例图。

## 权重与执行设置

实际重新全量计算了官方 `checkpoints/output/2views.pth` 的 SHA256：

```text
660aed13d6a8bccd078be35210f803d0f6d3b312f63dd468bbccc1c7f3aacf24
```

源码基于 `7a7f80205fb075228ff75aac43635b96de901110` 加当前工作区变更，精确文件哈希见清单。PyTorch 2.1.0+cu121，A100 80GB，seed=777；fp32、autocast/TF32 关闭。完整 LSeg/distiller/123-channel 属性头保留；未改变原重建源码。峰值 allocated 显存 12.3564 GiB。

## 实测结果

| 检查 | 结果 |
|---|---|
| 直接原 forward vs FrozenUni3R 的 28 个输出张量 | 全部逐元素一致，最大绝对差 0 |
| 四层 reference tokens vs 包装导出 | 全部一致，最大绝对差 0 |
| 原 renderer 的两张预测源相机 RGB | 完全一致，最大绝对差 0 |
| fixture 保存/读取后的 37 个张量 | 全部一致，最大绝对差 0 |
| B 的复制 DPT vs 原 DPT（真实 tokens） | 完全一致，最大绝对差 0 |
| B 默认学生的在线 vs 缓存输出 | mask/objectness/query 三项全部一致，最大绝对差 0 |
| 无标签诊断一步的学生梯度 | norm=48.3901，query 参数确实更新 |
| 同一步的 base 隔离 | 无梯度，参数地址/版本不变，全部 requires_grad=False |
| 父模块 `.train()` | base 全部子模块保持 eval |
| A 边界单测 | 4/4 通过 |
| B 现有单测回归 | 51 项：50 通过，1 个原有真实 fixture 占位跳过 |

学生是默认 100-query/6-layer 配置，DPT 使用 B 验证过的官方拷贝，其他头随机初始化。训练一步仅用 `mean(objectness_logits**2)` 检查梯度隔离，结果不保存为学生权重，不是实例学习或效果证据。

RGB 回归调用原 `cuda_splatting.render`，使用预测源相机、`scale=None`、原默认 active SH degree 0，未做 GT pose 优化；该检查只证明接入前后不变，不是新视角质量成绩。原协方差/渲染器的四元数解释差异已在 README 中记录，未静默改变基线。

## 接下来的协作边界

1. C 确认 fixture 的实例 ID/ignore/crop，并交付独立正式 train/val 输入清单。处理标签必须重放完整 `crop_resize.operations`，不能仅按最终尺寸拉伸。
2. B 用该真实 fixture 完成正式 loss 接入；收到 C 的训练样本后再做单场景 GT overfit。
3. A 按相同预处理导出正式小规模缓存；B 提供已训练 checkpoint/config 后，再实现组合推理及 Gaussian instance IDs，并重跑在线/缓存回归。

现有作者同版本完整训练数据的访问与版本确认问题未被本交付解决。没有启动正式实例训练，没有将外部服务器的历史实验冒充此次结果。
