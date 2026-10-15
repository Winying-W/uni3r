# B｜B-03 当前入口

状态：**B03_PARTIAL_AWAITING_REVIEW**。B独立的最小CPU训练引擎与严格诊断checkpoint已验证；A真实缓存默认学生eval/no_grad前向通过；C真实GT接入受依赖阻塞。不是完整正式训练流水线，停止等待审核。

- [B-03审核报告](reports/B-03_REVIEW.md)、[本轮结果](results/B-03/20261008T140224Z_cpu_b03/)
- [training.py](training.py)：固定AdamW两组、独立持久8步预算、每窗口1..4个micro场景、mask/objectness各自有效场景归一化、无监督跳步、异常清梯度。
- [checkpointing.py](checkpointing.py)：完整窗口边界保存、原子不覆盖发布、receipt/hash/用途/配置/源码/学生与optimizer有序名称校验、安全weights_only加载、恢复到新引擎。
- [probe_m0.py](probe_m0.py)：只读A已导出fixture、验证来源和tensor合同、独立未优化默认学生前向；不执行A/C入口、不改metadata或输入，不支持无证据猜测C格式。
- [train_smoke_v0.json](configs/train_smoke_v0.json)、[runner](run_b03_checks.py)、[训练测试](tests/test_training.py)、[checkpoint测试](tests/test_checkpointing.py)。

本轮实际5次synthetic optimizer.step，连续和恢复逻辑更新都为3；88项测试中87 PASS、1历史占位SKIP。真实缓存前向PASS不代表真实GT loss/训练/精度通过。完整性声明默认False，B-02 loss定义保持不变。

checkpoint用途固定 `SYNTHETIC_ENGINE_DIAGNOSTIC_NOT_FOR_REAL_TRAINING`，禁止用它初始化后续GT/pseudo实验。真实训练、默认完整学生反向、GPU、AMP未运行，没有部署checkpoint。阶段账本位于既有B-03 run根，不能因重跑或恢复重置；**不要自动重跑完整runner**，剩余预算3不足再执行整个5步恢复协议。

```python
from instance_B.training import BudgetLedger, TrainingEngine
# student为指定synthetic_small真实InstanceStudent；ledger必须指向既有阶段账本。
engine = TrainingEngine(student, BudgetLedger(existing_stage_ledger))
# 每个item使用唯一共享合同，无原始数据解码职责。
# report = engine.run_window([(frozen_batch, gt_targets, False)], label='authorized_window')
# 只有明确阶段授权且预算充足时才能实际执行。
```

历史报告和代码保持只读。下方是B-02及更早阶段说明；当前状态以本节/B-03报告为准。

---

# B｜Uni3R Instance Student — B-02 当前入口

当前状态：**B02_COMPLETE_AWAITING_REVIEW**。跨全部输入视角的精确集合匹配、GT weighted BCE/Dice/objectness 和现有学生 CPU synthetic 反向检查已完成；停止等待审核，不进入 B-03。

- 审核报告：[B-02_REVIEW.md](reports/B-02_REVIEW.md)
- 实测证据：[本轮结果](results/B-02/20261008T130554Z_cpu_b02/)
- 入口：[matcher.py](matcher.py)、[criterion.py](criterion.py)，共用 [loss_v0.json](configs/loss_v0.json)。共享合同 0.1.0 未改。
- 本轮：matcher 9 PASS、criterion 14 PASS、B-02 集成2 PASS；原合同31 PASS、原学生19 PASS、1历史真实占位SKIP。正式尺寸 logits-only 前向 PASS。
- A 已交付 M0 真实 fixture，本轮只读交接信息，真实集成 NOT_RUN；C targets 本次限定检查未发现。旧测试的缺失 fixture 文案是历史事实，不能作为当前 A 状态。

```python
from instance_B.criterion import InstanceCriterion
from instance_B.loss_utils import LossConfig
from uni3r.instance.contracts import validate_training_batch

# batch / targets / prediction 必须来自同一输入网格与顺序。
validate_training_batch(batch, targets, prediction, spec=student.config.contract_spec)
criterion = InstanceCriterion(LossConfig.from_json(), spec=student.config.contract_spec)
result = criterion(prediction, targets, gt_set_exhaustive=[False] * prediction['mask_logits'].shape[0])
result['loss_total'].backward()  # 本阶段到 backward 为止，没有 optimizer
```

gt_set_exhaustive 默认 False；真实调用只有标签提供方确认完整集合后才可设 True。零支持正 ID 会关闭 unmatched 负监督。confidence 是相对像素权重；像素归一→实例等权→有效场景等权；objectness 场景内固定除 Q，no-object=0.1。pseudo 明确拒绝。

runner 从仓库根启动，使用新的结果子目录，mode 为 unit、integration 或 full；不能复用既有目录。integration 回归旧模块并运行真实学生 synthetic backward，full 只实例化预测 logits。命令例：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 \
TMPDIR=/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-02/20261008T130554Z_cpu_b02/tmp XDG_CACHE_HOME=/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-02/20261008T130554Z_cpu_b02/cache \
/root/miniconda3/envs/uni3r/bin/python -B -m instance_B.run_b02_checks \
  --mode unit --run-dir instance_B/results/B-02/<new-unique-run>/unit_01
```

本轮未安装依赖、未使用 GPU、未运行 optimizer.step、未创建训练后学生 checkpoint。学生仍 CPU float32；loss 归约/标量为 float64。无真实训练/精度、默认完整学生反向或 AMP 结论。

---

以下保留 B-00/B-01 历史接口与阶段说明。出现“当前 B-01”“等待 A/C”“尚无 matching/loss”等文字时，均指当时阶段；本轮状态以上述 B-02 入口及报告为准。

# B｜Uni3R Instance Student

B 负责 Instance DPT、multi-scale adapter、共享 object queries、mask decoder、场景级 Hungarian matching、loss、训练与 checkpoint。A 负责冻结 Uni3R、真实特征和 Gaussian 重建；C 负责数据、GT/teacher 与评价。

当前为 **B-01**：实例学生已实现，DPT官方权重初始化和synthetic前向/反向已验证。第一阶段固定官方 `2views.pth`；未来实例训练使用真实GT监督和A的冻结Uni3R特征。真实集成尚未运行，matching/loss/训练循环留到B-02。

- 唯一共享接口：[`uni3r/instance/contracts.py`](../uni3r/instance/contracts.py)
- 长期入口：[`B_PROGRESS.md`](B_PROGRESS.md)；每轮开始先读、进行中更新、关键步骤追加记录、结束写真实状态
- 学生入口：[`student.py`](student.py)，`StudentConfig` / `InstanceStudent`
- 默认配置：[`configs/student_v0.json`](configs/student_v0.json)
- 本轮审核：[`reports/B-01_REVIEW.md`](reports/B-01_REVIEW.md)；历史：[`reports/B-00_REVIEW.md`](reports/B-00_REVIEW.md)
- 实测证据：[`results/B-01/20261008_cpu_01/`](results/B-01/20261008_cpu_01/)
- 合同版本：`0.1.0`，状态 `PROPOSED_FOR_TEAM_INTEGRATION`，A/C 尚未人工确认

## 运行

在远程授权根目录使用已有环境，只运行 CPU 合同测试，无安装、模型下载或 GPU 训练：

```bash
cd /mnt/workspace/Jensen/project/3DGS/Uni3R
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/uni3r/bin/python -B -m unittest discover -s instance_B/tests -v
```

从现有 `uni3r` 包导入；`uni3r/instance` 是其 namespace 子包，不创建第二套同名包，也不修改现有 `uni3r/__init__.py`：

```python
from uni3r.instance.contracts import (
    validate_frozen_scene_batch, validate_instance_targets,
    validate_instance_prediction, validate_training_batch, scene_instance_ids,
)
```

默认 profile 为 `V=2, H=W=256, patch_size=16, token_dim=2048, query_dim=256`，Q 从预测 tensor 推导。小尺寸仅显式传 `ContractSpec` 用于 synthetic 单测；真实 v0.1.0 缓存必须用默认 profile。

## 使用语义

- 校验函数不修改输入，失败抛出带字段名的 `ContractError`；所有 tensor 拒绝 NaN/Inf（包括 ignore 区），frozen/target tensor 拒绝 requires_grad=True。
- `-1` 是 ignore、`0` 是可信背景/stuff、正数是场景级物体 ID。无效标签置信权重必须为 0；GT 有效权重默认由生产者填 1，允许显式 `[0,1]` 重加权，validator 不自动补值。
- `scene_instance_ids()` 返回所有视角中有效且正权重的正 ID 并集。背景和全 ignore 场景得到空 int64 向量；不实现 matching/loss。
- Prediction 为 raw logits，可含负数或大于 1 的值；所有视角共享 query 维。验证器无法凭数值判断是否错误执行过 sigmoid。
- `geometry_confidence` 是非负有限模型分数，当前源码为 `1+exp`，不限定 `[0,1]`；与监督 `confidence` 不同。
- `metadata` 的字段和检查以合同代码为准。`scene_id` 为 B 个字符串，`frame_ids` 为 B×V 字符串；crop 使用原图整数像素边界 `[x0,y0,x1,y1]`、最后两项 exclusive，resize 是 `[H,W]`，description 写出操作顺序/插值/畸变处理。
- `gaussians` 为 A 的不透明原始 payload，不预先声明旋转/尺度/相机语义。唯一保留的合同键 `source_indices` 是长度 B 的 int64 一维 tensor 列表，过滤时必需，使用 `view*H*W+y*W+x`；原字段数量/排列与映射的关联仍待 A 的 schema-specific 集成校验。
- `tensor_dtypes` 使用 `str(tensor.dtype)`，路径如 `patch_tokens.4`、`gaussians.source_indices[0]`，需列出所有 payload tensor。
- 真实缓存需要 checkpoint SHA256、commit、dirty 状态；dirty 时另需源码 fingerprint。synthetic 明确使用 `evidence_kind=synthetic`、`SYNTHETIC_NO_CHECKPOINT`、SHA256=None、`uni3r_git_commit=SYNTHETIC`，不能冒充真实特征。
- 测试类明确区分 `CONTRACT_SYNTHETIC_TEST` 和 `REAL_UNI3R_FIXTURE_TEST`；后者目前仅为明确 skip 的占位，不能视为已集成。

## B-01 最小调用

```python
from pathlib import Path
from instance_B.student import StudentConfig, InstanceStudent, load_dpt_initialization

root = Path('/mnt/workspace/Jensen/project/3DGS/Uni3R')
run = root / 'instance_B/results/B-01/20261008_cpu_01'
config = StudentConfig.from_json(root / 'instance_B/configs/student_v0.json')
initialization = load_dpt_initialization(
    run / 'dpt_init_from_2views.pt', run / 'dpt_init_receipt.json')
student = InstanceStudent(config, dpt_init=initialization)
# A提供满足共享合同的FrozenSceneBatch后：
# prediction = student(frozen_scene_batch)
# 返回mask_logits/objectness_logits/query_features；不读取targets或teacher
```

从仓库根目录运行。实际导入使用本checkout的 `uni3r/dpt_head.py` 和 vendored VGGT DPT。只构造独立DPT，不构造Uni3R、distiller或LSeg。原始模型源码和共享合同未修改。

正式默认 `InstanceStudent()` 在缺少已验证DPT初始化时抛出 `DPT_INIT_BLOCKED`，不会悄悄随机初始化。测试确需随机权重时必须显式 `init_mode='synthetic_random_init'`，且只能接synthetic缓存。初始化文件约132 MB，仅含60个DPT张量，是初始化产物，不是训练成果。

默认：256×256、V2、Q100、hidden256、6层/8头、mask64、FFN1024。降采样memory为1/8和1/16，共2,560 tokens；不做高分辨率memory self-attention。所有视角共用场景内queries，输出 `[B,Q,V,H,W]`，64D点积除以sqrt(64)，不施加sigmoid。

小测试显式 `StudentConfig(height=32, width=32, num_queries=7, decoder_layers=2, feedforward_dim=512, profile='synthetic_small')`；DPT仍是原始完整结构/2048通道，hidden256、mask64不变。小配置不代替默认配置：本轮两者均实际运行。

XYZ在每场景全部V/H/W上用float64求均值中心和RMS半径，分母下限1e-6；先detach再归一化。几何分数用 `log1p(score)` 后平均池化，保留非概率含义。2D像素中心、view embedding和level embedding同时接入。物理坐标系仍待A确认。

当前学生支持同设备float32，AMP尚未验证；合同支持的其他浮点dtype会收到明确错误，不自动转换输入。合同若拒绝attached frozen tensor，学生不会悄悄修复；adapter内部另做detach以隔离单独调用。

纯索引接口 `flatten_membership_logits(mask_logits, source_indices=None)` 使用view-y-x排列，过滤返回每场景 `[Q,Mi]` 列表；没有决定实例阈值或聚类规则。

## 测试与重复运行

上方unittest discovery会运行31项合同测试和19项学生测试；真实fixture占位明确SKIP。本轮第一次实际结果51项：50 PASS、1 SKIP、0失败/错误。

完整检查脚本同时运行默认配置前向并保存证据，可用**新的**结果子目录重复（默认不做optimizer.step）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/uni3r/bin/python -B -m instance_B.run_b01_checks \
  --run-dir instance_B/results/B-01/20261008_cpu_01/checks_02 --diagnostic-steps 0
```

本轮已用2步小配置标量诊断，`optimizer_diagnostic_admission.json`防止无意重跑。诊断目标是logits平方均值，不是实例loss；其数值变化不能证明分割有效。`eval_roundtrip_state_dict.pt`和`diagnostic_after_2steps_state_dict.pt`仅为测试产物。

A仍需交付真实tokens/冻结证据/RGB与几何和Gaussian映射确认；C仍需真实targets/跨视角ID和数据划分。DPT提取已由B完成，可共同核对receipt。B-01结束停在审核门，不自动进入B-02或真实训练。
