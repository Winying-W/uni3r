# B-02 matching 与 loss 审核报告

日期：2026-10-08（Asia/Shanghai）。WORK_PACKAGE=B-02_MATCHING_AND_LOSS。
状态：**B02_COMPLETE_AWAITING_REVIEW**。本报告只证明已运行的 synthetic 匹配、loss 与现有学生反向链路；不代表真实实例训练、精度或默认完整学生反向通过。

据用户交接，ChatGPT 已批准 B-00/B-01 的报告级阶段审核；该批准不代表 ChatGPT 独立读取服务器源码或完成真实集成验收。旧报告的 AWAITING_REVIEW、旧结果和历史诊断预算原样保留。本任务书仅解除了 B-02 进入门槛，未授权 B-03。

## 1. 实际环境与范围

- Windows 指定客户端 `C:\Windows\System32\OpenSSH\ssh.exe`，Host `010`。hostname：`dsw-824375-57c745db88-n6tv9`。
- 实际根目录 `/mnt/workspace/Jensen/project/3DGS/Uni3R`；唯一远程写入边界 `/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B`。所有使用路径先 resolve 检查；未跟随 data 越界链接。
- START_HEAD `7a7f80205fb075228ff75aac43635b96de901110`；END_HEAD `7a7f80205fb075228ff75aac43635b96de901110`；branch `main`。与提示词历史 HEAD 不同，以本次实际核查为准。
- Python 3.10.22，PyTorch 2.1.0+cu121，SciPy 1.15.3；解释器 `/root/miniconda3/envs/uni3r/bin/python`。
- 所有 Python 调用使用 `-B`，CUDA_VISIBLE_DEVICES 为空，OMP/MKL/OPENBLAS_NUM_THREADS=2，PYTHONDONTWRITEBYTECODE=1；runner intra-op=2、inter-op=1。测试子进程在启动前即指向本轮 TMPDIR/TMP/TEMP/XDG_CACHE_HOME。
- Git 只读操作设置 GIT_OPTIONAL_LOCKS=0。没有依赖安装、GPU 调用、optimizer 构造/step、远程 Codex、分支切换、reset/clean、服务器 commit/push。
- 本轮最新任务明确禁止自动 commit/push，因此也没有触发本地审核仓库自动发布；已有发布历史未变。

实际读取：任务书指定的 B_PROGRESS、README、B-00/B-01 报告、共享合同、student/dpt_adapter/student_v0、旧测试与 run_b01_checks；B 的既有 DPT receipt 和初始化；真实被导入的 Uni3R DPT、vendored VGGT DPT/head_act/utils 与包入口。相关小文件保护哈希见 start_audit/additional_source_baseline。
已知报告位置 `reproduce/INSTANCE_TEAM_PLAN.md`、`reproduce/INSTANCE_METHOD_REVIEW.md` 的相关章节已读，旧规划由本任务书冻结范围取代。
对 A 只读 `instance_A/A_PROGRESS.md`、`instance_A/README.md` 和已知 M0 目录文件名。**没有改动或运行 A/C 的代码，也没有加载其大型 fixture/reference。**

新增代码/配置/测试/报告共 9 文件：matcher.py、criterion.py、loss_utils.py、configs/loss_v0.json、tests/test_matcher.py、tests/test_criterion.py、tests/test_b02_integration.py、run_b02_checks.py、reports/B-02_REVIEW.md（均在 instance_B）。只更新 B_PROGRESS.md 与 README.md；其余本轮产物均在 `/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-02/20261008T130554Z_cpu_b02`。

## 2. 唯一共享合同与对齐边界

实际 `uni3r/instance/contracts.py`：CONTRACT_VERSION=0.1.0，CONTRACT_STATUS=PROPOSED_FOR_TEAM_INTEGRATION。
SHA256：`2810deaffde90952ce5f088d12ec112d24aed8c7f583c9498169261687ea072f`，与本机 B-01 保存的合同基线一致；本轮开工/收尾也一致。没有新建外部合同、标签格式或修改 A/C 接口。

matcher/criterion 直接接共享 InstancePrediction 与 InstanceTargets，先调用实际 `validate_instance_targets`、`validate_instance_prediction`，再检查预测/标签的 batch/device。gt 为唯一支持来源；pseudo 抛 GT_ONLY 错误；混合/未知来源由共享 validator 明确拒绝。没有蒸馏路径。
集成调用实际 `validate_training_batch(batch, targets, prediction, spec=...)`。它核对 scene/frame 顺序、shape/device，并校验 frozen batch 的 crop metadata。**InstanceTargets 没有独立 crop provenance 字段，Prediction 也没有 metadata；该合同不能独立证明标签物理裁剪/跨视角身份正确。** 本轮标签直接构造在同一 synthetic identity-crop 像素网格；真实 C 交付仍须提供同步预处理与身份说明。测试未放宽合同：公开接口用合法 16x16 synthetic ContractSpec，学生为 32x32；仅 kernel gradcheck 使用极小内部张量。

## 3. 冻结公式和实现

每 batch 项独立，以固定 `i=view*H*W+y*W+x` 展平全部输入视角。-1 ignore；0 可信背景/stuff；正整数为场景内实例。

```text
w_bi = valid_mask_bi * confidence_bi   (detached label weight)
S_b = sum_i w_bi
u_bi = w_bi / S_b                     only when S_b > 0
I_b = sorted positive IDs having at least one w_bi > 0 pixel
K_b = |I_b|;  T_bki = 1[instance_ids_bi == I_b[k]]

BCE_b(q,k) = sum_i u_bi * (T_bki*softplus(-z_bqi) + (1-T_bki)*softplus(z_bqi))
Dice_b(q,k) = 1 - (2*sum_i u_bi*sigmoid(z_bqi)*T_bki + 1e-6)
                   / (sum_i u_bi*sigmoid(z_bqi) + sum_i u_bi*T_bki + 1e-6)
ObjCost_b(q) = softplus(-o_bq)
C_b(q,k) = 5*BCE_b(q,k) + 5*Dice_b(q,k) + 2*ObjCost_b(q)
pi_b = argmin_injective_pi sum_k C_b(pi(k),k)
```

原始 ID 按整数排序仅作稳定索引，不连续化为外部标签、不取模、不按 scene_id 字符串或 ID 合并 batch。每个返回记录含 batch_index、scene_id、target_instance_ids、query/target indices、原始 matched_instance_ids、unmatched queries、监督像素数/S/排除 ID/状态/cost finite。保证 `matched_instance_ids[j] = target_instance_ids[matched_target_indices[j]]`。

SciPy 1.15.3 的 `scipy.optimize.linear_sum_assignment` 求精确矩形线性分配；不将其实际实现冒称原始 Hungarian 算法。匹配整体 no_grad，最终 cost 为 CPU float64，不改预测 requires_grad，无 ID 微偏置、贪心、逐视角匹配或部分目标采样。有并列最优时只验最优值/合法覆盖；唯一最优时验证 ID/query 置换等变。

confidence 只通过 u 使用一次，表示场景内相对像素监督权重。所有正权重同比缩放且合法时 mask loss 不变；没有实例平均 confidence、w²、geometry_confidence 替代、全场景低 confidence 降权。每实例先得到归一化 mask 项后实例等权，不称为 inverse-area 或完全消除物体大小/可见性差异。单视角可见实例保留；其他视角可信像素为负监督，全 ignore 的视角不贡献。

`loss_v0.json` 是 5/5/2 唯一运行权重来源，matcher 与 criterion 引用同一 LossConfig。实际训练 loss 从原始可导 logits 按离散匹配索引重新计算，不返回 detached matching cost。

```text
e_b = gt_set_exhaustive[b] and S_b>0 and excluded_zero_support_ids is empty
L_obj,b = (sum_matched softplus(-o_bq) + 0.1*e_b*sum_unmatched softplus(o_bq)) / Q
L_bce,b = mean_matched_instance BCE_b(pi(k),k)
L_dice,b = mean_matched_instance Dice_b(pi(k),k)
L_bce = mean_{b:K_b>0} L_bce,b
L_dice = mean_{b:K_b>0} L_dice,b
L_obj = mean_{b:K_b>0 or e_b} L_obj,b
L_total = 5*L_bce + 5*L_dice + 2*L_obj
```

即：**像素按 u 归一化 → 实例等权 → 各 loss 的有效场景等权；objectness 在场景内固定除以 Q。** 不按正负权重和归一，不向 unmatched queries 追加零 mask loss，不建立 background query。
matching cost 是固定分配代理；unmatched objectness 和训练归一化不同，不能说精确分配器最小化整个最终训练 loss。

## 4. 完整性门控和边界

`gt_set_exhaustive` 是 B 私有 bool list/tuple 参数，默认逐项 False，不改变共享合同。True 必须来自标签提供方对当前视角可见实例完整性的明确声明，不能从 label_source=gt、尺寸或 K=0 推导。本轮人工构造数据的实际取值均记录在测试/集成配置；学生为 `[True, False]`，正式形状为 `[True]`。

| 情形 | 实际行为 |
|---|---|
| S=0 | mask/objectness 图连接零；不进入任何有效场景分母；backward 两类 logits 梯度为零 |
| S>0,K=0,e=True | 只监督 objectness：0.1*mean(softplus(o)) |
| S>0,K=0,e=False | 无可用实例级监督，图连接零并记录原因；不进入有效分母 |
| S>0,K>0,e=False | matched mask + matched positive objectness；unmatched objectness 直接梯度为零 |
| S>0,K>0,e=True | 追加权重 0.1 的 unmatched negative objectness |
| 正 ID 全零支持 | 从匹配目标集排除并记录 ID，不视为不存在；若声明 exhaustive，记录 NEGATIVE_SUPERVISION_DISABLED_ZERO_SUPPORT_INSTANCE 并关闭负监督 |
| K=Q | 正常精确匹配，unmatched 为空 |
| K>Q | CapacityError 包含 batch_index/scene_id/K/Q，绝不截断/复制/跳过 |
| 非法 dtype/shape/ID/confidence/source、NaN/Inf | 共享校验或明确错误；ignore 区也拒绝非有限 logits |

零值从原始 mask/objectness 各一个元素先乘零构造，避免巨大 logits 全量求和溢出。追加全 ignore 场景不稀释已有 loss/梯度；纯背景场景不进入 mask 分母。返回未乘团队权重的三个可导标量、total、matches、三个有效场景计数及逐场景诊断。输出标量 float64；学生输入/参数/预测仍 float32，不代表学生支持 float64 或 AMP。

## 5. 分块与资源

按有效像素固定顺序分块，默认 chunk_size=4096，所有正权重像素均使用。匹配阶段按背景/实例组 scatter_add 累加正/负 softplus、sigmoid 质量和 target mass；排除目标组的负项用 prefix/suffix 相加，避免巨大近等量相减。criterion 只对 matched pairs 构造 K×chunk 可导张量，不建 Q×K×N 图。没有 resize、插值、随机点采样、下采样监督、面积过滤或 top-K 目标近似；不使用粗暴 clamp/nan_to_num 修补。

正式尺寸 B1/Q100/V2/256x256/K100：`FULL_SHAPE_LOGIT_MATCH_LOSS_FORWARD=PASS`，耗时 0.755676028s，cost/loss 有限，未实例化学生、未反向。最大自有分块张量 `[100,4096]` float64 = 3,276,800 bytes（3.125 MiB），group matrix `[100,101]`。保守工作张量包络 `12*Q*C*8 + 16*Q*(K+1)*8 + 64*N` = 49,003,008 bytes（46.7339 MiB），小于 64 MiB；包含多份同时存活 chunk、分组矩阵和索引/权重 bookkeeping。它是设计估计，不是 profiler 精确算子峰值。
正式形状独立进程峰值 RSS=732712 KiB；学生回归/联通进程峰值 RSS=1460572 KiB；单测进程峰值 RSS=594460 KiB。RSS 包括解释器、库、分配器、预测和模型，与自有算子工作内存预算不同。

## 6. 实测测试和独立参考

固定随机种子 7302（旧 B-01 测试内部原有固定种子保持不变）。unit_01 共 23 PASS（matcher 9、criterion 14），0 FAIL/ERROR/SKIP，0.846708s。integration_01 共 53 项：52 PASS、1 SKIP，0 FAIL/ERROR，11.052065s，其中合同回归31 PASS、学生回归19 PASS、B-02 集成2 PASS。总计76项发现、75 PASS、1历史SKIP；正式形状前向另计。

逐项名称、状态和 stdout 保存在各 unit_results.json/unittest_output.txt；测试成功未照抄历史数量。覆盖：
1. 固定反贪心矩阵最优值4，query0→target1、query1→target0；40 个 Q≤5 矩形/负值/并列/K0/KQ 样例逐一以 itertools 穷举全部单射核对。
2. 非连续 ID 7/42/9001 的全局双射、query 置换、同步 view 置换；唯一最优映射等变；batch 顺序与相同 scene_id/ID 的隔离。
3. 跨视角只交换第二视角预测或 GT ID 会提高 loss：正确 1.38964814195995，预测交换 23.8879688929869，仅 GT 第二视角交换 23.8879688929869。
4. 手算 weighted BCE/Dice/objectness、完整背景零 logits、相对 confidence 改变与整体缩放、ignore/invalid/零 confidence 位置直接 mask 梯度严格零、可信背景负目标的 logit 梯度为正（梯度下降将降低 logit）。这只论直接 logits 监督，不推论模型特征空间无其他影响。
5. 单视角可见目标、另一视角可信背景/全 ignore；batch 隔离、ignore 追加的 loss/梯度不变、复制场景不翻倍、背景不稀释 mask 分母。
6. 八种空集/partial/exhaustive/零支持门控情况；K=Q/K>Q；非法 shape/dtype/metadata、来源拒绝、NaN/Inf。
7. logits -1000/-100/0/100/1000 下 cost/loss/梯度有限；confidence=1e-300 仍按相对权重使用；全 ignore 的 1e308 logits backward 安全。
8. 优化矩阵与独立 Q×K×N 密集参考比 BCE/Dice/objectness/cost、穷举最优值；matched loss 三项/total 和两类原始预测梯度对照。参考不调用优化分组 helper。chunk=1/7/4096、float32/float64 输入均通过。矩阵最大误差 1.77636e-15，梯度最大误差 6.93889e-18。
9. float64 正常数值对照 atol=1e-10/rtol=1e-8；float32 原始梯度 atol=1e-6/rtol=1e-5。所有归约升为 float64，所以 float32 输入的标量/矩阵同样使用较严格 float64 容差。固定匹配的微型 kernel gradcheck 使用数值差分 eps=1e-6/atol=1e-5/rtol=1e-3；未对离散求解器求导、未放宽正常值 parity 容差。

手算实测：

| 项 | 期望 | 实测 | 绝对误差 |
|---|---:|---:|---:|
| loss_bce | 0.42283710848783562 | 0.42283710848783568 | 5.55112e-17 |
| loss_dice | 0.24999981250014058 | 0.24999981250014058 | 0 |
| loss_objectness | 0.17849839525388769 | 0.17849839525388772 | 2.77556e-17 |
| loss_total | 3.7211813954476565 | 3.7211813954476565 | 0 |

q0→ID7；u=[2/3,1/3]。matching 未加权 BCE=[0.4228371084878357,1.1552453009332422]，Dice=[0.24999981250014058,0.6666660000006667]，ObjCost=[0.2876820724517809,0.6931471805599453]；总 cost=[3.9395487498434427,10.495850865789436]。分配 cost 与最终训练 total 不相同，符合不同 objectness 归一化的定义。完整背景空场景 L_obj=0.069314718055994526，total=0.13862943611198905。

## 7. 现有学生反向联通

直接调用未改动 `instance_B.student.InstanceStudent`，没有玩具替代。B2/V2/H=W32/Q7/hidden256/decoder2/FFN512、原完整 DPT、CPU float32。两场景具有非对称前景/背景、跨视角共享 ID、单视角目标、ignore 和非均匀 confidence。
初始化只用 `results/B-01/20261008_cpu_01/dpt_init_from_2views.pt`，现有 receipt/加载器验证 SHA256 `a47b647c1d545b491dcb649ee509e911eea6d2ebd9b5a43a8642441d955b6b45`。runner 仅缓存已有加载器返回结果，实测 file-load/hash misses=1，供旧测试与新集成复用；没有修改加载器源码，没有读取诊断后 checkpoint、重新提取或哈希 12GB base checkpoint。

路径：student forward → 共享对齐校验 → criterion 内 matcher → 原 logits 重算 loss → backward。输出 mask `[2,7,2,32,32]`、objectness `[2,7]`、queries `[2,7,256]` 均有限。scene0: q1→7、q5→42；scene1: q1→7、q5→9001。
未加权 L_bce=0.68840397797394171，L_dice=0.87832627271831987，L_obj=0.18355076600770331，total=8.2007527854767144。

| 参数组 | 参数张量 | 有梯度 | 有限 | 非零张量 | 非零元素 | 梯度 L2 范数 |
|---|---:|---:|---:|---:|---:|---:|
| object_queries | 1 | 1 | 1 | 1 | 1792 | 3.82972990593 |
| instance_dpt | 60 | 60 | 60 | 60 | 20303748 | 2.97651639309 |
| adapter | 8 | 8 | 8 | 8 | 132608 | 4.85265692051 |
| position_encoder | 10 | 10 | 10 | 10 | 69063 | 0.0237695669176 |
| decoder | 36 | 36 | 36 | 36 | 1581516 | 8.44623578216 |
| query_norm | 2 | 2 | 2 | 2 | 512 | 0.241417321485 |
| mask_features | 2 | 2 | 2 | 2 | 16448 | 2.84733160602 |
| query_mask_embedding | 2 | 2 | 2 | 2 | 16448 | 3.1448704203 |
| objectness_head | 2 | 2 | 2 | 2 | 257 | 3.09312622029 |

合计 123 个参与参数张量全部有有限非零梯度；不意味着每个参数元素非零。frozen inputs 与 targets 均无梯度。学生参数前后摘要一致 `aaf83f2be381ba18a0299ec61354d1628076dbd97e1b0821023d61e5779055b7`；DPT 初始化源 tensor 摘要一致 `acd62a79d02bbd33f9dc7f715449c95274024f1aaf7e1a141960b4347419bc6b`。无 optimizer 更新，无训练后学生 checkpoint；旧测试 roundtrip 在 BytesIO 中，未再保存完整 state_dict。

## 8. 失败、警告、A/C 依赖与未验证项

- 环境初查一次 Python `-c` 命令因 Windows/SSH 引号转义出现 SyntaxError；当时只读，无科研文件写入，改 stdin 脚本后成功。全部测试首次运行通过，没有隐藏失败或为过测修改旧测试。
- 报告首次生成遇到 f-string 公式花括号未转义的 NameError；当时仅已保存合同对比 JSON，尚未写报告或更新文档。修正报告模板后续写，既有结果相同内容只校验，不覆盖；不是测试或模型失败。
- PyTorch TypedStorage deprecation warning 保留在 integration_01_console.txt，未升级或安装依赖。
- 原真实 fixture 占位测试仍显示“A未提供”的旧 skip 字符串。该旧测试保持只读，**文字已过时**：当前 A_PROGRESS 声明 A_M0_REAL_FIXTURE_VERIFIED，`instance_A/results/M0_20261008_01/fixture.pt` 已存在；本轮只读交接/目录索引，未加载验证。应记“已有交付/待核对；本轮 REAL_INTEGRATION=NOT_RUN”。
- C：本次限定检查未发现合同格式 GT targets/交接目录，不推论队友尚未完成。A 交接说明自身没有 InstanceTargets；将来仍需 C 核对同一 fixture 的 ID、ignore、完整 crop/resize 链和 GT 完整性，且真实完整性声明默认不得凭空设 True。
- 短 Git 状态与开工相同：True。这不是全仓内容未改变的证明，instance_B 本已整体 untracked；本轮用实际写入清单和小文件哈希补充，不归因或回滚队友的共享 dirty 工作区。
- REAL_INTEGRATION=NOT_RUN；REAL_TRAINING=NOT_RUN；DEFAULT_FULL_STUDENT_BACKWARD=NOT_RUN；GPU_USED=NO；AMP=NOT_RUN；PSEUDO_DISTILLATION=NOT_SUPPORTED；OPTIMIZER_STEPS=0。没有分割精度、学习能力、收敛或真实分割效果结论。

## 9. 审计、证据与停止

16 个必要小文件保护 SHA256 与开工一致；历史 B 结果文件 size/mtime 全部一致（不对历史大 checkpoint 重哈希），DPT 文件只按既有加载器核验一次。写入清单、实际读回 SHA256、边界结论见 `boundary_audit.json`、`file_changes.json`、`source_sha256_manifest.json`。不宣称全仓无并发修改。代码、报告、日志、tmp/cache 全部远程落在 instance_B；A/C/共享合同/B-01 源码测试报告未写入。

证据根目录：`/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-02/20261008T130554Z_cpu_b02`。保存完整 loss/runner/student 配置、commands.txt、console、逐项测试、手算与匹配 cost、cross-view 反例、dense parity、逐参数梯度、正式尺寸资源、依赖索引、保护基线、收尾审计和 completion.json。无大型 logits dump 或 trained checkpoint。

必要的下一阶段条件：主控审核本报告；核对 A 的真实 fixture 与 C 同一 scene/frame/crop 的合法 GT targets、标签完整性说明及数据用途/划分；另行明确真实集成或后续阶段授权。本轮到此停止，不启动 B-03、训练、过拟合或额外 optimizer 诊断。
