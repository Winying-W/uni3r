# B-03 最小训练引擎、checkpoint 与 M0 探测审核

日期：2026-10-08（Asia/Shanghai）。WORK_PACKAGE=B-03_TRAINING_ENGINE_CHECKPOINT_AND_M0_PROBE。
状态：**B03_PARTIAL_AWAITING_REVIEW**。

**B 独立部分完成，A 真实缓存默认学生前向通过；真实 GT 接入受 C 依赖阻塞。** 本轮不是整个 M0 联合验收，不是完整正式训练流水线、真实训练、过拟合或部署模型。完成后停止，等待主控审核，不进入 B-04。

## 1. 审核交接与服务器实际基线

B02_ACCEPTED_CODE_AND_EVIDENCE_REVIEW：据用户交接，ChatGPT 实际审阅的是 GitHub 私有审核仓库 `Jupiter0914/uni3r-instance-b-review` 的 `25f048e93f19ba126338fd509d179437b93c8e71` 快照，结论为 B-02 核心代码静态审查及已提交 synthetic 证据通过。该结论不是 ChatGPT 独立读取服务器当前源码或独立复跑。

本轮远程 START_HEAD=`7a7f80205fb075228ff75aac43635b96de901110`，END_HEAD=`7a7f80205fb075228ff75aac43635b96de901110`，branch=`main`；审核仓库 commit 与服务器 HEAD 分别记录，未据此 reset。hostname=`dsw-824375-57c745db88-n6tv9`，pwd/resolve root=`/mnt/workspace/Jensen/project/3DGS/Uni3R`。
B-02 source_sha256_manifest、boundary_audit 中列出的源码、配置、旧测试/报告和当前文档共 27 项逐文件比对全部一致。当前实际源码未假定等于 GitHub；比对结果保存在 start_audit.json。共享合同仍 0.1.0，SHA256=`2810deaffde90952ce5f088d12ec112d24aed8c7f583c9498169261687ea072f`。

Python 3.10.22、PyTorch 2.1.0+cu121、SciPy 1.15.3；指定解释器 `/root/miniconda3/envs/uni3r/bin/python`。通过 Windows `C:\Windows\System32\OpenSSH\ssh.exe`、Host `010` 执行。Python 使用 `-B`，CUDA_VISIBLE_DEVICES 为空，OMP/MKL/OPENBLAS_NUM_THREADS=2、PYTHONDONTWRITEBYTECODE=1；runner intra-op=2、inter-op=1、确定性算法开启。CPU float32 学生、B-02 float64 loss 归约，无 autocast。只读 Git 设置 GIT_OPTIONAL_LOCKS=0。

## 2. 读写边界

科研读取限 resolve 后位于 Uni3R 内的已知路径。实际读取既有 B-01/B-02 接口、配置、测试、报告、结果清单；官方提取 DPT 初始化及 receipt；原 DPT/VGGT 必需源码；A_PROGRESS、A README、已知 M0 report/metadata/source_manifest 和 fixture。A/C 入口源码只作为必要保护小文件读哈希，**未执行 A/C 的导出、测试、训练或评价脚本，也未修改 A/C 代码**。

A metadata 中 source_rgb 路径指向根目录之外的原始数据；这些路径仅作为 producer 描述保留，未跟随或读取。未加载完整 Uni3R、未重新哈希约 12GB checkpoint，未加载 B-01 诊断后状态。不安装/升级依赖，不使用服务器 Codex/认证目录，不切分支/reset/clean，不提交或推送本机/服务器 Git。

新增：training.py、checkpointing.py、b03_utils.py、probe_m0.py、run_b03_checks.py、configs/train_smoke_v0.json、tests/test_training.py、tests/test_checkpointing.py、reports/B-03_REVIEW.md（均为 instance_B 下）。仅更新 README.md/B_PROGRESS.md，保留旧阶段文字。全部新日志、temp/cache、诊断 checkpoint 与回执位于 `/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03`。B-01/B-02 核心、配置、测试、报告、结果只读。

## 3. 参数组、AdamW 与阶段调用账本

从官方 DPT 初始化新建真实 `InstanceStudent`，synthetic_small 固定 B1/micro、V2/32×32/Q7/hidden256/decoder2/FFN512/dropout0，保留原完整 DPT。已有加载器对 131,820,514-byte 初始化文件与 receipt 验证 **1 次**，在同一 runner 内供旧回归、新测试、诊断和独立真实探测复用；SHA256=`a47b647c1d545b491dcb649ee509e911eea6d2ebd9b5a43a8642441d955b6b45`。

参数名称按确定顺序排序，检查名称、对象和 storage 覆盖，无遗漏/重复；参考初始化源、冻结输入、标签不进入 optimizer。

| 参数组 | 张量数 | 元素数 | LR |
|---|---:|---:|---:|
| instance_dpt | 60 | 32950528 | 1e-05 |
| new_modules | 63 | 1818753 | 0.0001 |

两组全部可训练参数均 weight_decay=0.05，无 bias/norm 豁免；betas=(0.9,0.999)、eps=1e-8、amsgrad=False、foreach=False、fused=False；已核对当前 PyTorch 实际 AdamW 签名。学习率常数，不实现或声称 warmup/cosine 已验证。完整有序参数名、超参数保存在 effective_configuration.json 和 training_tests_test_training_evidence.json。

每有效窗口全部 backward 完成后，检查已产生梯度有限，对全部学生参数联合 `clip_grad_norm_(max_norm=1.0, foreach=False, error_if_nonfinite=True)` 一次，再执行一次 AdamW step，最后 zero_grad(set_to_none=True)。W1 裁剪前 float64 复核范数 7.59913206079602，裁剪后 1.00000099396221。后者比 1 高约 9.94e-7：实际 float32 PyTorch 裁剪的归约/舍入误差，已原样记录；未通过改学生精度或多次裁剪修饰结果。

独立于 checkpoint 的持久账本：`/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03/optimizer_budget.jsonl`。唯一阶段 ledger 在每次调用前 flock 锁定、登记并 fsync，记录 RESERVED → CALLING → SUCCEEDED/FAILED。恢复逻辑计数不恢复物理预算；发现未完成预约或第二份阶段 ledger 则拒绝继续。预算上限8，实际预约/调用/成功均 **5**，新 Python 进程复读仍为5。

物理顺序：CONTINUOUS_W1、CONTINUOUS_W2、CONTINUOUS_W3、RESUMED_W2、RESUMED_W3。连续和恢复路径最终 global_update_step 都为3。无监督 I 在两条路径均不 step。余下3次不是自动继续的授权；本轮到此停止。真实数据 step=0。

## 4. 正确的有效场景梯度累积

引擎只接受已构造的共享 FrozenSceneBatch、InstanceTargets 和可选完整性 bool，不承担 RGB/GT 解码、裁剪、标签转换或特征提取。当前优化器路径只准入授权 synthetic_small CPU synthetic 输入，真实数据明确拒绝。

窗口固定1..4场景，micro-batch=1。先 no_grad 校验整个窗口，调用实际 validate_training_batch 和 B-02 prepare_scene，检查 GT 来源、容量、scene/frame/device，以及每场景的 K/S/完整性/零支持 ID。criterion 返回的 m_j/o_j 必须与预计算一致。

```text
M = sum_j mask_active_scene_count_j
O = sum_j objectness_active_scene_count_j
contribution_j = 5*(m_j/M)*loss_bce_j
               + 5*(m_j/M)*loss_dice_j
               + 2*(o_j/O)*loss_objectness_j
```

M 或 O 为0时，相应项使用原始预测构造的图连接零，不做除法；不按4或总场景数平均。逐 micro backward 后释放该图，不保留四份完整图；窗口只 clip/step 一次。尾窗口依据真实有效计数 flush。
整窗口 M=O=0 在学生前向前跳过，不 backward/clip/step，不改参数、buffer、AdamW moments/step、LR 或 global_update_step；seen/skipped/data/window cursor 按消费样本推进。使用监督计数而非 loss==0 判断；图连接零故障注入样例有监督时仍为 BACKWARD_READY，未为这个分支另花 optimizer 步数。

独立参考直接将相同四场景拼接成一个 batch，一次调用未改 B-02 criterion，不调用 training.py 归一化 helper。两路径同一学生、train 模式、dropout0，没有参数更新；实际无 BatchNorm 类有状态 batch 层。四场景为完整实例GT、partial实例GT、完整背景、全ignore，M=2/O=3：

| 项 | 累积结果 | 拼接 batch 参考 | 绝对误差 |
|---|---:|---:|---:|
| BCE | 0.69399007121088319 | 0.69399007701000459 | 5.79912e-09 |
| Dice | 0.85298828041956021 | 0.85298828021854378 | 2.01016e-10 |
| Objectness | 0.14265312260633672 | 0.14265312978065717 | 7.17432e-09 |
| Total | 8.0201980033648894 | 8.0201980457040563 | 4.23392e-08 |

学生梯度最大绝对误差 7.89761543274e-07，L2差 4.10779604817e-06，参考梯度L2范数 7.59913184754。固定标量容差 atol=1e-6/rtol=1e-5，float32梯度 atol=1e-5/rtol=1e-4，未放宽。3场景尾窗口 M2/O3 的 total误差 2.41023965231e-08，梯度最大差 7.45058059692e-07，L2差 3.44484746483e-06。完整记录含每个参数误差、三项loss、贡献和计数。

追加ignore的loss/原场景参数梯度逐元素不变；背景只进入objectness分母。默认完整性False、partial unmatched直接objectness梯度为0、零支持9001关闭负监督均保持B-02定义，pseudo拒绝。非有限输入/loss/gradient、K>Q、错序、pseudo均明确失败并清梯度，step调用数不增加，不当作普通跳过。

## 5. 更新和已有动量后的跳步

W1全部123个参与参数张量均有有限非零数据梯度，DPT与新模块发生实际参数变化。逐组锚点记录数据梯度、前后参数差和“仅weight decay”的预期差，不能仅凭权重改变推论数据路径有效。

| 模块 | 参数张量 | 有梯度 | 有限 | 非零张量 | 裁剪前梯度L2范数 |
|---|---:|---:|---:|---:|---:|
| object_queries | 1 | 1 | 1 | 1 | 2.33013180629 |
| instance_dpt | 60 | 60 | 60 | 60 | 2.06771522541 |
| adapter | 8 | 8 | 8 | 8 | 2.41716705639 |
| position_encoder | 10 | 10 | 10 | 10 | 0.0177856080026 |
| decoder | 36 | 36 | 36 | 36 | 4.97600195748 |
| query_norm | 2 | 2 | 2 | 2 | 0.151988097449 |
| mask_features | 2 | 2 | 2 | 2 | 3.03684159991 |
| query_mask_embedding | 2 | 2 | 2 | 2 | 2.63636875489 |
| objectness_head | 2 | 2 | 2 | 2 | 1.11458140382 |

DPT锚点真实参数差 -1.00433826447e-05，仅衰减预计 -4.34846319258e-08；object_queries真实差 -0.000100146979094，仅衰减预计 -1.47641254589e-07。各组锚点均有非零数据梯度且更新不同于仅衰减。输入/标签 tensor、DPT 初始化源保持不变且无梯度；没有加载/重新核验整个 Uni3R 主干参数。

W1有效step创建AdamW状态后，I窗口（全ignore + 无完整性声明的纯背景）跳过。参数/buffer、moments、每参数step、LR、global_update_step逐项摘要完全不变。计数从 `{'global_update_step': 1, 'seen_scenes': 4, 'skipped_scenes': 1, 'data_cursor': 4, 'window_cursor': 1, 'failed_windows': 0}` 变为 `{'global_update_step': 1, 'seen_scenes': 6, 'skipped_scenes': 3, 'data_cursor': 6, 'window_cursor': 2, 'failed_windows': 0}`；global_update_step保持1，seen/data cursor4→6、skipped1→3、window1→2。没有学生前向，物理step计数不变。

## 6. checkpoint 格式、严格安全加载与恢复

format_version=1；purpose固定 **SYNTHETIC_ENGINE_DIAGNOSTIC_NOT_FOR_REAL_TRAINING**；training_track=synthetic_engine_diagnostic。**不能用于后续真实GT或pseudo实验初始化，也不是部署checkpoint。**

保存学生state_dict和AdamW state_dict、每组有序参数名称/超参数、完整规范化Student/Loss/Train配置及摘要、独立有效场景归一化/常数LR策略、global/seen/skipped/failure计数、data/window cursor、torch CPU RNG与实际使用data/eval Generator状态、合同版本/hash、学生/loss/训练/checkpoint小源码摘要、base/DPT来源hash、synthetic样本清单/种子/完整性声明、环境版本。未使用Python random或NumPy采样；NumPy只用于tensor字节摘要，不作为采样器。

只在完整窗口边界且无未提交梯度时保存，明确拒绝窗口中途resume。唯一临时文件exclusive写入、flush/fsync后，以同目录硬链接原子发布到不存在的目标，目标已存在即拒绝；成功后仅清理本调用新建临时链接，不删除旧产物。receipt记录大小、SHA256和保存状态摘要。

加载仅接受授权B-03路径和已有匹配receipt，最大768MiB；同一文件描述符核对大小/hash后 `torch.load(map_location='cpu', weights_only=True)`，不回退不受限pickle。完整校验格式、用途、来源、完整配置、环境、学生key/shape/dtype/有限性、optimizer有序名称/组超参数与state shape/dtype/有限性、计数/游标/RNG后，才在**另建引擎**中 strict=True 加载。失败不修改当前引擎；新候选构造失败恢复调用前RNG。optimizer整数ID同时由保存有序名称校验，不能盲目zip。每参数AdamW局部step检查正整数且不大于global step，未机械要求每个参数局部step相等。

有效诊断checkpoint只保存1份：`/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03/attempt_01/synthetic_engine_W1.pt`，417403412 bytes，SHA256=`a1f3ec0c8bc41056037535540a3bc0bc6ecb456333066c538dd934cc25412672`。另外只有一个13-byte截断负面样例，未重复保存大型每步状态。大型checkpoint保留在B-03远程目录，不回传本机。

恢复协议实跑 W1→保存→I→W2→W3；新学生/optimizer从W1恢复后 I→W2→W3。W1保存时全部raw输出精确一致；最终学生完整state_dict、AdamW tensor/非tensor状态、LR、计数、游标、CPU/Generator RNG逐张量字节摘要一致；固定eval的 mask/objectness/query 最大绝对差均0。连续与恢复global step=3，合计5次物理调用。每个场景顺序、cursor和恢复后输入指纹也一致，未以“loss相近”替代精确恢复验收。

4项checkpoint测试包含24种错误payload：Q, layers, contract, base, DPT, source, loss, group_order, optimizer_name_order, student_missing_key, student_shape, student_dtype, student_nonfinite, purpose, format, optimizer_shape, optimizer_nonfinite, negative_second_moment, local_step, missing_cursor, missing_rng, generator_names, mid_window, backbone_key；另覆盖错误hash receipt、13-byte截断安全加载、已存在文件拒绝覆盖和中途保存拒绝。错误payload使用浅拷贝和小tensor替换，不复制多份巨大checkpoint；live engine及RNG不变。

## 7. A真实缓存准入与独立默认前向

实际fixture：`/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_A/results/M0_20261008_01/fixture.pt`，104123367 bytes；SHA256=`351888091f93f3c6c2d651e08ef73aa593894a5152e773c3643a38eff9d85f54`，与A既有report的artifact_sha256一致。metadata/source_manifest的文件hash亦与producer回执一致；当前完整tensor合同校验通过，producer来源/冻结/坐标描述存在并已交叉核对。二者分别记录：B并未重新运行A的在线主干/冻结实验。

来源官方2views，真实 `official Uni3R scannet_test/scene0686_01`，frame顺序000000.jpg、000020.jpg；用途 **M0_DIAGNOSTIC_ONLY_NOT_FORMAL_TRAIN_OR_VAL**。四层4/11/17/23、patch-only `[1,2,256,2048]`、RGB[-1,1]、所有浮点fp32，无casting/改metadata/重排/缩小。原图386×512，完整操作链为初始crop→339×256 resize→[42,0,298,256] final crop；metadata的crop_xyxy只表示第一步，不能省略operations。
共享预测XYZ为原模型坐标/单位，未作米制标定/GT pose对齐；geometry score为1+exp(raw)，非校准概率。Gaussian为A.raw_views.v1，保留原始两视角属性，source_indices=arange(131072)。B核对属性形状、Gaussian means与输入means一致、全量view/y/x索引对应。原旋转/covariance/renderer约定差异沿用A说明，没有静默修正。

另建**未经过synthetic优化**的默认学生（Q100/decoder6/256×256），官方DPT初始化、其他头未训练；eval/no_grad，执行1次真实forward，耗时 2.161019906s。没有threshold或分割指标。

| raw输出 | shape | dtype | finite | min | max |
|---|---|---|---|---:|---:|
| mask_logits | [1, 100, 2, 256, 256] | torch.float32 | True | -0.2924506664 | 0.3207592368 |
| objectness_logits | [1, 100] | torch.float32 | True | -0.3320839405 | -0.198406741 |
| query_features | [1, 100, 256] | torch.float32 | True | -3.252790928 | 2.834710598 |

输出摘要/hash、37个输入tensor指纹及metadata见m0_probe。输入与Gaussian payload前后不变，模型参数/buffer前后不变，全量flatten/source_indices检查通过；结束再次核对fixture文件hash未变。这只证明该缓存消费路径，不重新证明在线Uni3R重建全路径不变。

## 8. C条件分支与独立状态

当前检查instance_C/C_PROGRESS.md、README.md、handoff.json及共享目录/已知报告位置C_PROGRESS；**本次限定检查未发现C交接或合同格式GT targets**，不推论队友尚未完成。
缺项为同一A样本的InstanceTargets、scene/frame/像素网格对应、跨视角物理ID声明、valid/confidence/ignore规则、完整同步crop/resize记录、来源/split/诊断用途。完整性依据若缺省须False，不能凭GT标签推定完整。
共享validate_training_batch没有独立target crop provenance字段，仅函数PASS不能确认标签物理裁剪；B没有补造标签、解码原始数据或修改合同。因此 **C_TARGET_ADMISSION=BLOCKED；REAL_GT_LOSS_FORWARD=BLOCKED**，本轮真实loss前向未执行，真实backward/step始终0。原合同测试中的1项历史fixture占位SKIP保留，旧“A未交付”文字不代表本轮A探测状态。

## 9. 实测回归、资源与失败/警告

- 原B-00/B-01/B-02回归：76项，75 PASS、1历史SKIP、0失败/错误，7.543509s。旧测试未改；未运行run_b01_checks的optimizer诊断，旧state_dict roundtrip留在内存。
- 新训练测试：8 PASS、0失败/错误，9.765364s。
- 新checkpoint测试：4 PASS、0失败/错误，5.918828s。
- 合计88项发现、87 PASS、1 SKIP；预算5次调用全部成功；恢复和A真实前向另保存独立验收证据。
- 本轮未发生测试或模型执行失败。故障注入的合同/容量/非有限/checkpoint拒绝均是预期负面检查，原错误文本和逐项结果保留。TypedStorage弃用warning来自原测试/现有PyTorch路径，保留console，未升级依赖。
- clip后float64复核范数的约1e-6浮点偏差已在§3披露。合并batch与micro调用的浮点归约顺序造成约1e-8标量和1e-6以下梯度差，均在预设容差内，未改模式/结构或放宽容差。
- 大型检查顺序执行，释放模型后再做下一项。整个runner峰值RSS=2139336 KiB（约2.0402 GiB），是进程RSS而非算子专属峰值，也不将共享机器内存视为B独占配额。各阶段RSS保存在各结果文件。无GPU、AMP或默认完整学生反向。

## 10. 收尾边界与审核停止

33个保护小文件SHA256与开工一致，53份B-01/B-02历史产物size/mtime一致，其中小文件hash一致；DPT大初始化未重复全文件哈希，使用原加载器单次验证和内存源指纹复核。A fixture文件hash与准入时一致。所有本轮远程写入路径实际resolve在instance_B，旧产物未覆盖。HEAD/branch已复核，Git短状态与开工相同=True；它不是全仓无并发变化证明，另有明确写入清单和文件hash，不回滚队友工作区。

结果根：`/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03`；详细运行：`/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03/attempt_01`。独立checkpoint预算、逐项stdout/结果、完整配置/样本清单、梯度/累积、跳步、更新锚点、错误加载、严格恢复、A/C准入、真实raw输出摘要、源码manifest、boundary_audit和completion均保留。本机只回传新的b03_local小文件快照与白名单/SHA，不覆盖b02_local、不下载诊断checkpoint或A真实fixture。本轮未获得本机/远程commit/push授权，均未执行。

REAL_TRAINING=NOT_RUN；DEFAULT_FULL_STUDENT_BACKWARD=NOT_RUN；GPU_USED=NO；AMP=NOT_RUN；PSEUDO_DISTILLATION=NOT_SUPPORTED；DEPLOYABLE_STUDENT_CHECKPOINT=NOT_PRODUCED。

主控仍需审核：B引擎/恢复实现及当前证据，C缺项和完整性/crop证据，未来真实训练数据的正式split/用途，以及另行阶段授权。真实test诊断fixture不能转为正式训练样本；本轮synthetic checkpoint不能成为GT/pseudo实验初始化。停止等待审核，不进入B-04或真实过拟合。
