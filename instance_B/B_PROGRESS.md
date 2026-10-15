# B 当前状态（B-03，2026-10-08）

- WORK_PACKAGE=B-03_TRAINING_ENGINE_CHECKPOINT_AND_M0_PROBE
- STATUS=B03_PARTIAL_AWAITING_REVIEW
- B独立部分完成；A真实缓存准入/默认学生eval前向PASS；C targets及真实GT loss前向BLOCKED，当前限定检查未发现合格交接。
- 回归75 PASS+1历史SKIP；训练测试8 PASS；checkpoint测试4 PASS；恢复后完整学生/AdamW/计数/游标/RNG及raw输出精确一致。
- 物理optimizer调用5/8；连续/恢复global_update_step=3/3；真实数据step=0。
- HEAD=7a7f80205fb075228ff75aac43635b96de901110; branch=main。GitHub被审核快照25f048e93f19ba126338fd509d179437b93c8e71与服务器HEAD分开记录。
- 报告：reports/B-03_REVIEW.md；结果：/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03
- checkpoint仅SYNTHETIC_ENGINE_DIAGNOSTIC_NOT_FOR_REAL_TRAINING，禁止用于真实GT/pseudo初始化。
- STOP：等待主控审核，不进入B-04；真实训练、默认完整学生反向、GPU、AMP未运行；未commit/push。

以下保留旧阶段进度及审核历史，不改写旧AWAITING_REVIEW或旧实验事实。

# B 当前状态（B-02，2026-10-08）

- WORK_PACKAGE=B-02_MATCHING_AND_LOSS
- STATUS=B02_COMPLETE_AWAITING_REVIEW
- HEAD=7a7f80205fb075228ff75aac43635b96de901110; branch=main
- matcher 9 PASS；criterion 14 PASS；dense parity PASS；学生 synthetic loss backward PASS（123 参数张量）；正式形状 logits-only 前向 PASS。
- 原合同31 PASS；原学生19 PASS；1历史真实 fixture 占位 SKIP（A已交付，本轮真实集成未运行）。A/C代码未修改或运行。
- 当前依赖：A M0 fixture 已有交付/待核对；C合同 targets 本次限定检查未发现；真实完整性/crop说明待交接。
- 报告：reports/B-02_REVIEW.md；结果：/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-02/20261008T130554Z_cpu_b02
- OPTIMIZER_STEPS=0；GPU_USED=NO；REAL_INTEGRATION/REAL_TRAINING/DEFAULT_FULL_STUDENT_BACKWARD/AMP=NOT_RUN；PSEUDO_DISTILLATION=NOT_SUPPORTED。
- STOP：等待用户交主控审核，不进入 B-03。

以下保留 B-00/B-01 历史状态及本轮追加日志；历史 AWAITING_REVIEW 不覆写。

# B｜Instance Student Research Progress

## Project Overview
- Goal: 冻结官方 2views Uni3R，使用真实导出特征训练实例学生
- Scope: B 负责 Instance DPT、adapter、queries、decoder、matching、loss、训练与 checkpoint
- Owner: B
- Shared Contract: `uni3r/instance/contracts.py`
- Current Stage: B-01
- Last Updated: 2026-10-08 (Asia/Shanghai)

## Current Status
- Work Package: B-01
- Status: B01_COMPLETE_AWAITING_REVIEW; IMPLEMENTED; TESTED_SYNTHETIC; DPT_INIT_VERIFIED; REAL_INTEGRATION_NOT_RUN; WAITING_FOR_A; WAITING_FOR_C
- Git Commit: `4a5dd00b6737a95afd4c6b1760cad166cf17d3d4`, branch `main`, pre-existing dirty worktree
- Environment: `/root/miniconda3/envs/uni3r/bin/python`; Python 3.10.22; torch 2.1.0+cu121; CUDA runtime 12.1
- Contract Version: 0.1.0; PROPOSED_FOR_TEAM_INTEGRATION (A/C not yet confirmed)

## Completed Milestones
| ID | Date | Work | Result | Evidence |
|---|---|---|---|---|
| B-00.1 | 2026-10-08 | SSH/root/Git/package/source audit | PASS (read-only) | `uni3r/uni3r.py`, `uni3r/vggt_w_intr_emb.py`, `uni3r/dpt_head.py` |
| B-00.2 | 2026-10-08 | Checkpoint read/hash | AVAILABLE; 3/3 segments match existing manifest | `checkpoints/official_manifest.json`; SHA256 below |
| B-00.3 | 2026-10-08 | 创建五个交付文件并远程读回 | PASS | `uni3r/instance/contracts.py`; 本目录README/tests/reports |
| B-00.4 | 2026-10-08 | CPU合同测试 | 31 PASS, 1 real-fixture SKIP, 0 failures/errors | `reports/B-00_REVIEW.md` §5；本地 `test_run_01.txt` |
| B-00.5 | 2026-10-08 | 导入/语法/Git/文件边界审计 | PASS；仅新增5文件，既有文件0变化 | `reports/B-00_REVIEW.md` §5；本地 `audit.json` / `boundary_audit.json` |
| B-01.1 | 2026-10-08 | DPT结构适配/安全初始化 | DPT_INIT_VERIFIED; 60/60 keys, 32,950,528参数 | `results/B-01/20261008_cpu_01/dpt_init_receipt.json` |
| B-01.2 | 2026-10-08 | InstanceStudent与独立测试 | IMPLEMENTED; 31合同+19学生PASS; parity误差0 | `results/B-01/20261008_cpu_01/checks_01/unit_results.json` |
| B-01.3 | 2026-10-08 | 默认前向/梯度/两步诊断 | TESTED_SYNTHETIC; 123参数tensor梯度有限非零; 默认前向PASS | `checks_01/full_config_forward.json`、`optimizer_diagnostic.json`（同run目录） |
| B-01.4 | 2026-10-08 | 限定文件边界检查 | PASS; 9个保护文件hash一致; 自有写入hash/语法通过 | `results/B-01/20261008_cpu_01/boundary_audit.json` |

## Current Work
- B-01 DPT milestone: IMPLEMENTED; DPT_INIT_VERIFIED; 60/60 keys, 32,950,528 parameters; evidence `results/B-01/20261008_cpu_01/dpt_init_receipt.json`
- Objective: B自有InstanceStudent、原DPT结构适配与安全初始化、synthetic前向/反向/默认配置验证
- Completed: DPT真实结构/严格初始化、完整学生、31+19测试、默认完整前向、2步诊断、结果落盘、README/报告、限定边界检查
- Remaining: 本轮工程工作完成，等待审核；真实接入待A/C，B-02未实施

## Test Results
- B-01: TESTED_SYNTHETIC；31合同回归PASS + 19学生测试PASS，1真实fixture SKIP；合计51项/8.064s，exit=0
- B-01 default full forward: PASS；Q100/6layers/256×256，2,560 memory tokens；1.5655s CPU
- B-01 backward/diagnostic: 学生参与参数梯度有限且非零；inputs无grad；2次SGD诊断步已用，参考初始化权重不变；没有正式实例loss/训练
- B-01 evidence: `results/B-01/20261008_cpu_01/checks_01/`（命令/配置、原始测试输出、parity/梯度、默认前向、两步诊断和保存重载文件）
- B-00历史 Tests: 2026-10-08 首次运行32 tests in 0.353s，exit=0，OK (skipped=1)
- Real/Synthetic: B-01 CONTRACT_REGRESSION=31 PASS; STUDENT_SYNTHETIC_TESTS=19 PASS; REAL_UNI3R_FIXTURE_TEST=NOT RUN (1 explicit skip)
- Outcome: 独立学生和合同synthetic测试通过；真实模型集成未验证，无实例精度/真实训练指标

## Integration Dependencies
### Required from A
- 根目录内提供真实 2-view token fixture和全重建模型冻结证据；DPT初始化已由B提取并strict验证，可核对receipt
- 确认 RGB/intrinsics、坐标系/尺度、geometry confidence、Gaussian 导出 schema/过滤索引与完整缓存元数据
- 原构造器将 freeze_dust3r 设为 False，需 A 显式冻结所有重建参数
### Required from C
- 根目录内提供 GT InstanceTargets，确认 -1/0/>0、跨视角同物 ID、valid/confidence 与 scene/frame 顺序
- 合同格式的真实空场景、单视角缺失物体、ignore 案例与数据来源/划分证据

## Open Issues and Risks
- 四份指定团队文档未在授权目录扫描中找到；提示词不冒充文件内容
- `data -> ../data` 超出授权范围，未沿链接读取；A fixture、C targets 未找到
- Gaussian/坐标等字段仍需 A 人工确认；C 标签语义仍待人工确认
- REAL_INTEGRATION_NOT_RUN：未交付真实feature fixture/GT targets，真实张量/坐标语义未验证；DPT官方权重键匹配已在B-01解决
- B-01仅CPU float32验证；AMP/GPU、默认完整配置反向、正式matching/loss、真实训练和精度均NOT RUN
- B-00合同单元测试没有失败；real test的1项SKIP不计入PASS
- 本机 JSON 汇总首次失败于 Linux 大小写文件名冲突，原始基线完整保存，改为 case-sensitive Hashtable 解析

## Files and Artifacts
- B-00历史交付：共享contracts、本进度文件、README、tests/test_contracts.py、reports/B-00_REVIEW.md
- B-01新增：dpt_adapter.py、extract_dpt_init.py、student.py、configs/student_v0.json、tests/test_student.py、run_b01_checks.py、reports/B-01_REVIEW.md；更新README/B_PROGRESS；所有产物位于results/B-01/20261008_cpu_01/
- Base checkpoint: `checkpoints/output/2views.pth`, 11,913,939,138 bytes
- SHA256: `660aed13d6a8bccd078be35210f803d0f6d3b312f63dd468bbccc1c7f3aacf24`
- 本地审计目录: `D:\崇新学堂\大三上\开创2\codex_remote_3dgs_test\b00_local`
- B-00历史边界：36,318→36,323文件；只新增指定5文件；既有文件检查一致，见B-00报告
- B-01边界：所有写入仅instance_B；共享合同/旧测试/旧报告及关键源码9文件hash一致；HEAD/main和Git短状态与开工一致（instance_B本已untracked）；不声称全仓无并发变动
- B-01 FILES_MODIFIED（既有文件）= README.md, B_PROGRESS.md；原Uni3R与共享合同未修改

## Next Actions
- STOP：B-01完成等待审核，不自动进入B-02或真实训练，不再执行optimizer.step
- NEXT_STAGE=B-02_MATCHING_AND_LOSS（需后续明确任务）；真实集成仍需A/C输入

## Change Log
- 2026-10-08 B-00 开始：目标文件不存在；完成只读环境和源码核对后，首次建立本进度文件。测试尚未执行。
- 2026-10-08 B-00 关键进展：五文件创建后远程ls/readback确认，代码可导入，首次31项synthetic测试通过；真实fixture缺失的1项保持SKIP/NOT RUN。开始结束边界审计。
- 2026-10-08 B-00 完成：36,318份原文件元数据及1,554份内容哈希与基线一致；只新增要求的5文件，Git HEAD/branch和既有tracked diff未变。环境补查Conda 26.1.1、nvcc 12.8.93。报告补齐真实检查结果和B-01清单；状态改为B00_COMPLETE_AWAITING_REVIEW，真实集成仍BLOCKED，停在审核门。
- 2026-10-08 B-01 开工：用户已审核B-00并授权B-01独立实现；读回进度/报告/共享合同/旧测试/原DPT。HEAD仍4a5dd00，main；Python3.10.22/torch2.1.0+cu121。所有本轮远程写入限制instance_B；共享合同/旧测试/原源码只读。checkpoint ZIP静态检查含Namespace和tensor描述，计划不执行pickle GLOBAL，仅选择DPT存储。CPU限2线程，未分配GPU，不占卡。
- 2026-10-08 B-01 DPT完成：B自有PatchOnlyDPT仅本实例索引改为[0,1,2,3]/patch_start_idx=0，保留真实DPT和input_identity。静态解析checkpoint后读取131,802,112字节DPT存储；严格加载60个float32张量/32,950,528参数，0 missing/unexpected；小权重safe weights_only读回一致。DPT_INIT_VERIFIED。产物是初始化，不是训练成果；适配数值parity仍待测试。
- 2026-10-08 B-01 前向/反向完成：首次51项测试0失败0错误，31合同+19学生PASS，真实fixture1项SKIP；synthetic DPT parity、存储独立、全部参与参数反向、场景隔离、手工索引与state_dict重载通过。默认完整前向CPU通过（39,532,161参数，2,560 memory tokens）。仅执行2步诊断SGD，学生参数更新而参考不变，未执行实例监督训练。开始文档/边界收尾。
- 2026-10-08 B-01 完成：README和审核报告/命令/原始输出均落盘；限定9个保护文件hash一致，自有源文件语法/回执hash检查通过。DPT parity和eval重载最大误差0，CPU峰值RSS约1.43GiB。状态B01_COMPLETE_AWAITING_REVIEW；DPT_INIT_VERIFIED；REAL_INTEGRATION_NOT_RUN，等待A/C真实交接。停止，不进入B-02。


## B-02 review handoff and start (2026-10-08)
- 据 user handoff: ChatGPT approved B-00/B-01 report-level stage review; this does not mean independent server-source reading or real integration acceptance.
- Status: B02_IN_PROGRESS; scope: scene-global exact matching, weighted BCE/Dice/objectness, CPU synthetic forward/backward only; optimizer steps=0.
- Current task: implement target preparation/matcher. Results: /mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-02/20261008T130554Z_cpu_b02
- Actual HEAD 7a7f80205fb075228ff75aac43635b96de901110; main; Python 3.10.22; torch 2.1.0+cu121; scipy 1.15.3. Contract remains 0.1.0.
- A has delivered a real M0 fixture (not loaded); C targets not found in this bounded check. REAL_INTEGRATION=NOT_RUN.
- Read-only audit completed; protected hashes recorded; historical results preserved. Commands: designated OpenSSH, environment-limited python -B stdin audit; git with GIT_OPTIONAL_LOCKS=0.
- First environment command had a shell-quoting SyntaxError; stdin script retry succeeded, without writes. New files: results start_audit.json/dependencies.json. No checks yet marked PASS.

- B02_IN_PROGRESS: target preparation and exact scene-global matcher implemented; criterion and synthetic boundary tests authored. New files: matcher.py, criterion.py, loss_utils.py, configs/loss_v0.json, three B-02 test modules, run_b02_checks.py. Actual upload via deploy_b02.ps1 succeeded with readback SHA256 manifest. Unit checks next; no PASS claim yet. Dependencies unchanged; A code only read, not executed or edited.
- 据用户交接，ChatGPT 已批准 B-00/B-01 的报告级阶段审核；该批准不代表 ChatGPT 独立读取服务器源码或完成真实集成验收。

- B02_IN_PROGRESS: matcher/criterion boundary unit_01 completed: 23 tests PASS (9 matcher, 14 criterion), 0 failures/errors/skips, 0.847s. Hand formula, independent dense costs/gradients (chunks 1/7/4096; float32/64 inputs), 40 brute-force solver cases and fixed-pair gradcheck PASS. Actual command: python -B -m instance_B.run_b02_checks --mode unit --run-dir /mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-02/20261008T130554Z_cpu_b02/unit_01. Next: unchanged B-00/B-01 regression and actual small student loss backward; full-shape logits-only forward. Dependencies unchanged; no optimizer.

- B02_IN_PROGRESS: student linkage and regression completed. integration_01: 53 discovered, 52 PASS + 1 historical real-fixture placeholder SKIP; specifically 31 contract + 19 student + 2 B-02 integration PASS. Actual student instance loss backward reached all active parameter groups; no optimizer created. full_01: B1/Q100/V2/256x256/K100 logits-only forward PASS, 0.756s; cost working tensor envelope 49,003,008 bytes (<64 MiB), peak process RSS 732,712 KiB. Actual commands saved in results commands.txt. A now has a real fixture; old test skip wording is stale historical code retained read-only; real integration NOT_RUN. PyTorch TypedStorage deprecation warning retained. Next: final review report, file hashes and boundary audit; no B-03.

- B02_COMPLETE_AWAITING_REVIEW：正式收尾。实际完成9个新代码/配置/测试/报告文件、README/B_PROGRESS更新及本轮轻量证据。unit_01=23 PASS；integration_01=52 PASS+1历史SKIP；full_01=PASS；无测试失败。16个保护小文件SHA256一致，旧B结果size/mtime一致；HEAD/branch已复核。真实源位置、实际命令、读回清单见报告与boundary_audit。依赖：A已交付真实fixture待核对，C合同targets本次限定检查未发现。下一步只有主控审核；不进入B-03。


## B-03 start (2026-10-08)
- B02_ACCEPTED_CODE_AND_EVIDENCE_REVIEW: user handoff reports ChatGPT reviewed GitHub snapshot 25f048e93f19ba126338fd509d179437b93c8e71. This is static code plus submitted synthetic evidence review, not independent server source reading or rerun. Historical B-02 AWAITING_REVIEW is preserved.
- B03_IN_PROGRESS: training engine, strict diagnostic checkpoint and zero-training-step real M0 probe. Actual server HEAD 7a7f80205fb075228ff75aac43635b96de901110 / main; B-02 file SHA256 comparisons all match.
- Results: /mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03; physical optimizer.step stage limit=8; planned resume experiment uses 5 calls. Persistent checkpoint-independent ledger: optimizer_budget.jsonl.
- Current task: implement parameter groups, active-scene accumulation and skip/failure handling. No optimizer calls yet. A fixture delivery and safe loading format documented; C handoff not found in bounded current check. No commit/push; no B-04.

- B03_IN_PROGRESS: Training engine checks executed: regression=True, training_tests=True. Actual results /mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03/attempt_01. No optimizer diagnostic starts unless both suites pass.

- B03_IN_PROGRESS: Checkpoint strict payload/safe loading and rejection tests completed; W1 checkpoint saved; physical calls=1. Resume equivalence next. Evidence: /mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03/attempt_01

- B03_IN_PROGRESS: A/C probe executed separately from synthetic updates; result=PASS. C GT admission=BLOCKED. Real training remains NOT_RUN. Evidence /mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/20261008T140224Z_cpu_b03/attempt_01

- B03_DEPENDENCY_BLOCKED: C真实targets/crop/物理ID/用途交接在本次限定检查中未发现；B独立引擎与checkpoint检查完成，A缓存默认前向通过，但真实GT loss未运行。
- B03_PARTIAL_AWAITING_REVIEW: 收尾保护哈希33项、历史结果53项和A fixture文件hash核对通过。实际命令/88项测试/5次物理step/恢复输出零误差/RSS/白名单见本轮报告与结果；本机仅回传B-03轻量证据，不下载checkpoint，不发布GitHub。下一步仅主控审核。
