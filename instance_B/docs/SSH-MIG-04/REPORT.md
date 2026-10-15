# SSH-MIG-04：B-03 既有训练工程 CPU 迁移回归

本轮完成 CPU 迁移回归；M0 三方真实联调仍受 C 交付阻塞。没有重新实现训练器，没有真实 GT 训练或 GPU 计算。

## 输出状态

| 字段 | 本轮实际结果 |
|---|---|
| MIG04_STATUS | CPU_MIGRATION_PASS_M0_BLOCKED |
| B03_EXISTING_MODULES | training.py, checkpointing.py, b03_utils.py and existing tests verified; no reimplementation |
| B03_CPU_TEST_RESULTS | 12/12 PASS (training 8 + checkpoint 4); B02 75 PASS/1 SKIP/0 FAIL; combined 88 collected/87 PASS/1 SKIP/0 FAIL |
| CHECKPOINT_RECOVERY_STATUS | PASS new-environment save/restore and W1-I-W2 continuation exact; legacy checkpoint rejected by exact-environment guard |
| A_REAL_FEATURE_STATUS | PASS current read/hash/schema; prior real student forward remains historical evidence only |
| C_REAL_TARGET_STATUS | BLOCKED: no usable targets found at the 5 known handoff paths |
| M0_READINESS | BLOCKED_BY_C_REAL_TARGET_HANDOFF |
| BLOCKERS | ['C contract-format real InstanceTargets and matching provenance are unavailable at known entry points', 'Legacy checkpoint exact resume is blocked by Python/build environment mismatch; guard unchanged', 'Persistent synthetic diagnostic optimizer budget exhausted: 8/8; no further step authorized by this run'] |
| FILE_CHANGES | ['NEW instance_B/docs/SSH-MIG-04/**', 'NEW instance_B/results/B-03/SSH-MIG-04/attempt_01/**', 'APPEND ONLY instance_B/results/B-03/20261008T140224Z_cpu_b03/optimizer_budget.jsonl (calls 6-8; old bytes preserved)'] |
| GPU_USED | NO |
| NEXT_RECOMMENDATION | STOP for ChatGPT review. Obtain qualified C handoff before M0; separately review legacy resume policy and any future optimizer budget. |

## 第一阶段：历史与实现核对

11 个 B-03 历史源码/配置/测试/文档哈希全部匹配保存清单。训练器、优化器参数组、有效场景 M/O 归一化、无监督跳步、严格 checkpoint 接口均已实现；没有改写接口、配置或原测试断言。

历史 A 前向：instance_B/results/B-03/20261008T140224Z_cpu_b03/attempt_01/m0_probe/real_student_forward.json。历史输入来源：/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_A/results/M0_20261008_01/fixture.pt。旧报告结论只是该诊断缓存的学生前向通过，不是在线 Uni3R 全流程或真实分割精度。

## 本轮环境与测试

沿用 SSH-MIG-03 的 Python 3.10.21 / Torch 2.1.0+cu121 / NumPy 1.26.4 / SciPy 1.15.3，已比对 18 个包版本完全相同；未安装或升级依赖。CUDA_VISIBLE_DEVICES 为空、字节码关闭、CPU math threads=2、interop=1、确定性算法开启。临时目录和 XDG 缓存在本报告目录内；其他缓存保持 B 内隔离。

| 测试集合 | 收集 | PASS | SKIP | FAIL/ERROR | 与历史 ID 集合 |
|---|---:|---:|---:|---:|---|
| regression | 76 | 75 | 1 | 0 | 完全一致 |
| training_tests | 8 | 8 | 0 | 0 | 完全一致 |
| checkpoint_tests | 4 | 4 | 0 | 0 | 完全一致 |

## 预算与 checkpoint 恢复范围

旧账本已用 5/8 次，不创建第二份账本、不清零、不修改固定上限。当前用户授权必要的小规模 synthetic CPU step，本轮复用原 W1/I/W2 调度片段：CONTINUOUS_W1、CONTINUOUS_W2、RESUMED_W2，共 3 次真实 optimizer 调用，历史账本仅追加调用 6、7、8，全部成功，累计 8/8。原 5 调用 W1/I/W2/W3 完整诊断未重跑，W3 本轮未执行；不能把片段恢复验证写成原完整 5 调用诊断重跑。原 8+4 项 unittest 本身全部原样执行。

新环境生成诊断 checkpoint 后，严格恢复并继续 W2：学生参数/buffer、AdamW tensor/非 tensor 状态、LR、计数器、游标、CPU RNG/Generator RNG 完全一致；保存时输出及最终三类 raw 输出误差均为 0。连续与恢复路径逻辑 update_step 均为 2。无有效监督 I 在已有 AdamW moments 后不前向、不 step，参数/优化器/LR/RNG 均保持不变。

诊断 checkpoint：/mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/results/B-03/SSH-MIG-04/attempt_01/synthetic_migration_W1.pt；417403412 bytes；SHA256 eb0d8ed9da49fb7d3a71a6948baa1afc1fff35147993d85be946418b1210d7df。用途保持 SYNTHETIC_ENGINE_DIAGNOSTIC_NOT_FOR_REAL_TRAINING，不能作为正式 GT/pseudo 训练初始化或部署模型。

旧 checkpoint 仍可只读安全加载并核对回执，但原 validate_payload 在 Python 3.10.22/GCC 与当前 3.10.21/Clang 的精确环境比较处报 Exact resume environment mismatch。本轮记录 EXPECTED_STRICT_REJECTION，不绕过、不放宽、不将旧 checkpoint 恢复进新引擎。新环境内恢复通过不等于旧 checkpoint 跨环境续训获准。

4 项 checkpoint 测试包含 24 种错误 payload，以及错误 SHA、截断、安全加载、拒绝覆盖、中途保存拒绝等原检查。

本轮四场景累积 gradient_max_abs_error=7.897615432739258e-07；tail-window=7.450580596923828e-07；使用原容差，无放宽。

## A / C 与 M0 状态

A 当前实际 weights_only=True/map_location=cpu 读取缓存，fixture/metadata/source_manifest 哈希均与 A producer report 一致，共享 FrozenSceneBatch 合同校验通过。fixture SHA256=351888091f93f3c6c2d651e08ef73aa593894a5152e773c3643a38eff9d85f54。未调用完整 Uni3R、未做新的真实学生前向、未跟随 metadata 中的项目外源数据路径、未渲染 Gaussian。

C 当前按原检查器核对 5 个已知交接入口，均未发现合格交付；这是限定入口下的结论，不推论队友其他位置不存在工作。仍需要同一 A 样本的真实 InstanceTargets、scene/frame/grid 对应、跨视角物理 ID、valid/confidence/ignore 规则、同步 crop/resize、来源/split/诊断用途及完整性依据。没有伪造真实 fixture。真实 GT loss/backward/step 本轮均未运行。

## 文件边界与 Git

200 个既有 B 文件/证据的记录项未变；242 个共享/跟踪文件哈希与 4 个 A 产物哈希未变。唯一既有文件变化是持久 optimizer 账本追加 9 行事件，原字节前缀完全保留。B_PROGRESS、历史报告、MIG03 文档、代码、测试和接口未修改。

分支 main、HEAD 7a7f80205fb075228ff75aac43635b96de901110、Git 短状态列表前后一致；工作区原来已有修改/未跟踪文件，未清理。新结果位于原未跟踪 B 目录，短状态列表相同不代表无新文件。无 commit/push、切分支、依赖安装、GPU 使用。

## 原始证据与完整测试记录

本目录：console.txt、environment.json、A_cache_current.json、C_current.json、legacy_checkpoint_compatibility.json、checkpoint_recovery.json、skip_after_moments.json、test_summary.json、audit_before.json、audit_after.json、ledger_before.jsonl、final_status.json。原 runner 的 stdout、逐项结果和细分证据位于 instance_B/results/B-03/SSH-MIG-04/attempt_01/。TypedStorage 弃用警告原样保留。

| 序号 | 原测试 ID | 结果 | 原因 |
|---:|---|---|---|
| 1 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_confidence_range_and_fractional_weights` | PASS |  |
| 2 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_default_target_dimensions_q100` | PASS |  |
| 3 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_detached_inputs_required` | PASS |  |
| 4 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_empty_background_and_all_ignore_scenes` | PASS |  |
| 5 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_frame_scene_and_batch_alignment` | PASS |  |
| 6 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_gaussian_filter_duplicates_dtype_and_empty` | PASS |  |
| 7 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_gaussian_filter_mapping` | PASS |  |
| 8 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_geometry_confidence_is_not_label_probability` | PASS |  |
| 9 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_id_rules_background_and_unknown` | PASS |  |
| 10 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_ignored_nan_not_hidden` | PASS |  |
| 11 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_input_shapes_rejected` | PASS |  |
| 12 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_invalid_pixel_must_have_zero_weight` | PASS |  |
| 13 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_invalid_positive_and_zero_weight_ids_omitted` | PASS |  |
| 14 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_logits_allow_negative_and_greater_than_one_and_grad` | PASS |  |
| 15 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_metadata_crop_and_normalization` | PASS |  |
| 16 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_metadata_dtype_and_special_tokens` | PASS |  |
| 17 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_missing_metadata_precise_errors` | PASS |  |
| 18 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_nan_inf_rejected` | PASS |  |
| 19 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_no_input_mutation` | PASS |  |
| 20 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_opaque_gaussian_payload_finiteness` | PASS |  |
| 21 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_prediction_nan_and_integer_dtype` | PASS |  |
| 22 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_prediction_shapes_and_flexible_query_count` | PASS |  |
| 23 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_real_provenance_validation_negative_only` | PASS |  |
| 24 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_required_dtypes` | PASS |  |
| 25 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_scene_local_ids_not_batch_global` | PASS |  |
| 26 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_scene_union_and_single_view_visibility` | PASS |  |
| 27 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_special_token_shape_and_layer_keys` | PASS |  |
| 28 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_synthetic_cannot_claim_real_checkpoint` | PASS |  |
| 29 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_unknown_cannot_be_valid` | PASS |  |
| 30 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_valid_small_batch` | PASS |  |
| 31 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_version_and_status` | PASS |  |
| 32 | `instance_B.tests.test_contracts.REAL_UNI3R_FIXTURE_TEST.test_real_fixture_not_available` | SKIP | NOT RUN: A has not supplied a real feature fixture inside the authorized root; replace this placeholder when integration is authorized |
| 33 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_a_dpt_init_complete_and_independent_storage` | PASS |  |
| 34 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_b_DPT_ADAPTER_SYNTHETIC_PARITY` | PASS |  |
| 35 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_c_shapes_dtype_finite_b2` | PASS |  |
| 36 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_d_variable_q` | PASS |  |
| 37 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_e_scene_independence` | PASS |  |
| 38 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_f_backward_all_active_student_parameters` | PASS |  |
| 39 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_g_detach_in_adapter_and_reject_attached_contract` | PASS |  |
| 40 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_h_identical_xyz_confidence_scores` | PASS |  |
| 41 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_i_xyz_normalization_near_zero_and_no_batch_mix` | PASS |  |
| 42 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_j_hand_checked_flatten_and_filter` | PASS |  |
| 43 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_k_mask_dot_scale_raw_logits` | PASS |  |
| 44 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_l_eval_state_dict_save_reload` | PASS |  |
| 45 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_m_missing_initialization_fails_closed` | PASS |  |
| 46 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_n_reject_missing_or_wrong_dpt_weights` | PASS |  |
| 47 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_o_tokens_order_extra_special_and_rgb` | PASS |  |
| 48 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_p_default_config_and_memory_bound` | PASS |  |
| 49 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_q_no_gt_teacher_dependency` | PASS |  |
| 50 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_r_static_parser_rejects_executable_global` | PASS |  |
| 51 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_s_attention_operates_only_on_queries` | PASS |  |
| 52 | `instance_B.tests.test_matcher.MATCHER_TEST.test_01_global_optimum_not_greedy` | PASS |  |
| 53 | `instance_B.tests.test_matcher.MATCHER_TEST.test_02_rectangular_exhaustive` | PASS |  |
| 54 | `instance_B.tests.test_matcher.MATCHER_TEST.test_03_dense_matrices_and_chunks` | PASS |  |
| 55 | `instance_B.tests.test_matcher.MATCHER_TEST.test_04_unique_id_and_query_equivariance` | PASS |  |
| 56 | `instance_B.tests.test_matcher.MATCHER_TEST.test_05_single_view_object_and_zero_support` | PASS |  |
| 57 | `instance_B.tests.test_matcher.MATCHER_TEST.test_06_capacity_context_and_k_equals_q` | PASS |  |
| 58 | `instance_B.tests.test_matcher.MATCHER_TEST.test_07_label_source_rejection` | PASS |  |
| 59 | `instance_B.tests.test_matcher.MATCHER_TEST.test_08_invalid_predictions_and_targets` | PASS |  |
| 60 | `instance_B.tests.test_matcher.MATCHER_TEST.test_09_empty_solver_and_match` | PASS |  |
| 61 | `instance_B.tests.test_criterion.CRITERION_TEST.test_01_hand_computed` | PASS |  |
| 62 | `instance_B.tests.test_criterion.CRITERION_TEST.test_02_empty_exhaustive_background` | PASS |  |
| 63 | `instance_B.tests.test_criterion.CRITERION_TEST.test_03_confidence_relative_once` | PASS |  |
| 64 | `instance_B.tests.test_criterion.CRITERION_TEST.test_04_ignore_invalid_zero_confidence_gradients` | PASS |  |
| 65 | `instance_B.tests.test_criterion.CRITERION_TEST.test_05_cross_view_id_switch_penalty` | PASS |  |
| 66 | `instance_B.tests.test_criterion.CRITERION_TEST.test_06_global_id_and_view_permutation_invariance` | PASS |  |
| 67 | `instance_B.tests.test_criterion.CRITERION_TEST.test_07_exhaustiveness_branches` | PASS |  |
| 68 | `instance_B.tests.test_criterion.CRITERION_TEST.test_08_default_partial_and_bool_validation` | PASS |  |
| 69 | `instance_B.tests.test_criterion.CRITERION_TEST.test_09_batch_isolation_and_permutation` | PASS |  |
| 70 | `instance_B.tests.test_criterion.CRITERION_TEST.test_10_ignore_append_and_duplicate_do_not_dilute` | PASS |  |
| 71 | `instance_B.tests.test_criterion.CRITERION_TEST.test_11_extreme_logits_and_tiny_positive_confidence` | PASS |  |
| 72 | `instance_B.tests.test_criterion.CRITERION_TEST.test_12_dense_loss_and_gradient_parity` | PASS |  |
| 73 | `instance_B.tests.test_criterion.CRITERION_TEST.test_13_fixed_assignment_gradcheck` | PASS |  |
| 74 | `instance_B.tests.test_criterion.CRITERION_TEST.test_14_single_view_visibility_negative_other_view` | PASS |  |
| 75 | `instance_B.tests.test_b02_integration.B02_INTEGRATION_TEST.test_01_student_small_backward` | PASS |  |
| 76 | `instance_B.tests.test_b02_integration.B02_INTEGRATION_TEST.test_02_contract_alignment_rejects_mismatch` | PASS |  |
| 77 | `instance_B.tests.test_training.TRAINING_TEST.test_01_parameter_groups` | PASS |  |
| 78 | `instance_B.tests.test_training.TRAINING_TEST.test_02_four_scene_accumulation_reference` | PASS |  |
| 79 | `instance_B.tests.test_training.TRAINING_TEST.test_03_tail_window_and_background_denominator` | PASS |  |
| 80 | `instance_B.tests.test_training.TRAINING_TEST.test_04_append_ignore_not_diluted` | PASS |  |
| 81 | `instance_B.tests.test_training.TRAINING_TEST.test_05_unsupervised_before_forward` | PASS |  |
| 82 | `instance_B.tests.test_training.TRAINING_TEST.test_06_contract_capacity_order_and_pseudo_errors` | PASS |  |
| 83 | `instance_B.tests.test_training.TRAINING_TEST.test_07_nonfinite_loss_and_gradient_no_step` | PASS |  |
| 84 | `instance_B.tests.test_training.TRAINING_TEST.test_08_gating_and_numerical_zero_not_skip` | PASS |  |
| 85 | `instance_B.tests.test_checkpointing.CHECKPOINT_TEST.test_01_payload_complete_and_purpose_isolated` | PASS |  |
| 86 | `instance_B.tests.test_checkpointing.CHECKPOINT_TEST.test_02_invalid_payloads_do_not_mutate_live_engine` | PASS |  |
| 87 | `instance_B.tests.test_checkpointing.CHECKPOINT_TEST.test_03_truncated_and_bad_receipt` | PASS |  |
| 88 | `instance_B.tests.test_checkpointing.CHECKPOINT_TEST.test_04_no_overwrite_and_boundary_only` | PASS |  |

唯一 SKIP 是原 B-02 真实 fixture 占位项；其旧原因文字不代表本轮 A 缓存不可用。

## 停止

CPU 迁移目标已完成。等待 ChatGPT 审核；M0 需 C 真实目标先交付并准入，旧 checkpoint 跨环境恢复与后续 optimizer 预算必须另行决策。本轮不继续训练或新阶段实验。
