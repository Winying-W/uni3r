# SSH-MIG-03 环境恢复与 CPU 回归报告

结论：本轮实际执行原 B-02 的 76 项 CPU 测试，75 PASS、1 SKIP、0 FAIL；测试 ID 集合与历史记录逐项一致。历史结果未作为本轮结果使用。

## 实际状态

| 字段 | 结果 |
|---|---|
| SSH_MIG_03_STATUS | PASS_B02_CPU_REGRESSION |
| ENV_PATH | /mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/.venv |
| PYTHON_VERSION | 3.10.21 (main, Aug 25 2026, 14:01:09) [Clang 22.1.3 ] |
| TORCH_VERSION | 2.1.0+cu121 |
| CUDA_RUNTIME_VERSION | 12.1 |
| NUMPY_VERSION | 1.26.4 |
| SCIPY_VERSION | 1.15.3 |
| PYTEST_VERSION | 8.3.5 |
| DEPENDENCY_CHECK | PASS: uv pip check; 18 compatible packages |
| B02_TEST_COLLECTED | 76 |
| B02_PASS | 75 |
| B02_SKIP | 1 |
| B02_FAIL | 0 |
| GPU_USED | NO |
| SHARED_FILES_MODIFIED | NO |
| GIT_STATUS_BEFORE_AFTER | IDENTICAL existing dirty worktree; main 7a7f80205fb075228ff75aac43635b96de901110 |
| CPU_READINESS | B-02 original CPU regression passed |
| GPU_READINESS | NOT_VERIFIED |
| BLOCKERS | [] |
| REPORT_PATH | /mnt/workspace/Jensen/project/3DGS/Uni3R/instance_B/docs/SSH-MIG-03/REPORT.md |
| NEXT_RECOMMENDATION | STOP; await ChatGPT review. No GPU/training/A-C integration authorized. |

## 环境及范围差异

- Python 从旧 3.10.22 改为本轮 3.10.21（Clang 构建）；固定 uv 0.12.7 未提供 3.10.22，失败原文已保留。用户已允许 3.10.x 补丁替代。
- Torch 2.1.0+cu121、NumPy 1.26.4、SciPy 1.15.3 保持目标版本；新增固定 pytest 8.3.5 用于收集，执行仍复用原 unittest runner。
- 本轮官方 Torch wheel SHA256 为 0d4e8c52a1fcf5ed6cfc256d9a370fcf4360958fc79d0b08a51d55e70914df46；旧项目 lock 的本地 wheel SHA256 为 bf57f8184b2c317ef81fb33dc233ce4d850cd98ef3f4a38be59c7c1572d175db。未声称二进制相同，依据本轮版本核验与回归判断可用性。
- 共安装 18 个实际依赖；官方 Torch CUDA wheel 自带相关运行库，并声明 Triton 2.1.0 依赖。未安装 Torchvision、Torchaudio 或完整 Uni3R 科研依赖。
- GPU_USED=NO；CUDA_VISIBLE_DEVICES 为空，CPU 张量运算和 NumPy 互转通过，CUDA 未初始化。不加载完整 Uni3R、不渲染、不做 optimizer step、不创建训练 checkpoint。
- 仅按追加授权读取 B 内已有 DPT 初始化权重并运行原小型 synthetic CPU 回归。原状态字典往返使用 BytesIO，不写新 checkpoint。

## 按模块汇总

| 模块 | PASS | SKIP | FAIL/ERROR |
|---|---:|---:|---:|
| test_b02_integration | 2 | 0 | 0 |
| test_contracts | 31 | 1 | 0 |
| test_criterion | 14 | 0 | 0 |
| test_matcher | 9 | 0 | 0 |
| test_student | 19 | 0 | 0 |

## 全部 76 项实际结果

| 序号 | 原测试 ID | 结果 | 说明 |
|---:|---|---|---|
| 1 | `instance_B.tests.test_matcher.MATCHER_TEST.test_01_global_optimum_not_greedy` | PASS |  |
| 2 | `instance_B.tests.test_matcher.MATCHER_TEST.test_02_rectangular_exhaustive` | PASS |  |
| 3 | `instance_B.tests.test_matcher.MATCHER_TEST.test_03_dense_matrices_and_chunks` | PASS |  |
| 4 | `instance_B.tests.test_matcher.MATCHER_TEST.test_04_unique_id_and_query_equivariance` | PASS |  |
| 5 | `instance_B.tests.test_matcher.MATCHER_TEST.test_05_single_view_object_and_zero_support` | PASS |  |
| 6 | `instance_B.tests.test_matcher.MATCHER_TEST.test_06_capacity_context_and_k_equals_q` | PASS |  |
| 7 | `instance_B.tests.test_matcher.MATCHER_TEST.test_07_label_source_rejection` | PASS |  |
| 8 | `instance_B.tests.test_matcher.MATCHER_TEST.test_08_invalid_predictions_and_targets` | PASS |  |
| 9 | `instance_B.tests.test_matcher.MATCHER_TEST.test_09_empty_solver_and_match` | PASS |  |
| 10 | `instance_B.tests.test_criterion.CRITERION_TEST.test_01_hand_computed` | PASS |  |
| 11 | `instance_B.tests.test_criterion.CRITERION_TEST.test_02_empty_exhaustive_background` | PASS |  |
| 12 | `instance_B.tests.test_criterion.CRITERION_TEST.test_03_confidence_relative_once` | PASS |  |
| 13 | `instance_B.tests.test_criterion.CRITERION_TEST.test_04_ignore_invalid_zero_confidence_gradients` | PASS |  |
| 14 | `instance_B.tests.test_criterion.CRITERION_TEST.test_05_cross_view_id_switch_penalty` | PASS |  |
| 15 | `instance_B.tests.test_criterion.CRITERION_TEST.test_06_global_id_and_view_permutation_invariance` | PASS |  |
| 16 | `instance_B.tests.test_criterion.CRITERION_TEST.test_07_exhaustiveness_branches` | PASS |  |
| 17 | `instance_B.tests.test_criterion.CRITERION_TEST.test_08_default_partial_and_bool_validation` | PASS |  |
| 18 | `instance_B.tests.test_criterion.CRITERION_TEST.test_09_batch_isolation_and_permutation` | PASS |  |
| 19 | `instance_B.tests.test_criterion.CRITERION_TEST.test_10_ignore_append_and_duplicate_do_not_dilute` | PASS |  |
| 20 | `instance_B.tests.test_criterion.CRITERION_TEST.test_11_extreme_logits_and_tiny_positive_confidence` | PASS |  |
| 21 | `instance_B.tests.test_criterion.CRITERION_TEST.test_12_dense_loss_and_gradient_parity` | PASS |  |
| 22 | `instance_B.tests.test_criterion.CRITERION_TEST.test_13_fixed_assignment_gradcheck` | PASS |  |
| 23 | `instance_B.tests.test_criterion.CRITERION_TEST.test_14_single_view_visibility_negative_other_view` | PASS |  |
| 24 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_confidence_range_and_fractional_weights` | PASS |  |
| 25 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_default_target_dimensions_q100` | PASS |  |
| 26 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_detached_inputs_required` | PASS |  |
| 27 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_empty_background_and_all_ignore_scenes` | PASS |  |
| 28 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_frame_scene_and_batch_alignment` | PASS |  |
| 29 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_gaussian_filter_duplicates_dtype_and_empty` | PASS |  |
| 30 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_gaussian_filter_mapping` | PASS |  |
| 31 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_geometry_confidence_is_not_label_probability` | PASS |  |
| 32 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_id_rules_background_and_unknown` | PASS |  |
| 33 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_ignored_nan_not_hidden` | PASS |  |
| 34 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_input_shapes_rejected` | PASS |  |
| 35 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_invalid_pixel_must_have_zero_weight` | PASS |  |
| 36 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_invalid_positive_and_zero_weight_ids_omitted` | PASS |  |
| 37 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_logits_allow_negative_and_greater_than_one_and_grad` | PASS |  |
| 38 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_metadata_crop_and_normalization` | PASS |  |
| 39 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_metadata_dtype_and_special_tokens` | PASS |  |
| 40 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_missing_metadata_precise_errors` | PASS |  |
| 41 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_nan_inf_rejected` | PASS |  |
| 42 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_no_input_mutation` | PASS |  |
| 43 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_opaque_gaussian_payload_finiteness` | PASS |  |
| 44 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_prediction_nan_and_integer_dtype` | PASS |  |
| 45 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_prediction_shapes_and_flexible_query_count` | PASS |  |
| 46 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_real_provenance_validation_negative_only` | PASS |  |
| 47 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_required_dtypes` | PASS |  |
| 48 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_scene_local_ids_not_batch_global` | PASS |  |
| 49 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_scene_union_and_single_view_visibility` | PASS |  |
| 50 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_special_token_shape_and_layer_keys` | PASS |  |
| 51 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_synthetic_cannot_claim_real_checkpoint` | PASS |  |
| 52 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_unknown_cannot_be_valid` | PASS |  |
| 53 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_valid_small_batch` | PASS |  |
| 54 | `instance_B.tests.test_contracts.CONTRACT_SYNTHETIC_TEST.test_version_and_status` | PASS |  |
| 55 | `instance_B.tests.test_contracts.REAL_UNI3R_FIXTURE_TEST.test_real_fixture_not_available` | SKIP | NOT RUN: A has not supplied a real feature fixture inside the authorized root; replace this placeholder when integration is authorized |
| 56 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_a_dpt_init_complete_and_independent_storage` | PASS |  |
| 57 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_b_DPT_ADAPTER_SYNTHETIC_PARITY` | PASS |  |
| 58 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_c_shapes_dtype_finite_b2` | PASS |  |
| 59 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_d_variable_q` | PASS |  |
| 60 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_e_scene_independence` | PASS |  |
| 61 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_f_backward_all_active_student_parameters` | PASS |  |
| 62 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_g_detach_in_adapter_and_reject_attached_contract` | PASS |  |
| 63 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_h_identical_xyz_confidence_scores` | PASS |  |
| 64 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_i_xyz_normalization_near_zero_and_no_batch_mix` | PASS |  |
| 65 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_j_hand_checked_flatten_and_filter` | PASS |  |
| 66 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_k_mask_dot_scale_raw_logits` | PASS |  |
| 67 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_l_eval_state_dict_save_reload` | PASS |  |
| 68 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_m_missing_initialization_fails_closed` | PASS |  |
| 69 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_n_reject_missing_or_wrong_dpt_weights` | PASS |  |
| 70 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_o_tokens_order_extra_special_and_rgb` | PASS |  |
| 71 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_p_default_config_and_memory_bound` | PASS |  |
| 72 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_q_no_gt_teacher_dependency` | PASS |  |
| 73 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_r_static_parser_rejects_executable_global` | PASS |  |
| 74 | `instance_B.tests.test_student.STUDENT_SYNTHETIC_TEST.test_s_attention_operates_only_on_queries` | PASS |  |
| 75 | `instance_B.tests.test_b02_integration.B02_INTEGRATION_TEST.test_01_student_small_backward` | PASS |  |
| 76 | `instance_B.tests.test_b02_integration.B02_INTEGRATION_TEST.test_02_contract_alignment_rejects_mismatch` | PASS |  |

唯一 SKIP 是原测试中保留的真实 fixture 占位项；保留其原断言和原因文本，不将其解释为本轮对 A/C 当前交付状态的验证。Hungarian matching、mask BCE/Dice、objectness、ignore/confidence 逻辑及小型 student 梯度测试均包含在上述原集合中。

## 边界与 Git 审计

- 原 B 文件 132 个：原有大小与 mtime 均未变，已记录哈希的文件哈希均未变，包括只读 DPT 初始化权重。
- 受保护共享/跟踪文件 242 个哈希均未变；共享 contracts 另与 B-02 历史哈希一致。
- Git 分支、HEAD 与短状态列表前后一致；原工作区本来非干净，不能将状态相同解释成 clean。新增内容均在原未跟踪的 instance_B 内。
- B 自己的 .gitignore 已核验忽略 .venv/.runtime/.cache/.tmp/.tools；顶层 .gitignore 未改。无 commit/push。
- 安装前约 42 TiB 可用、inode 充足；quota 工具不可用，未宣称已验证用户配额。前后磁盘输出在审计 JSON 中。

## 问题与处理

- 官方源单连接下载缓慢/超时：保留失败与中止日志，改为官方 URL 分段下载，按官方 SHA256 校验完整文件后才使用。未执行下载的安装脚本。
- Windows 沙盒网络阻断后，用户批准沙盒外尝试；外部尝试也超时。本机未留下下载文件，未上传任何本机凭证。
- 中止的慢速下载仅针对本轮经 PID/命令核对的 uv 进程，没有操作其他进程或 GPU。
- 一次新 SSH 连接超时，一次安装会话连接重置；重新连接后核实远程安装日志及最终依赖一致性。未把 SSH 退出错误隐藏为安装成功。
- 网络盘安装大包较慢；环境最终通过版本、依赖和 CPU 回归检查。
- 报告生成命令曾因嵌套引号产生语法错误，该命令未执行写入；修正后生成报告，未重跑或修改测试。

## 审核文件与复现

- environment_versions.json / requirements-installed.txt：本轮实际版本清单。
- requirements.lock：完整带哈希依赖锁；requirements.in / constraints.txt：直接依赖和约束。
- *_source.json / python_candidates.json：固定版本和官方来源。
- commands.sh、isolation.sh、REBUILD.md：安装历史、启用方式、其他服务器重建步骤。
- pytest_collection.txt、unit_console.txt、integration_console.txt、regression_summary.json：收集和原始结果。
- 原 runner 结构化证据在 instance_B/results/B-02/SSH-MIG-03/unit/ 和 integration/；历史目录保持不变。
- audit_before.json、shared_before_regression.json、audit_after.json、artifact_sha256.json：边界审计与轻量证据哈希。

## 停止条件

B-02 CPU 回归已通过，无阻塞本轮 CPU 目标的问题。GPU_READINESS=NOT_VERIFIED；没有验证 GPU 训练恢复、完整 Uni3R 或 A/C 在线集成。到此停止，等待 ChatGPT 审核。
