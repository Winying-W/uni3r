# B-00｜共享合同与 B 工作区审核

日期：2026-10-08（Asia/Shanghai）。状态：**B00_COMPLETE_AWAITING_REVIEW**。合同synthetic测试与工作区初始化已完成；**REAL_INTEGRATION_BLOCKED**，等待A/C交接和用户审核。

## 1. 操作范围与既有材料

仅通过 Windows 系统 OpenSSH 连接 `010`。远程 pwd、realpath 和 Git root 均为 `/mnt/workspace/Jensen/project/3DGS/Uni3R`；用户 root，主机 `dsw-824375-57c745db88-n6tv9`。所有项目读取、写入限制在该根目录内。`data -> ../data` 未跟随，未读取外部数据；不运行服务器 Codex、不访问 `/root/.codex`、不复制认证信息、不安装依赖、不切换分支、不提交/推送、不执行训练。

现有 `uni3r/__init__.py` 为空，真实包名与计划一致；五个目标文件及 `uni3r/instance`、`instance_B` 目录均不存在，未发现 AGENTS.md。创建仅限五个要求文件，使用独占创建；后续只以 SHA256 compare-and-swap 更新本轮自己的文件。

未找到 `核心架构.md`、`INSTANCE_METHOD_REVIEW.md`、`INSTANCE_TEAM_PLAN.md`、`ORACLE_INSTANCE_RESULTS.md`。实际阅读的是根目录的 README、REPRODUCTION、AUTHOR_DATASETS（相关部分）、环境脚本和模型源码。REPRODUCTION 记载既有重建评测；AUTHOR_DATASETS 明确作者版训练数据尚未完成。那些记载不作为本轮实例训练结果，也未追踪其中根目录外的证据链接。本轮接口语义来源于用户授权和实际源码核对，不冒充已有团队文件或 A/C 的人工确认。

## 2. Git、环境与资源

- HEAD `4a5dd00b6737a95afd4c6b1760cad166cf17d3d4`；branch `main`；任务开始时已有 dirty worktree。
- 预存修改：`.gitignore`、`scripts/test.sh`、`scripts/test_16views.sh`、`scripts/test_8views.sh`、`scripts/train.sh`、`test.py`、`uni3r/loss.py`、`uni3r/lseg.py`、`uni3r/utils/path_manager.py`，以及 `submodules/diff-gaussian-rasterization`、`submodules/lang_seg` 的既有状态；另有复现文档、requirements、脚本等 untracked 文件。完整基线保存在本地审计目录。
- 默认 shell：`/usr/bin/python` 3.12.3；torch `2.7.0a0+7c8ec84dab.nv25.03`，CUDA runtime 12.8。
- 选用仓库 `scripts/env.sh` 指定的现有解释器 `/root/miniconda3/envs/uni3r/bin/python`：Python 3.10.22、torch 2.1.0+cu121、CUDA runtime 12.1；实际 torch 导入位置为该环境的 site-packages。Conda 未激活且不在默认 PATH；没有切换/改写共享环境。
- GPU 检查：8×NVIDIA A100-SXM4-80GB，driver 570.133.20，torch CUDA available=True。初次显存已用 MiB（0..7）：54403、32824、63228、50511、37545、52130、76015、75179，属于瞬时共享状态。单测屏蔽 GPU。
- 默认PATH未找到Conda/nvcc；通过现有程序绝对路径补查：Conda 26.1.1，`/usr/local/cuda/bin/nvcc` 为 CUDA 12.8.93。编译器版本与torch runtime 12.1分开记录；未编译扩展。
- `checkpoints/output/2views.pth` 可读，11,913,939,138 bytes；本轮流式计算 SHA256：`660aed13d6a8bccd078be35210f803d0f6d3b312f63dd468bbccc1c7f3aacf24`。
- 拼接文件的 aa/ab/ac 三段分别匹配根目录内既有 `checkpoints/official_manifest.json` 的 LFS 哈希，耗时 95.406 s。无网络下载、无在线重新认证官方清单、无 checkpoint 反序列化或真实模型 forward。
- A feature fixture：NOT FOUND in authorized root。C contract-format GT targets：NOT FOUND；不将原始处理脚本、重建测试指标或外部 `data` 链接当作实例 targets。

## 3. 模型与 DPT 源码核对（STATIC_SOURCE_EVIDENCE）

以下尺寸是源码推导及团队目标，**尚无真实 forward tensor 实证**。

| 内容 | 观察与证据 |
|---|---|
| 原入口 | `uni3r/uni3r.py:37` 的 Uni3R；`test.py:98` 使用 `from_pretrained(..., num_views_value=2)`；训练入口为 `train.py` |
| 主干与冻结 | `uni3r/uni3r.py:62` 构造 `VGGT_IntrEmb(patch_size=16,img_size=256)`；第 66 行把 `freeze_dust3r=False`，仅现有 distiller/LSeg 有相应冻结逻辑。不能默认 reconstruction 已冻结，A 必须显式 `eval`、禁用梯度并导出 detached 特征 |
| 原 DPT 初始化 | `uni3r/uni3r.py:68`：`DPTHead(dim_in=2048, feature_only=True, input_identity=True, patch_size=16)`；自定义 input_merger 是 3→256 的 7×7 conv+ReLU（`uni3r/dpt_head.py:19`）。构造时默认参数初始化；`from_pretrained` 第 199–211 行加载 `ckpt['model']`，strict=False。本轮未验证 2views 中 DPT keys 的完整匹配 |
| DPT 父类 | 实际定位至 `submodules/vggt/vggt/heads/dpt_head.py`，不是环境内另一个同名副本；默认层 `[4,11,17,23]`（zero-based）、feature dim 256、四层投影 `[256,512,1024,1024]`、resize ×4/×2/identity/stride2、feature_only 输出分支 |
| Token 排列 | `Aggregator_IntrEmb` 第 83 行将 intrinsic token 加在 patch 序列尾部；第 92 行在最前面添加 camera/register；第 129–132 行将 frame/global 各 1024 通道拼为 2048；24 层列表 |
| 特殊 token 移除 | `VGGT_IntrEmb.forward` 第 206–215 行：aggregator 后先 `[:,:,:-1,:]` 移除 intrinsic，再存 `outputs['tokens']`。父 aggregator 默认 4 register，加 1 camera，`patch_start_idx=5`（`aggregator.py:60,131`）；DPT 第 118 行取 `list[layer][:,:,patch_start_idx:]`。A 应从该 outputs 列表抽取四层并只移除前缀一次；不能再次删除 intrinsic 尾项 |
| 静态目标形状 | 原始 aggregator 预计 `[B,V,262,2048]`，包装后 261，移除 5 个前缀后 `[B,V,256,2048]`。16×16 patch grid 通过 `patch_embed.py:74` flatten，为 y-major/x-minor；本轮未实测这些中间值 |
| 导出位置 | `uni3r/uni3r.py:124–131` 得到 `outputs` 后可交接 `outputs['tokens']`。原 `Uni3R.forward` 最终返回按 view 的 Gaussian 列表，不会直接导出 FrozenSceneBatch；A 仍需提供导出适配 |
| RGB/内参 | Uni3R 将输入 `(images+1)/2` 给 VGGT，但 DPT 调用使用原始 images；DPT docstring 写 `[0,1]`，与实际调用存在文档差异。aggregator 又应用 ImageNet mean/std。内参前两行除以 H（当前方图）；A 必须说明缓存保存 pixel K 还是 normalized K，并说明 crop/resize 后变换 |
| 预测坐标 | `world_points` 来自 point head（`vggt_w_intr_emb.py:229–234`）；Gaussian VGGT 路径第 66 行直接 flatten 该预测值，未取 GT XYZ。物理坐标轴、尺度、第一相机/世界系最终含义仍待 A 交接确认，不能只凭变量名定论 |
| 几何分数 | point head 的 `conf_activation='expp1'`；`head_act.py:103–104` 是 `1+exp`。合同仅要求非负有限，不使用 label-confidence 的 `[0,1]` 限制 |

GaussianHead 真实原始输出是长度 V 的列表，每 view 包含 `scales`, `rotations`, `covs`, `opacities`, `sh_coeffs`, `means`, `gs_feats`（第 131–140 行）。Uni3R 还追加 depth/extr/intr 及 distiller 字段。源码预期每 view 为 `[B,H,W,...]`，其中 means/scales 尾维3、rotations尾维4、opacities尾维1、SH为16×3、gs_feats为64；covs布局/旋转分量顺序/相机 extr 约定不在本轮定为共享语义。源码使用 view-y-x flatten，但合同 `gaussians` 的映射容器还需 A 包装；没有伪造 A 已存在的导出实现。

## 4. 已提出的可执行合同

`CONTRACT_VERSION=0.1.0`；`CONTRACT_STATUS=PROPOSED_FOR_TEAM_INTEGRATION`。使用 TypedDict、frozen ContractSpec、独立 validation，依赖仅 stdlib+PyTorch；不导入 Uni3R 模型、renderer 或训练框架。`uni3r/instance` 使用现有包下的 namespace 子包。

入口：`validate_frozen_scene_batch`、`validate_instance_targets`、`validate_instance_prediction`、`validate_training_batch`；补充 `scene_instance_ids` 场景级目标并集。默认严格要求 V=2、256×256、四层 patch-only tokens/2048 通道。Q 从预测获取，不写死100。支持浮点16/bfloat16/32/64，实例 int64，valid bool；所有值包括 ignore 区都要求有限。Frozen/targets 不得带梯度；prediction 允许带梯度。函数不修改输入。

Targets：-1 ignore、0可信背景/stuff、正ID真实实例；invalid confidence必须0，-1不得valid=True；confidence为[0,1]监督权重。GT默认由C填1，显式重加权允许；不自动填充或更改。所有视角统一场景ID；无实例/全ignore得到空集合，物体只在一视角可见仍保留。未来空GT匹配为空分配，可信背景上可学习无物体；全无监督场景需loss显式跳过/返回安全零值，不把未知像素当负例。该loss行为仅B-01/B-02规范，本轮没有实现或测试loss。

每缓存 metadata 明确 checkpoint 名/SHA、commit、dirty/source fingerprint、尺寸/normalization、token层/顺序/移除方式、scene/frame顺序、crop/resize、intrinsics、坐标/means来源、geometry confidence语义、所有tensor dtype、Gaussian schema/flatten/filter。dirty源码的 `uni3r_source_sha256` 由A对实际参与推理的源码文件按相对路径排序，记录逐文件SHA256后对该清单UTF-8内容取SHA256，并关联原清单以供审计。字符串声明和格式通过不代表来源事实已被独立验证。

过滤需 `gaussians.source_indices`，长度B的int64向量列表，允许每场景保留数量不同及空向量，检查唯一性/范围。只过滤（不复制/分裂）对应一源像素一Gaussian。`index=view*H*W+y*W+x`；原payload与indices的一一对应仍需A给出schema后做真实集成检查。

验证边界：标签栅格无法自行辨认不同视角的同一物体，故同ID并集测试只检验合同算法，C仍需物理身份一致性证据；数值无法证明logits是否已sigmoid；requires_grad=False无法证明原模型所有参数已冻结。真实缓存必须用真实 provenance，synthetic明确无checkpoint。上述限制不得包装为真实集成通过。

## 5. 测试与审核结果

首次远程CPU测试完成，命令如下（从授权根目录执行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/uni3r/bin/python -B -m unittest discover -s instance_B/tests -v
```

实际结果：`Ran 32 tests in 0.353s`，`OK (skipped=1)`，exit=0。**31项 CONTRACT_SYNTHETIC_TEST PASS，0失败/0错误；1项 REAL_UNI3R_FIXTURE_TEST SKIP/NOT RUN。** 跳过原因原文：`NOT RUN: A has not supplied a real feature fixture inside the authorized root; replace this placeholder when integration is authorized`。

覆盖正确输入、shape/dtype、NaN/Inf、ignore/背景/正ID、valid/权重、跨视角ID并集、单视角可见、空背景/全ignore、B>1的场景ID隔离、raw logits/Q可变、metadata缺失/不一致、detached输入、Gaussian过滤源索引/空过滤、跨batch/scene/frame对齐和无输入修改。显式小尺寸profile外，还用默认256×256、256 patches、2048通道、Q=100做完整尺寸验证。没有真实Uni3R forward、DPT checkpoint加载、matching/loss训练或性能评价。

测试成功包含共享模块实际导入。独立读回确认 `contracts.__file__=/mnt/workspace/Jensen/project/3DGS/Uni3R/uni3r/instance/contracts.py`，版本/状态正确；两个Python文件AST解析通过。完整测试输出位于本地 `b00_local/test_run_01.txt`；远程报告保存命令、计数、跳过原因及可重跑入口。

结束边界审计结果：

- 文件总数36,318→36,323，只新增下面5个要求文件；无原文件删除、size/mtime变化；1,554份适用源码/文档内容SHA256全部一致。
- HEAD、branch、既有 `git diff --binary HEAD` SHA256及符号链接清单均一致；不跟随外部data链接。既有A/C dirty修改完整保留。
- `git diff --check` 对既有tracked工作区 exit=0；五个新增文件分别以 `git diff --no-index --check /dev/null <file>` 检查，均无空白诊断（exit=1表示与空文件有差异）。完整五文件diff保存在本地 `audit.json`，因为它们是untracked，普通git diff不会显示其内容。
- 所有远程写入都通过五文件白名单/路径检查/独占创建或本轮文件SHA256冲突检查；读回SHA256匹配本地交付内容。没有额外__init__.py、pycache、student/criterion/train脚本或远程测试日志文件。
- FILES_CREATED：`uni3r/instance/contracts.py`、`instance_B/B_PROGRESS.md`、`instance_B/README.md`、`instance_B/tests/test_contracts.py`、`instance_B/reports/B-00_REVIEW.md`。FILES_MODIFIED（相对任务开始的原文件）=NONE；本轮新建文档内的进度更新不算覆盖他人文件。

代码SHA256：contracts=`2810deaffde90952ce5f088d12ec112d24aed8c7f583c9498169261687ea072f`；tests=`67d3a768cc8dea6110a60f4bd3470e14d8c7c2215e4162ab08fa6a6347ea45e4`。最终文档自身hash见本地传输收据，避免报告自引用hash。

失败与限制：合同测试无失败；真实fixture缺失使真实集成NOT RUN，未规避。初始本地基线JSON汇总大小写解析问题及修复见§8。没有实测实例精度、训练loss或重建性能；未声称A/C已确认。

## 6. B-01 可执行清单（计划，未实施）

| 模块 | 下一轮实施内容 | Synthetic 可验证 | 真实依赖 |
|---|---|---|---|
| instance_dpt | B自有独立DPT；接四层dict，正确对齐zero-based层；支持纯patch tokens，避免二次去special tokens；复制原DPT初始化后独立训练 | shape、层映射、输入梯度隔离 | A的token fixture、RGB约定、实际原DPT state_dict/key/shape一致性；不能随机初始化后宣称已复制 |
| multi-scale adapter | 将多尺度feature投影到256维，下采样memory，记录各尺度/视角布局 | memory长度、dtype、mask/reshape | 真实feature尺度分布与显存实测 |
| detached XYZ positional encoding | 使用预测means.detach()，按对应尺度/视角下采样后编码到256维；保留尺度与有效性记录 | 有限性、排列、对XYZ及主干无反向梯度 | A确认世界/相机坐标、尺度、缺失/过滤规则 |
| shared object queries | 默认Q=100，hidden=256，同场景所有视角共用query | 多Q、B>1、视角共享、permutation | 真实目标可见性与ID对齐 |
| 6-layer instance decoder | 6层query self/cross-attention，使用降采样多尺度memory | 前向/反向、张量维度、内存上界初测 | 真实tokens及受控资源准入，不自动启动训练 |
| 64D mask features | 稠密/恢复分辨率的64维像素特征；query投影至64维 | 输出分辨率、梯度、有限值 | DPT输出与真实几何对齐 |
| objectness head | 每query一个raw logit，[B,Q] | shape、有限性、raw范围 | 场景级matching与真实GT空场景口径 |
| pixel/Gaussian membership | query-mask dot product形成[B,Q,V,H,W]；按view-y-x映射到Gaussian，过滤用source_indices | 手算索引、permutation、不在不同view独立match | A原Gaussian schema/过滤映射、C跨视角ID证据 |

后续 criterion 必须做场景级 Hungarian matching，不允许逐视角匹配掩盖 ID switching；本轮不创建 student.py、criterion.py、train_instance.py。B-01先做模块synthetic smoke测试不等于真实验证；真实fixture/DPT初始化和C targets到位后另过集成门。GT优先，不启动8-view、伪标签正式训练、render loss、joint fine-tuning或长期GPU实验。

## 7. A/C 最终确认与停止门

A：真实token缓存、导出具体位置与特殊token规则；官方2views和dirty源码fingerprint；明确冻结参数/模式/梯度证据；原DPT权重及可复制键；RGB/K/crop/frame顺序；预测XYZ坐标/尺度；confidence语义；Gaussian raw schema与过滤源索引。

C：实现实际数据到合同的解码；确认-1/0/>0，背景/stuff，valid/confidence及跨视角ID；提供真实targets和同一scene/frame顺序，以及空GT、单视角可见、ignore案例；数据来源/划分隔离与标签一致性证据。B不提前解析原始ScanNet/SIU3R。

B-00完成后停止等待用户审核。真实集成依赖为BLOCKED，合同synthetic工作不受其阻塞。下一阶段仅建议B-01，未授权自动实施。

## 8. 证据路径

远程交付仅五个要求文件。本地 `D:\崇新学堂\大三上\开创2\codex_remote_3dgs_test\b00_local` 保存 `before.json`、`checkpoint_hash.json`、传输hash收据与后续测试/结束审计。基线包含36,318个文件；全部记录size/mtime，适用的小型源码/文档另记录SHA256（缓存、日志、已有结果不做内容哈希）。不将此范围声称为全部大文件内容逐字节复核。

审计工具事件：首次 PowerShell ConvertFrom-Json 默认对象解析拒绝 Linux 下仅大小写不同的路径；原始 JSON 未损坏，改用 `-AsHashtable` 后成功读回36,318条。不是合同测试失败；完整事件保留于会话输出。
