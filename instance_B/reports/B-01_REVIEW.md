# B-01｜实例学生实现与独立测试

日期：2026-10-08（Asia/Shanghai）。状态：**B01_COMPLETE_AWAITING_REVIEW**；IMPLEMENTED、TESTED_SYNTHETIC、DPT_INIT_VERIFIED。真实集成：NOT RUN；WAITING_FOR_A / WAITING_FOR_C。最终限定文件边界检查通过，已停止等待审核。

## 1. 本轮范围与源码复核

在SSH Host `010` 的 `/mnt/workspace/Jensen/project/3DGS/Uni3R` 操作。HEAD仍为 `4a5dd00b6737a95afd4c6b1760cad166cf17d3d4`，branch `main`，既有dirty工作区保留。解释器 `/root/miniconda3/envs/uni3r/bin/python`，Python3.10.22，torch2.1.0+cu121；不改环境、不装依赖、不用GPU、不启动重建/实例正式训练。

开工读取B_PROGRESS、B-00_REVIEW、共享contracts、旧test_contracts与原DPT及父类/位置函数。共享合同仍0.1.0/PROPOSED_FOR_TEAM_INTEGRATION。真实调用在 `uni3r/uni3r.py:131` 将原images直接传DPT；`(images+1)/2`在VGGT路径。B因此保留DPT input_identity 的[-1,1]输入，不按docstring更改归一化。

所有本轮写入仅instance_B。共享合同、B-00测试/历史报告、原Uni3R及VGGT源码只读。没有完整仓库重新扫描，没有再次计算12 GB checkpoint全文件hash，没有沿data越界链接读取。没有修改A/C模块、环境或Git分支；没有commit/push。

## 2. 学生实现

- `student.py`提供StudentConfig、InstanceStudent和初始化读取、纯索引工具，`forward(FrozenSceneBatch)->InstancePrediction`复用唯一共享合同。
- `dpt_adapter.py`继承实际Uni3R DPT；仅B自己的实例将 `intermediate_layer_idx` 设为[0,1,2,3]，按[4,11,17,23]取四个patch-only缓存，以patch_start_idx=0调用原结构。没有虚构24层缓存、没有再次删除special tokens、没有修改原类/全局配置。
- instance_dpt保留全部原结构、256维dense输出和7×7 RGB input_merger，独立且可训练。所有frozen输入detach；DPT、adapter和decoder计算保留autograd。
- adapter把dense features按1/8、1/16平均池化后独立1×1投影+LayerNorm。默认每场景memory=2×(32²+16²)=2,560；配置预算不得超过8,192。只有Q×Q self-attention和Q×M cross-attention，没有M×M高分辨率attention。
- 位置包含2D像素中心、learned view embedding、scale embedding和detached predicted XYZ。每场景在V/H/W全部坐标上求均值center；scale=sqrt(mean(||xyz-center||²))，分母clamp_min(1e-6)。float64归约减少float32溢出风险，全同坐标产生零标准化坐标；不跨batch混合、不使用GT。
- geometry_confidence先detach，再 `log1p(score)`，随后平均池化并线性投影。保留无上界非负模型分数含义，不clamp到[0,1]、不当成标签概率，也不用于选择GT或坐标监督。
- 每场景同一组Q个queries共享所有视角，通过6层decoder，输出256维query_features。64维query mask embedding与64维pixels点积、除sqrt(64)=8，生成raw mask_logits；独立objectness线性头输出raw logits，无sigmoid/softmax。
- `flatten_membership_logits`保留 `index=view*H*W+y*W+x`，支持每场景不同Mi的source_indices选择，返回list[Q,Mi]；不定义阈值、聚类或renderer规则。
- 正式默认初始化必须提供可校验的2views DPT artifact+receipt，缺少即DPT_INIT_BLOCKED。显式synthetic_random_init仅供synthetic输入，不能消费真实缓存。

边界：共享合同允许的所有dtype不代表学生全部支持；本轮学生显式支持float32，其他dtype报错，AMP未验证。物理坐标系、真实token数值语义仍待A确认。共享合同没有修改，也没有通过放宽旧测试换取通过。

## 3. DPT安全提取与严格初始化

原文件 `checkpoints/output/2views.pth`，11,913,939,138 bytes，mtime_ns=1791421891990014631，与B-00保存的stat一致。checkpoint SHA256沿用B-00完整核验收据：

`660aed13d6a8bccd078be35210f803d0f6d3b312f63dd468bbccc1c7f3aacf24`

本轮没有重新全文件hash；size/mtime一致是收据复用依据，不能代替对抗修改场景的重新密码学验证。初查主机MemAvailable约402,242,780 KiB，仅为当时可用内存，不是独占资源配额。

ZIP有4,400个record，data.pkl=1,015,394 bytes，protocol2。实际GLOBAL仅Namespace、OrderedDict、Float/Half/LongStorage与_rebuild_tensor_v2。`extract_dpt_init.py`用pickletools解析opcode，仅构造惰性Python描述；GLOBAL/REDUCE/NEWOBJ/BUILD不执行checkpoint指定函数/对象逻辑，未知形式拒绝。随后只读取选中的 `model.dpt_head.*` ZIP存储，校验CRC/字节数/stride范围；不对原checkpoint调用torch.load，不实例化Uni3R/LSeg/distiller。只构造原DPT做strict检查。恶意GLOBAL拒绝路径另有单测。

结果：

| 项目 | 实测 |
|---|---|
| DPT keys | 60/60，全部float32 |
| 参数数 | 32,950,528 |
| selected storage读取 | 131,802,112 bytes |
| missing/unexpected | 0 / 0，strict=True |
| key/shape/有限值/复制内容 | 全部通过 |
| 初始化后requires_grad | 全部True |
| 提取和strict检查时间 | 1.2016 s，不含完整模型加载（未发生） |
| 小权重文件 | `results/B-01/20261008_cpu_01/dpt_init_from_2views.pt`，131,820,514 bytes |
| 小权重SHA256 | `a47b647c1d545b491dcb649ee509e911eea6d2ebd9b5a43a8642441d955b6b45` |

小权重是纯Tensor state_dict，通过已有torch的weights_only=True重新读取并逐张量比对。receipt列出全部60 keys/shapes、源文件stat、源hash依据、提取方法与产物hash。目的为 `INITIALIZATION_ONLY_NOT_TRAINED_STUDENT`；不是实例训练成果。

学生/reference/源state_dict分别有独立存储。修改学生参数不改变源权重；适配parity时原DPT参数仍与初始化一致；两步诊断后初始化参考也逐张量未变。没有用strict=False掩盖差异。

## 4. 默认与小配置

| 项目 | 默认完整前向 | 独立小测试/反向 |
|---|---|---|
| profile | default_v0 | synthetic_small（显式） |
| H×W / V | 256×256 / 2 | 32×32 / 2 |
| B | 1 | 1及2 |
| Q | 100 | 主测试7，另测1/11 |
| decoder层/heads | 6 / 8 | 2 / 8 |
| FFN维度 | 1024 | 512 |
| query dim / mask dim | 256 / 64 | 256 / 64 |
| patch/token/DPT结构 | 16 / 2048 / 原完整DPT | 同默认，未改成玩具网络 |
| memory tokens | 2,560 | 40 |
| 输入来源 | synthetic | synthetic |
| DPT初始化 | 官方2views提取 | 官方2views提取；另有显式random拒绝回退测试 |

## 5. 实际命令和测试结果

工作目录均为项目根；完整命令保存在run目录 `commands.txt`，stdout/console与JSON均可审计。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
 /root/miniconda3/envs/uni3r/bin/python -B -m instance_B.extract_dpt_init \
 --output-dir instance_B/results/B-01/20261008_cpu_01

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
 /root/miniconda3/envs/uni3r/bin/python -B -m instance_B.run_b01_checks \
 --run-dir instance_B/results/B-01/20261008_cpu_01/checks_01 --diagnostic-steps 2
```

两条命令exit=0。首次单测输出：`Ran 51 tests in 8.064s`；`OK (skipped=1)`。

| 测试 | 真实结果 |
|---|---|
| B-00合同回归 | 31 PASS，原测试未修改 |
| B-01学生独立测试 | 19 PASS，0 failures / 0 errors |
| REAL_UNI3R_FIXTURE_TEST | 1 SKIP，NOT RUN，A真实fixture未交付 |
| DPT_ADAPTER_SYNTHETIC_PARITY | PASS，max_abs_error=0.0，rtol1e-5/atol1e-6；B2/32²；24层独立生成synthetic参考输入，与四层patch-only适配对照 |
| special tokens和层序 | 改camera/register前缀不影响输出；dict插入顺序不影响层序；多余special token被拒绝；原reference层索引未变 |
| 前向/可变Q/B>1 | shape/dtype/有限性通过；B2的[2,7,2,32,32]、[2,7]、[2,7,256]；Q1/11通过 |
| batch隔离 | 改第二场景不影响第一场景，第一场景输出逐元素一致 |
| 梯度 | 123个参与参数tensor全部grad存在、有限且含非零；frozen inputs无grad；reference source不变 |
| XYZ/confidence边界 | 全同、近零尺度、大坐标有限性、score=0/10000均通过，不当概率截断 |
| flatten/filter | 手算2views×2×3，索引11和筛选[11,0,7]正确；空过滤有定义；重复/越界/错误dtype拒绝 |
| 64D mask缩放/raw | 全1 dot64/sqrt64=8，负query得到-8，无sigmoid |
| eval保存/重载 | strict state_dict重载后全部输出精确相同，max_abs_error=0.0；文件为eval_roundtrip_state_dict.pt |
| 初始化拒绝/随机显式 | 默认缺初始化拒绝；key/shape错误拒绝；synthetic_random_init显式可测 |
| GT/Teacher独立性 | 添加无关GT/teacher哨兵字段不改变输出 |
| attention范围 | 小配置两层实际hook均self(7,7)、cross(7,40)，无memory全局self-attention |

梯度分组（参数tensor计数）：DPT60、adapter8、position10、queries1、decoder36、query_norm2、mask_features2、query_mask_embedding2、objectness2；全部有限且非零。完整细节在unit_results.json。

默认完整配置另跑一次，**FULL_CONFIG_FORWARD=PASS**：39,532,161参数，CPU float32，mask `[1,100,2,256,256]`、objectness `[1,100]`、query `[1,100,256]`全部有限；前向1.5655s。不是把小配置结果称为默认通过。

资源：GPU屏蔽；intra-op=2、inter-op=1。完整检查总13.3958s，进程峰值RSS=1,498,496 KiB（约1.43 GiB，包含单测/模型/张量，不是纯前向内存）。未尝试AMP、GPU性能或默认完整配置反向；这些仍NOT RUN。时间为这一次诊断，非正式速度benchmark。

## 6. 两步optimizer诊断（非实例监督）

仅2次SGD.step，lr=1e-4，B2小配置，目标 `mean(mask_logits²)+mean(objectness_logits²)`。标量0.5519750→0.2976365仅记录数值，不证明分割有效、不作为实例训练loss。

| 跟踪参数 | 最大绝对更新 |
|---|---:|
| object_queries | 0.0002404074 |
| instance_dpt.projects.0.weight | 0.0000028433 |
| adapter.projections.0.weight | 0.0000919700 |
| decoder.0.cross_attention.in_proj_weight | 0.0000742525 |
| mask_features.weight | 0.0000026226 |
| objectness_head.weight | 0.0007843152 |

参考初始化权重逐张量不变。两步耗时1.1773s。结果及每步记录均保存；artifact名 `diagnostic_after_2steps_state_dict.pt`。持久化 `optimizer_diagnostic_admission.json` 防止重复调用诊断导致突破本轮最多3步预算；本轮不再执行optimizer.step。

## 7. 未验证项、失败与下一阶段

- REAL_INTEGRATION=NOT RUN：本轮未收到A真实特征fixture或C合同GT targets。不虚构真实scene/frame或提取特征来源；synthetic metadata保持SYNTHETIC_NO_CHECKPOINT，即使学生DPT使用真实预训练初始化也不改变输入证据类别。
- A需要给出真实四层缓存、明确全主干冻结证据、RGB/K/crop顺序、预测XYZ物理坐标/尺度、geometry score语义，以及Gaussian schema/filter映射。DPT提取已完成，可核对本轮receipt，不再把提取本身列为阻塞。
- C需要真实GT解码、跨视角同物ID/scene-frame对齐、-1/0/>0、valid/confidence和空场景案例、数据划分/来源。学生前向不读取这些监督。
- 不实现Hungarian matching、正式loss、criterion、train_instance；没有精度/召回/分割指标。下一步B-02_MATCHING_AND_LOSS仅为建议，须等待审核。
- 本轮实际测试无失败。观察到PyTorch TypedStorage弃用warning，保留在console，未安装/升级依赖。编辑进度文档时一次本地patch上下文匹配失败（无远程写入），随后依据当前文档定位成功；不是被隐藏的模型/测试失败。

## 8. 文件与证据

新增源码/配置/测试：`dpt_adapter.py`、`extract_dpt_init.py`、`student.py`、`configs/student_v0.json`、`tests/test_student.py`、`run_b01_checks.py`；新增本报告；更新README和B_PROGRESS。所有路径相对于instance_B。

`results/B-01/20261008_cpu_01/`保存初始化权重/receipt、commands、extraction_stdout、checks_01_console；`checks_01/`保存configuration_and_command、unittest_output、unit_results、full_config_forward、diagnostic_step_1/2、optimizer_diagnostic、completion以及两个测试state_dict；另有持久诊断准入记录。

最终限定检查已完成，证据 `results/B-01/20261008_cpu_01/boundary_audit.json`：

- 9个保护文件hash与开工一致：共享contracts、B-00测试、B-00历史报告、uni3r/dpt_head.py、uni3r/uni3r.py、uni3r/vggt_w_intr_emb.py、vendored DPT/heads utils/patch_embed源码。
- 本轮自有源码/配置/说明及上传日志读回hash匹配发布回执；Python AST语法检查通过，源码/文档无尾随空白。新增源码用git diff --no-index --stat核对；untracked文件不依赖普通git diff列出。
- HEAD和main保持原值；Git短状态与开工一致，因为instance_B开工前已整体untracked。该现象不意味着B没有新增文件；B目录清单和回执记录了实际写入。
- 审计只涵盖本轮自有写入及实际使用的关键源码，不声称整个共享工作区没有并发变化；未回滚任何他人改动。
- 结束时仅再更新本报告/B_PROGRESS并独占保存边界证据，使用expected SHA256检查防止覆盖并发编辑；最后读回结束状态。

合同源码SHA256仍为 `2810deaffde90952ce5f088d12ec112d24aed8c7f583c9498169261687ea072f`；旧合同测试仍为 `67d3a768cc8dea6110a60f4bd3470e14d8c7c2215e4162ab08fa6a6347ea45e4`。详细逐文件hash见边界证据。B-00历史结论/报告保留。

本轮停止：不自动进入B-02_MATCHING_AND_LOSS，不进行真实训练或更多optimizer诊断。
