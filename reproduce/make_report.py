"""Build the final report only from all three completed full evaluations."""
import hashlib
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parent
state = json.loads((here / 'active_runs.json').read_text())
root = Path(state['storage'])
refs = {2: [25.53, .8727, .1380, .5584, .8268],
        8: [24.71, .851, None, .554, .807],
        16: [23.32, .816, None, .558, .835]}
keys = ['psnr', 'ssim', 'lpips', 'miou', 'accuracy']
results = {}
for views, expected in [(2, 320), (8, 49), (16, 49)]:
    folder = root / state['runs'][str(views)]['directory']
    path = folder / 'metrics.json'
    if not path.exists():
        raise SystemExit(f'Incomplete: {views}-view metrics not written yet')
    data = json.loads(path.read_text())
    assert not data['smoke_run'] and data['num_batches'] == data['total_batches'] == expected
    assert data['input_views'] == views and data['pose_align_steps'] == 100
    assert data['gpu'] == 'NVIDIA L20' and all(math.isfinite(data[k]) for k in keys)
    log = (folder / 'eval.log').read_text()
    assert 'Total time:' in log and 'Averaged stats:' in log and 'Traceback (most recent call last)' not in log
    assert state['runs'][str(views)].get('exit_code') == 0, 'Confirm process exit before final report'
    results[views] = (data, path, hashlib.sha256(path.read_bytes()).hexdigest())

lines = ['# Uni3R ScanNet：开发机 L20 完整评测', '',
         '官方 2／8／16 视角权重均已完成发布测试加载器的全量评测，三项进程均正常退出。',
         '下面是本机实际结果，不声称与论文分数完全一致。', '',
         '| 输入视角 | 场景／样本 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | mIoU ↑ | Accuracy ↑ |',
         '|---|---|---:|---:|---:|---:|---:|']
for views, (data, path, digest) in results.items():
    values = ' | '.join(f'{data[k]:.6f}' for k in keys)
    lines.append(f"| {views} | {data['scene_count']}／{data['num_batches']} | {values} |")
lines += ['', '## 论文对比', '', '参考：https://arxiv.org/html/2508.03643，固定视角表 1／2。', '',
          '| 视角 | 指标 | 本机 | 论文 | 本机 − 论文 |', '|---|---|---:|---:|---:|']
for views, (data, _, _) in results.items():
    for key, ref in zip(keys, refs[views]):
        if ref is not None:
            lines.append(f'| {views} | {key} | {data[key]:.6f} | {ref:.6f} | {data[key]-ref:+.6f} |')
lines += ['', '论文第 4.1 节写 40 个未见 ScanNet 场景；发布代码的 2 视角使用选定的 40 场景，',
          '8／16 视角则读取 49 场景。本次完整保留发布加载器，因此多视角比较存在场景协议差异。',
          '原安装说明使用 PyTorch 2.1.0，本次为 2.7.1；gsplat 在原说明中未固定版本。分数差异原因尚未被逐项归因。', '',
          '深度指标保存在原始 JSON 中，来自目标视角；未与论文表 1 的源视角深度混比。',
          'mIoU 沿用发布代码的 MeanIoU 聚合。推理时间包含原模型 distiller 前向，',
          '运行在共享 L20 上，不与论文 A100 的推理速度作直接比较。', '',
          '## 数据、环境与运行方法', '',
          f'所有资产和运行日志位于 `{root}`。ScanNet 压缩包及官方权重分片均通过 SHA256 校验，',
          '合并权重校验清单位于 `checkpoints/{2,8,16}views.manifest.json`。',
          'Python 3.10 venv 继承 fastw 的 torch 2.7.1+cu128；CUDA 12.9，gsplat 1.5.3。',
          '三次正式运行的代码、包版本和 LSeg 补丁快照一致，见 `provenance_audit.json`。', '',
          '配置：256×256、batch size 1、seed 777、AMP 关闭、100 步位姿对齐。',
          '加速只跳过中间步骤不参与 RGB L1 优化目标的语义计算，最终渲染和所有指标保持完整。',
          '与原始路径的同样本全部指标精确一致，见 `alignment_validation.json`。',
          '额外尝试的 RGB-only 光栅化未用于正式结果。没有训练或微调权重。', '',
          '```bash', 'cd /file_system/vepfs/intern/haozhe.jia/projects/FastWAM/3DGS/Uni3R',
          '# 先用 nvidia-smi 选择空闲显存足够的开发机 L20；GPU 5 仅为示例。',
          'UNI3R_GPU=5 UNI3R_VIEWS="2 8 16" UNI3R_FAST_ALIGNMENT=True bash reproduce/run_local.sh',
          '```', '', '详细环境准备与单视角命令见 `reproduce/README.md`。', '', '## 原始结果', '']
for views, (data, path, digest) in results.items():
    lines += [f'- {views} 视角：`{path}`', f'  SHA256：`{digest}`；同目录 `eval.log` 包含完整日志。']
report = '\n'.join(lines) + '\n'
(here / 'RESULTS.md').write_text(report)
(root / 'RESULTS.md').write_text(report)
(root / 'results_summary.json').write_text(json.dumps({str(v): d for v, (d, _, _) in results.items()}, indent=2))
print(here / 'RESULTS.md')
print(root / 'RESULTS.md')
