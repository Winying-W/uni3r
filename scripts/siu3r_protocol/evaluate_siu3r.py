"""Score an export directory with SIU3R's own Evaluator (src/evaluator.py, unchanged).

Config mirrors SIU3R configs/main.yaml + src/config.py panoptic branch:
id2label = PANOPTIC_SEMANTIC2NAME (20 classes), stuffs = [0, 1], things = 2..19.
"""
import argparse
import json
import sys
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--siu3r_repo', required=True)
    ap.add_argument('--path', required=True)
    ap.add_argument('--no_instance', action='store_true',
                    help='disable mAP/PQ (methods without instance output, e.g. Uni3R / LSM)')
    args = ap.parse_args()
    sys.path.insert(0, args.siu3r_repo)
    from src.evaluator import Evaluator
    from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME, STUFF_CLASSES, THING_CLASSES
    inst = not args.no_instance
    cfg = SimpleNamespace(
        dataset_name='scannet', device='cuda', eval_path=args.path,
        eval_image_quality=True, eval_depth_quality=True,
        eval_context_miou=True, eval_target_miou=True,
        eval_context_pq=inst, eval_target_pq=inst, eval_context_map=inst, eval_target_map=inst,
        id2label=PANOPTIC_SEMANTIC2NAME, stuffs=STUFF_CLASSES, things=THING_CLASSES)
    assert len(cfg.id2label) == 20
    ev = Evaluator(cfg)
    ev.setup()
    result = ev.evaluate(args.path)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))


if __name__ == '__main__':
    main()
