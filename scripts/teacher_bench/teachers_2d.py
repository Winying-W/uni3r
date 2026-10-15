"""2D teachers on the validation benchmark (run in envs/teacher, torch 2.5).

  --method sam2_frames   SAM2.1-L automatic mask generator on every 10th frame of each val
                         scan + all context frames -> <work>/sam2_frames/<scan>/<fid>.png.
                         These are (a) the single-frame SAM2 teacher and (b) the 2D input of
                         MaskClustering (replacing CropFormer, whose weights are offline).
  --method sam2_amg      pack sam2_frames into the benchmark pair layout (no cross-view IDs)
  --method sam2_video    SAM2.1-L video predictor: AMG masks on c1, propagate through <=16
                         frames to c2, add c2 masks not explained by tracks, propagate back.
                         -> cross-view consistent IDs from 2D tracking only
  --method grounded_sam  GroundingDINO-B (18 ScanNet thing names) boxes -> SAM2.1-L masks

Instance-id maps: overlapping masks are painted largest-first so smaller masks stay on top
(standard practice for SAM-style AMG, e.g. SAM3D); for detector teachers painted by
ascending score. 0 = no instance.
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
GT_ROOT = os.environ.get('TEACHER_BENCH_DATA', str(ROOT / '.cache/teacher_bench_data'))
CKPT = os.environ.get('TEACHER_CHECKPOINT_ROOT', str(ROOT / 'checkpoints/teachers'))
SAM2_CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
THINGS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
          'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
          'toilet', 'sink', 'bathtub', 'otherfurniture']
PROMPT_NAME = {t: ('other furniture' if t == 'otherfurniture' else t) for t in THINGS}


def rgb(scan, fid):
    return np.array(Image.open(f'{GT_ROOT}/{scan}/color/{fid}.jpg').convert('RGB'))


def paint(masks, order, shape):
    idmap = np.zeros(shape, np.uint16)
    for k in order:
        idmap[masks[k]] = k + 1
    return idmap


def amg_idmap(gen, img):
    out = gen.generate(img)
    if not out:
        return np.zeros(img.shape[:2], np.uint16), {}
    masks = [o['segmentation'] for o in out]
    order = np.argsort([-m.sum() for m in masks])  # largest first, small on top
    idmap = paint(masks, order, img.shape[:2])
    meta = {str(k + 1): {'score': float(out[k]['predicted_iou']), 'label': None}
            for k in range(len(out)) if (idmap == k + 1).any()}
    return idmap, meta


def save_pair(d, frames, meta):
    os.makedirs(d, exist_ok=True)
    for fid, m in frames.items():
        Image.fromarray(m.astype(np.uint16)).save(f'{d}/{fid}.png')
    json.dump(meta, open(f'{d}/meta.json', 'w'))


def scan_frames(scan, pairs):
    items = sorted(int(f.split('.')[0]) for f in os.listdir(f'{GT_ROOT}/{scan}/color'))
    fr = {f for f in items if f % 10 == 0}
    for p in pairs:
        if p['scan'] == scan:
            fr |= set(p['context_ids'])
    return sorted(fr)


def run_sam2_frames(args, pairs):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    gen = SAM2AutomaticMaskGenerator(build_sam2(SAM2_CFG, f'{CKPT}/sam2.1/sam2.1_hiera_large.pt'))
    scans = sorted({p['scan'] for p in pairs})[args.shard_index::args.shard_count]
    for scan in scans:
        od = f'{args.work}/sam2_frames/{scan}'
        os.makedirs(od, exist_ok=True)
        for fid in scan_frames(scan, pairs):
            if os.path.exists(f'{od}/{fid}.json'):
                continue
            idmap, meta = amg_idmap(gen, rgb(scan, fid))
            Image.fromarray(idmap).save(f'{od}/{fid}.png')
            json.dump(meta, open(f'{od}/{fid}.json', 'w'))  # written last = done marker
        print('done', scan, flush=True)


def run_sam2_amg(args, pairs):
    for p in pairs:
        (c1, c2), scan = p['context_ids'], p['scan']
        src = f'{args.work}/sam2_frames/{scan}'
        m1 = np.array(Image.open(f'{src}/{c1}.png')).astype(np.int64)
        m2 = np.array(Image.open(f'{src}/{c2}.png')).astype(np.int64)
        j1, j2 = json.load(open(f'{src}/{c1}.json')), json.load(open(f'{src}/{c2}.json'))
        off = int(m1.max()) + 1  # disjoint ID ranges: no cross-view identity claimed
        m2 = np.where(m2 > 0, m2 + off, 0)
        meta = {**j1, **{str(int(k) + off): v for k, v in j2.items()}}
        save_pair(f'{args.out}/{scan}_{c1}_{c2}', {c1: m1, c2: m2}, meta)


def run_sam2_video(args, pairs):
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    ck = f'{CKPT}/sam2.1/sam2.1_hiera_large.pt'
    gen = SAM2AutomaticMaskGenerator(build_sam2(SAM2_CFG, ck))
    vp = build_sam2_video_predictor(SAM2_CFG, ck)
    for p in pairs[args.shard_index::args.shard_count]:
        (c1, c2), scan = p['context_ids'], p['scan']
        od = f'{args.out}/{scan}_{c1}_{c2}'
        if os.path.exists(f'{od}/meta.json'):
            continue
        items = sorted(int(f.split('.')[0]) for f in os.listdir(f'{GT_ROOT}/{scan}/color'))
        seq = [f for f in items if c1 <= f <= c2]
        idx = np.unique(np.round(np.linspace(0, len(seq) - 1, min(16, len(seq)))).astype(int))
        seq = [seq[i] for i in idx]
        tmp = tempfile.mkdtemp(dir=args.work)
        for k, f in enumerate(seq):
            shutil.copy(f'{GT_ROOT}/{scan}/color/{f}.jpg', f'{tmp}/{k:05d}.jpg')
        first, meta0 = amg_idmap(gen, rgb(scan, c1))
        ids0 = [int(i) for i in np.unique(first) if i > 0][:args.max_objects]
        score = {}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            st = vp.init_state(video_path=tmp)
            for oid in ids0:
                vp.add_new_mask(st, frame_idx=0, obj_id=oid, mask=first == oid)
                score[oid] = meta0[str(oid)]['score']  # SAM2 predicted IoU of the seed mask
            last = {}
            for fi, oids, logits in vp.propagate_in_video(st):
                if fi == len(seq) - 1:
                    last = {o: (logits[k, 0] > 0).cpu().numpy() for k, o in enumerate(oids)}
            # objects visible only in c2: AMG masks not explained (IoU<0.5) by any track
            lastmap, metaL = amg_idmap(gen, rgb(scan, c2))
            nxt = max(ids0, default=0) + 1
            for i in [int(i) for i in np.unique(lastmap) if i > 0]:
                m = lastmap == i
                best = max((np.logical_and(m, t).sum() / np.logical_or(m, t).sum()
                            for t in last.values()), default=0.0)
                if best < 0.5 and len(score) < args.max_objects:
                    vp.add_new_mask(st, frame_idx=len(seq) - 1, obj_id=nxt, mask=m)
                    score[nxt] = metaL[str(i)]['score']
                    nxt += 1
            out = {}
            for fi, oids, logits in vp.propagate_in_video(st, start_frame_idx=len(seq) - 1, reverse=True):
                if fi == 0:
                    out[c1] = {o: (logits[k, 0] > 0).cpu().numpy() for k, o in enumerate(oids)}
            for fi, oids, logits in vp.propagate_in_video(st):
                if fi == len(seq) - 1:
                    out[c2] = {o: (logits[k, 0] > 0).cpu().numpy() for k, o in enumerate(oids)}
            vp.reset_state(st)
        shutil.rmtree(tmp)
        frames = {}
        for f, d in out.items():
            oids = list(d)
            order = np.argsort([-d[o].sum() for o in oids])
            idmap = np.zeros((256, 256), np.uint16)
            for k in order:
                idmap[d[oids[k]]] = oids[k]
            frames[f] = idmap
        save_pair(od, frames, {str(o): {'score': s, 'label': None} for o, s in score.items()})
        print('done', od, flush=True)


def run_grounded_sam(args, pairs):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    proc = AutoProcessor.from_pretrained(f'{CKPT}/grounding-dino-base')
    det = AutoModelForZeroShotObjectDetection.from_pretrained(f'{CKPT}/grounding-dino-base').cuda().eval()
    sam = SAM2ImagePredictor(build_sam2(SAM2_CFG, f'{CKPT}/sam2.1/sam2.1_hiera_large.pt'))
    text = '. '.join(PROMPT_NAME[t] for t in THINGS) + '.'
    name2cls = {PROMPT_NAME[t]: t for t in THINGS}

    def to_cls(phrase):
        phrase = phrase.strip().lower()
        if phrase in name2cls:
            return name2cls[phrase]
        hits = [n for n in name2cls if n in phrase]  # e.g. "shower curtain curtain"
        return name2cls[max(hits, key=len)] if hits else None

    for p in pairs[args.shard_index::args.shard_count]:
        (c1, c2), scan = p['context_ids'], p['scan']
        od = f'{args.out}/{scan}_{c1}_{c2}'
        if os.path.exists(f'{od}/meta.json'):
            continue
        frames, meta, nid = {}, {}, 1
        for fid in (c1, c2):
            img = rgb(scan, fid)
            inp = proc(images=Image.fromarray(img), text=text, return_tensors='pt').to('cuda')
            with torch.no_grad():
                o = det(**inp)
            r = proc.post_process_grounded_object_detection(
                o, inp.input_ids, threshold=args.box_threshold, text_threshold=args.text_threshold,
                target_sizes=[img.shape[:2]])[0]
            labels = r.get('text_labels', r['labels'])
            keep = [k for k, l in enumerate(labels) if to_cls(l)]
            masks, scores, cls = [], [], []
            if keep:
                boxes = r['boxes'][keep].cpu().numpy()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    sam.set_image(img)
                    m, _, _ = sam.predict(box=boxes, multimask_output=False)
                m = m.reshape(len(keep), *img.shape[:2]) > 0
                # mask NMS: drop a mask that overlaps a higher-scored one by IoU > 0.8
                sc = r['scores'][keep].cpu().numpy()
                for k in np.argsort(-sc):
                    if m[k].sum() == 0 or any(
                            np.logical_and(m[k], q).sum() / np.logical_or(m[k], q).sum() > 0.8 for q in masks):
                        continue
                    masks.append(m[k]); scores.append(float(sc[k])); cls.append(to_cls(labels[keep[k]]))
            idmap = np.zeros(img.shape[:2], np.uint16)
            for k in np.argsort(scores):  # ascending score: best on top
                idmap[masks[k]] = nid + k
            for k in range(len(masks)):
                if (idmap == nid + k).any():
                    meta[str(nid + k)] = {'score': scores[k], 'label': cls[k]}
            nid += len(masks)
            frames[fid] = idmap
        save_pair(od, frames, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--pairs', default='/opt/jensen_cache/teacher_bench_data/pairs.json')
    ap.add_argument('--work', default='/opt/jensen_cache/teacher_bench/work')
    ap.add_argument('--out', default=None)
    ap.add_argument('--shard_count', type=int, default=1)
    ap.add_argument('--shard_index', type=int, default=0)
    ap.add_argument('--max_objects', type=int, default=64)
    ap.add_argument('--box_threshold', type=float, default=0.3)
    ap.add_argument('--text_threshold', type=float, default=0.25)
    args = ap.parse_args()
    args.out = args.out or f'/opt/jensen_cache/teacher_bench/{args.method}'
    os.makedirs(args.work, exist_ok=True)
    pairs = json.load(open(args.pairs))
    {'sam2_frames': run_sam2_frames, 'sam2_amg': run_sam2_amg, 'sam2_video': run_sam2_video,
     'grounded_sam': run_grounded_sam}[args.method](args, pairs)


if __name__ == '__main__':
    main()
