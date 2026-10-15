"""Convert a public ScanNet RGB-D archive using verified per-scene RGB cameras.

The mirror contains 640x480 RGB and depth, with a subset of original frames.
Intrinsics come from the original-resolution RGB images in a second mirror,
and are rescaled using dimensions read from those images. This explicitly uses
the RGB camera model; it is not the README's original depth-camera pipeline.
"""
import argparse
import concurrent.futures
import io
import json
from pathlib import Path
import tarfile
import cv2
import numpy as np


def process(scene, buffers, calibration, output):
    dst = output / scene
    marker = dst / 'conversion.json'
    if marker.exists():
        return json.loads(marker.read_text())
    if not (calibration / scene / 'intrinsic.txt').exists() or not (calibration / scene / 'image_size.json').exists():
        return {'scene': scene, 'frames': 0, 'status': 'excluded_missing_calibration'}
    k = np.loadtxt(calibration / scene / 'intrinsic.txt')[:3, :3].astype(np.float32)
    size = json.loads((calibration / scene / 'image_size.json').read_text())
    common = sorted(set(buffers['color']) & set(buffers['depth']) & set(buffers['pose']))
    if not common:
        raise ValueError(f'No complete frames: {scene}')
    for folder in ('color', 'depth', 'pose'):
        (dst / folder).mkdir(parents=True, exist_ok=True)
    saved = 0
    for frame in common:
        pose = np.loadtxt(io.BytesIO(buffers['pose'][frame])).astype(np.float32)
        if not np.isfinite(pose).all():
            continue
        rgb = cv2.imdecode(np.frombuffer(buffers['color'][frame], np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(buffers['depth'][frame], np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
            raise ValueError(f'RGB/depth mismatch: {scene}/{frame}')
        if depth.dtype != np.uint16:
            raise ValueError(f'Expected uint16 millimetre depth: {scene}/{frame}')
        if not np.any(depth):
            continue
        h, w = depth.shape
        intrinsic = k.copy()
        intrinsic[0] *= w / size['width']
        intrinsic[1] *= h / size['height']
        name = f'{frame:06d}'
        if not cv2.imwrite(str(dst / 'color' / f'{name}.png'), rgb):
            raise IOError(f'Cannot write RGB: {scene}/{frame}')
        (dst / 'depth' / f'{name}.png').write_bytes(buffers['depth'][frame])
        np.savez(dst / 'pose' / f'{name}.npz', camera_intrinsics=intrinsic, camera_pose=pose)
        saved += 1
    result = {'scene': scene, 'frames': saved, 'status': 'converted', 'intrinsics_reference': 'resized_RGB_camera',
              'calibration_native_size': [size['width'], size['height']],
              'depth_unit': 'millimetres', 'pose': 'camera_to_world'}
    marker.write_text(json.dumps(result, indent=2))
    return result


def main():
    root = Path(__file__).resolve().parents[2] / 'data'
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=root / 'scannet/archives/scannet_2d.tar.gz')
    p.add_argument('--calibration', type=Path, default=root / 'scannet/calibration')
    p.add_argument('--output', type=Path, default=root / 'scannet_processed')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-scenes', type=int, default=0, help='0 converts the full archive')
    args = p.parse_args()
    cv2.setNumThreads(1)
    reports, pending = [], set()
    submitted = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        def submit(scene, buffers):
            nonlocal pending, submitted
            pending.add(pool.submit(process, scene, buffers, args.calibration, args.output))
            submitted += 1
            if len(pending) >= 2 * args.workers:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    result = f.result(); reports.append(result); print(result, flush=True)

        scene = None
        buffers = None
        # Large reads avoid millions of small network-filesystem read calls.
        with tarfile.open(args.archive, mode='r|gz', bufsize=8 * 1024 * 1024) as archive:
            for member in archive:
                parts = Path(member.name).parts
                if len(parts) < 2 or not parts[1].startswith('scene'):
                    continue
                new_scene = parts[1]
                if new_scene != scene:
                    if scene is not None:
                        submit(scene, buffers)
                        if args.max_scenes and submitted >= args.max_scenes:
                            scene = None
                            break
                    scene = new_scene
                    buffers = {k: {} for k in ('color', 'depth', 'pose')}
                    already_done = (args.output / scene / 'conversion.json').exists()
                if already_done:
                    continue
                if member.isfile() and len(parts) == 4 and parts[2] in buffers:
                    frame = int(Path(parts[3]).stem)
                    buffers[parts[2]][frame] = archive.extractfile(member).read()
            if scene is not None:
                submit(scene, buffers)
        for f in concurrent.futures.as_completed(pending):
            result = f.result(); reports.append(result); print(result, flush=True)
    reports.sort(key=lambda r: r['scene'])
    report_path = args.output.parent / ('scannet_conversion_smoke.json' if args.max_scenes else 'scannet_conversion.json')
    report_path.write_text(json.dumps({'archive': str(args.archive), 'max_scenes': args.max_scenes, 'scenes': reports}, indent=2))


if __name__ == '__main__':
    main()
