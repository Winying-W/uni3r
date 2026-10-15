"""Convert CUT3R-preprocessed DSLR RGB-D into Uni3R's ScanNet++ layout.

Source poses are camera-to-world and uint16 depth is in millimetres, matching
Uni3R. iPhone images are excluded. RGB/depth pixels and intrinsics stay intact.
This is a public mirror subset, not a claim of the paper's full training split.
"""
import argparse
import concurrent.futures
import io
import json
from pathlib import Path
import tarfile
import numpy as np
from PIL import Image


def convert(archive, output):
    scene = archive.stem
    dst = output / scene / 'dslr'
    marker = dst / 'conversion.json'
    if marker.exists():
        return json.loads(marker.read_text())
    with tarfile.open(archive) as tar:
        with np.load(io.BytesIO(tar.extractfile(f'{scene}/scene_metadata.npz').read()), allow_pickle=False) as meta:
            images = meta['images'].copy()
            poses = meta['trajectories'].copy()
            intrinsics = meta['intrinsics'].copy()
        valid = [i for i, name in enumerate(images) if str(name).startswith('DSC')
                 and np.isfinite(poses[i]).all() and np.isfinite(intrinsics[i]).all()]
        # The upstream loader silently drops scenes with <=128 images.
        if len(valid) <= 128:
            return {'scene': scene, 'status': 'excluded_insufficient_dslr_frames', 'frames': len(valid)}
        for folder in ['rgb_resized_undistorted', 'render_depth', 'camera']:
            (dst / folder).mkdir(parents=True, exist_ok=True)
        for i in valid:
            name = str(images[i])
            rgb = tar.extractfile(f'{scene}/images/{name}.jpg').read()
            depth = tar.extractfile(f'{scene}/depth/{name}.png').read()
            with Image.open(io.BytesIO(rgb)) as im, Image.open(io.BytesIO(depth)) as dm:
                if im.size != dm.size:
                    raise ValueError(f'RGB/depth size mismatch: {scene}/{name}')
                if np.asarray(dm).dtype != np.uint16:
                    raise ValueError(f'Expected millimetre uint16 depth: {scene}/{name}')
            if not np.allclose(poses[i][3], [0, 0, 0, 1]):
                raise ValueError(f'Invalid homogeneous pose: {scene}/{name}')
            (dst / 'rgb_resized_undistorted' / f'{name}.JPG').write_bytes(rgb)
            (dst / 'render_depth' / f'{name}.png').write_bytes(depth)
            np.savez(dst / 'camera' / f'{name}.npz', intrinsic=intrinsics[i].astype(np.float32),
                     extrinsic=poses[i].astype(np.float32))
    report = {'scene': scene, 'status': 'converted', 'frames': len(valid),
              'source': str(archive.resolve()), 'depth_unit': 'millimetres', 'pose': 'camera_to_world'}
    marker.write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    data = Path(__file__).resolve().parents[2] / 'data'
    p.add_argument('--archives', type=Path, default=data / 'scannetpp/archives')
    p.add_argument('--output', type=Path, default=data / 'scannetpp_processed')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    archives = sorted(args.archives.glob('*.tar'))
    if not archives:
        raise SystemExit('No completed archives found')
    reports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert, archive, args.output) for archive in archives]
        for future in concurrent.futures.as_completed(futures):
            report = future.result()
            reports.append(report)
            print(report, flush=True)
    reports.sort(key=lambda r: r['scene'])
    (args.output.parent / 'scannetpp_conversion.json').write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
