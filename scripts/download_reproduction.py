"""Download public reproduction assets, resuming and verifying HF LFS hashes."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import shutil

ROOT = Path(__file__).resolve().parents[1]


def api(path):
    with urllib.request.urlopen('https://huggingface.co/api/' + path, timeout=60) as r:
        return json.load(r)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_ranges(url, partial, size):
    """Retain a sequential prefix and fetch the remainder in bounded chunks."""
    directory = partial.with_name(partial.name + '.ranges')
    directory.mkdir(exist_ok=True)
    plan_path = directory / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan['size'] != size:
            raise RuntimeError('Remote size changed; refusing to reuse range files')
    else:
        plan = {'prefix': partial.stat().st_size if partial.exists() else 0, 'size': size}
        plan_path.write_text(json.dumps(plan))
    if (partial.stat().st_size if partial.exists() else 0) != plan['prefix']:
        raise RuntimeError('Sequential prefix changed while fetching ranges')
    ranges = [(start, min(start + 256 * 1024 * 1024, size) - 1)
              for start in range(plan['prefix'], size, 256 * 1024 * 1024)]

    def fetch_range(bounds):
        start, end = bounds
        dest = directory / str(start)
        if dest.exists() and dest.stat().st_size == end - start + 1:
            return dest
        subprocess.run(['curl', '-fsSL', '--noproxy', '*', '--retry', '5', '--retry-all-errors',
                        '--connect-timeout', '30', '--max-time', '600',
                        '--range', f'{start}-{end}', '-o', str(dest),
                        url + f'&range_start={start}'], check=True)
        if dest.stat().st_size != end - start + 1:
            raise RuntimeError(f'Incorrect range size: {dest}')
        return dest

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(fetch_range, ranges))
    merged = partial.with_name(partial.name + '.assembling')
    with merged.open('wb') as out:
        if plan['prefix']:
            with partial.open('rb') as src:
                shutil.copyfileobj(src, out, 8 * 1024 * 1024)
        for chunk in chunks:
            with chunk.open('rb') as src:
                shutil.copyfileobj(src, out, 8 * 1024 * 1024)
    # The caller verifies the full LFS SHA before publishing.
    return merged, directory


def download(repo, kind, item, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('lfs', {}).get('oid')
    if target.exists() and target.stat().st_size == item['size']:
        if not expected or sha256(target) == expected:
            print('Verified existing:', target, flush=True)
            return target
    partial = target.with_name(target.name + '.part')
    prefix = 'datasets/' if kind == 'datasets' else ''
    url = f'https://huggingface.co/{prefix}{repo}/resolve/main/{item["path"]}?download=true&t={int(time.time())}'
    print('Downloading:', target, item['size'], flush=True)
    if item['size'] >= 1024 ** 3:
        downloaded, range_dir = download_ranges(url, partial, item['size'])
    else:
        subprocess.run(['curl', '-fsSL', '--noproxy', '*', '--retry', '5', '--retry-all-errors', '--connect-timeout', '30',
                        '--speed-time', '120', '--speed-limit', '1024', '-C', '-',
                        '-o', str(partial), url], check=True)
        downloaded, range_dir = partial, None
    if downloaded.stat().st_size != item['size'] or (expected and sha256(downloaded) != expected):
        raise RuntimeError(f'Size or SHA-256 mismatch: {downloaded}')
    downloaded.replace(target)
    if range_dir:
        partial.unlink(missing_ok=True)
        shutil.rmtree(range_dir)
    print('Verified:', target, flush=True)
    return target


def download_public_url(url, target, expected_sha256):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == expected_sha256:
        print('Verified existing:', target, flush=True)
        return
    partial = target.with_name(target.name + '.part')
    subprocess.run(['curl', '-fsSL', '--noproxy', '*', '--retry', '5', '--retry-all-errors',
                    '--connect-timeout', '30', '-C', '-', '-o', str(partial), url], check=True)
    if sha256(partial) != expected_sha256:
        raise RuntimeError(f'SHA-256 mismatch: {partial}')
    partial.replace(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--views', type=int, nargs='+', default=[2], choices=[2, 8, 16])
    p.add_argument('--data-root', type=Path, default=ROOT.parent / 'data')
    p.add_argument('--weights-only', action='store_true',
                   help='Download official model weights without the ScanNet test archive')
    args = p.parse_args()
    manifest = api('models/HorizonRobotics/Uni3R/tree/main')
    (ROOT / 'checkpoints').mkdir(exist_ok=True)
    (ROOT / 'checkpoints/official_manifest.json').write_text(json.dumps(manifest, indent=2))
    tasks = []
    for n in args.views:
        for item in manifest:
            if item['path'].startswith(f'{n}views.pth.part_'):
                tasks.append(('HorizonRobotics/Uni3R', 'models', item, ROOT / 'checkpoints/output' / item['path']))
    if not args.weights_only:
        for item in api('datasets/Journey9ni/LSM/tree/main'):
            if item['path'] == 'scannet_test.tar':
                tasks.insert(0, ('Journey9ni/LSM', 'datasets', item, args.data_root / item['path']))
    for item in api('models/facebook/VGGT-1B/tree/main'):
        if item['path'] == 'model.pt':
            tasks.append(('facebook/VGGT-1B', 'models', item, ROOT / 'checkpoints/torch/hub/checkpoints/model.pt'))
    for item in api('models/Gofinge/LSeg/tree/main'):
        if item['path'] == 'demo_e200.ckpt':
            tasks.append(('Gofinge/LSeg', 'models', item, ROOT / 'checkpoints/pretrained_models/demo_e200.ckpt'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, *t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    clip_hash = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
    download_public_url(f'https://openaipublic.azureedge.net/clip/models/{clip_hash}/ViT-B-32.pt',
                        ROOT / 'checkpoints/clip/ViT-B-32.pt', clip_hash)
    download_public_url('https://download.pytorch.org/models/vgg16-397923af.pth',
                        ROOT / 'checkpoints/torch/hub/checkpoints/vgg16-397923af.pth',
                        '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0')
    import shutil
    for n in args.views:
        parts = sorted((ROOT / 'checkpoints/output').glob(f'{n}views.pth.part_*'))
        parts = [x for x in parts if not x.name.endswith('.part')]
        if not parts:
            raise RuntimeError(f'Missing checkpoint parts for {n} views')
        target = ROOT / f'checkpoints/output/{n}views.pth'
        tmp = target.with_suffix('.assembling')
        with tmp.open('wb') as out:
            for part in parts:
                with part.open('rb') as src:
                    shutil.copyfileobj(src, out, 8 * 1024 * 1024)
        tmp.replace(target)
        print('Assembled:', target, flush=True)


if __name__ == '__main__':
    main()
