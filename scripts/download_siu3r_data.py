"""Recreate the exact SIU3R val scans and train-GT archives from saved splits."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

from download_reproduction import HF_ENDPOINT, ROOT, download


def tree(directory, work):
    url = f'{HF_ENDPOINT}/api/datasets/insomnia7/SIU3R/tree/main/{directory}?limit=1000'
    result = []
    while url:
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            headers = Path(tmp) / 'headers'
            raw = subprocess.check_output([
                'curl', '-4', '-fsSL', '--retry', '5', '--connect-timeout', '20',
                '--max-time', '180', '-D', str(headers), url,
            ])
            result.extend(json.loads(raw))
            links = re.findall(r'<([^>]+)>;\s*rel="next"', headers.read_text())
            url = links[-1] if links else None
            if url:
                url = url.replace('https://huggingface.co', HF_ENDPOINT)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['test', 'validation'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    data = Path(os.environ.get('DATA_ROOT', ROOT / 'data'))
    test = args.kind == 'test'
    split = data / 'splits' / ('test_siu3r_val.txt' if test else 'val_scannet.txt')
    scans = split.read_text().split()
    assert len(scans) == (312 if test else 41), 'Unexpected split count'
    assert len(set(scans)) == len(scans)
    assert all(re.fullmatch(r'scene\d{4}_\d{2}', scan) for scan in scans)
    dest = data / ('siu3r_val_full' if test else 'siu3r_trainval_gt')
    archives = dest / 'archives'
    archives.mkdir(parents=True, exist_ok=True)
    manifest = tree('scannet/val' if test else 'scannet/train', dest)
    lookup = {Path(item['path']).stem.removesuffix('.tar'): item
              for item in manifest if item['type'] == 'file'}
    missing = sorted(set(scans) - lookup.keys())
    if missing:
        raise RuntimeError(f'Scans missing from official source: {missing}')
    (dest / 'source_manifest.json').write_text(json.dumps(
        {'repository': 'insomnia7/SIU3R', 'split': split.read_text(),
         'files': [lookup[s] for s in scans]}, indent=2))

    def fetch(scan):
        marker = dest / scan / '.extracted'
        if test and marker.exists():
            return {'scan': scan, 'status': 'already_extracted'}
        archive = download('insomnia7/SIU3R', 'datasets', lookup[scan],
                           archives / f'{scan}.tar.gz')
        if test:
            # Python's data filter rejects unsafe paths, links and special files.
            count = 0
            with tarfile.open(archive, 'r|gz', bufsize=8 * 1024 * 1024) as stream:
                for member in stream:
                    parts = Path(member.name).parts
                    if not parts or parts[0] != scan:
                        raise RuntimeError(f'Unexpected archive root: {member.name}')
                    stream.extract(member, path=dest, filter='data')
                    count += member.isfile()
            scene = dest / scan
            for folder in ['color', 'depth', 'extrinsic', 'instance']:
                if not (scene / folder).is_dir():
                    raise RuntimeError(f'Missing extracted folder: {scene / folder}')
            marker.write_text(json.dumps({'files': count, 'source_lfs': lookup[scan].get('lfs')}))
            archive.unlink()  # Test split retains extracted data only, as requested.
        print('Complete:', scan, flush=True)
        return {'scan': scan, 'status': 'verified_and_extracted' if test else 'verified_archive'}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(fetch, scans))
    (dest / 'download_verification.json').write_text(json.dumps(
        {'complete': True, 'count': len(records), 'records': records}, indent=2))


if __name__ == '__main__':
    main()
