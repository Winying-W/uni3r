"""Fetch a small, checksummed SIU3R GT-instance subset overlapping Uni3R eval."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import requests


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/instance_oracle/overlap_data'))
    parser.add_argument('--scenes', nargs='+', default=['scene0702_02', 'scene0689_00', 'scene0701_00'])
    parser.add_argument('--proxy', default='http://127.0.0.1:37890')
    parser.add_argument('--metadata-cache', type=Path, default=Path('/tmp/uni3r_oracle'))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.proxies.update({'https': args.proxy})
    base = 'https://huggingface.co/datasets/insomnia7/SIU3R/resolve/main/'

    def metadata(name, url):
        p = args.metadata_cache / name
        if p.exists():
            return p.read_bytes()
        r = session.get(url, timeout=60)
        r.raise_for_status()
        return r.content

    tree = json.loads(metadata('siu3r_val.json', 'https://huggingface.co/api/datasets/insomnia7/SIU3R/tree/main/scannet/val'))
    pairs = json.loads(metadata('siu3r_pairs.json', base + 'scannet/val_pair.json'))
    selected = [dict(p, original_pair_index=i) for i, p in enumerate(pairs) if p['scan'] in args.scenes]
    assert set(p['scan'] for p in selected) == set(args.scenes)
    (out / 'pairs.json').write_text(json.dumps(selected, indent=2))
    (out / 'SOURCE_README.md').write_bytes(metadata('siu3r_readme.md', base + 'scannet/README.md'))

    def download(scene):
        record = next(f for f in tree if f['path'].endswith(scene + '.tar.gz'))
        archive = out / (scene + '.tar.gz')
        expected_hash = record['lfs']['oid']
        url = base + record['path']
        if not archive.exists():
            response = session.head(url, allow_redirects=False, timeout=45)
            response.raise_for_status()
            resolved = response.headers.get('Location', url)
            part = archive.with_suffix('.part')
            for proxy_args in (['--noproxy', '*'], ['--proxy', args.proxy]):
                with (out / (scene + '.download.log')).open('a') as log:
                    result = subprocess.run(['curl', *proxy_args, '-fL', '--connect-timeout', '20',
                                             '--max-time', '240', '--speed-limit', '1024', '--speed-time', '45',
                                             '-C', '-', resolved, '-o', str(part)], stdout=log, stderr=log)
                if result.returncode == 0:
                    part.rename(archive)
                    break
            if not archive.exists():
                raise RuntimeError('Download failed: ' + scene)
        assert archive.stat().st_size == record['size'], scene
        h = hashlib.sha256()
        with archive.open('rb') as f:
            for chunk in iter(lambda: f.read(8 << 20), b''):
                h.update(chunk)
        assert h.hexdigest() == expected_hash, 'SHA256 mismatch: ' + scene
        with tarfile.open(archive) as tf:
            # Only extract ordinary scene files; never links or special devices.
            for member in tf:
                dest = (out / member.name).resolve()
                if not dest.is_relative_to(out) or not (member.isdir() or member.isfile()):
                    raise ValueError('Unsafe archive entry: ' + member.name)
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, dest.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
        print('Verified and extracted', scene, archive.stat().st_size, flush=True)
        return {'scene': scene, 'url': url, 'sha256': expected_hash, 'bytes': record['size']}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, args.scenes))
    (out / 'provenance.json').write_text(json.dumps({
        'source': 'WU-CVGL/SIU3R processed ScanNet validation, GT instance annotations',
        'selection': 'Three small archives in intersection with local Uni3R 40-scene two-view evaluation; selected before inference',
        'pairs': 'All six published SIU3R pairs per scene; original frame selection, not Uni3R evaluation frames',
        'archives': records}, indent=2))


if __name__ == '__main__':
    main()
