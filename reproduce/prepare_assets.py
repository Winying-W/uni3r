"""Download the authors' evaluation assets to the user's EFS, resumably."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import requests

ROOT = Path(os.environ['UNI3R_STORAGE'])
ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')

def download(url, path, expected_size=None, sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_name(path.name + '.part')
        if 'huggingface.co/' in url or 'hf-mirror.com/' in url:
            response = requests.head(url, allow_redirects=False, timeout=60)
            response.raise_for_status()
            url = response.headers.get('Location', url)
        subprocess.run(['curl', '--noproxy', '*', '-fL', '--retry', '5', '--retry-delay', '5',
                        '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '120',
                        '-C', '-', url, '-o', str(tmp)], check=True)
        if expected_size is not None and tmp.stat().st_size != expected_size:
            raise RuntimeError(f'Incomplete download: {tmp}')
        tmp.rename(path)
    if sha256:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        if h.hexdigest() != sha256:
            raise RuntimeError(f'Checksum mismatch: {path}')
    return path

tar = download(f'{ENDPOINT}/datasets/Journey9ni/LSM/resolve/main/scannet_test.tar',
               ROOT/'data/scannet_test.tar', 1066352640,
               'd327b360c0743a9f7cc9d3fe895baaa3aa9ffbbbbae960b58ee9fe84e5d83a45')
marker = ROOT/'data/.extracted'
if not marker.exists():
    with tarfile.open(tar) as archive:
        for member in archive.getmembers():
            dest = (ROOT/'data'/member.name).resolve()
            if not dest.is_relative_to((ROOT/'data').resolve()) or member.issym() or member.islnk():
                raise RuntimeError(f'Unsafe tar member: {member.name}')
        archive.extractall(ROOT/'data')
    marker.touch()
roots = list((ROOT/'data').rglob('scannetv2-labels.combined.tsv'))
if len(roots) != 1:
    raise RuntimeError(f'Cannot uniquely locate ScanNet root: {roots}')
(ROOT/'data_root.txt').write_text(str(roots[0].parent))
manifest = ROOT/'checkpoints/manifest.json'
if manifest.exists():
    files = json.loads(manifest.read_text())
else:
    response = requests.get(f'{ENDPOINT}/api/models/HorizonRobotics/Uni3R/tree/main',
                            params={'recursive': 'true'}, timeout=60)
    response.raise_for_status()
    files = response.json()
    manifest.write_text(json.dumps(files, indent=2))
for views in (2, 8, 16):
    output = ROOT/f'checkpoints/{views}views.pth'
    complete = [f for f in files if f.get('type') == 'file'
                and Path(f['path']).name == f'{views}views.pth']
    parts = sorted([f for f in files if f.get('type') == 'file'
                    and Path(f['path']).name.startswith(f'{views}views.pth.part_')],
                   key=lambda f: f['path'])
    selected = complete or parts
    if not selected:
        raise RuntimeError(f'No published checkpoint files for {views} views')
    if complete:
        f = complete[0]
        download(f"{ENDPOINT}/HorizonRobotics/Uni3R/resolve/main/{f['path']}",
                 output, f.get('size'), f.get('lfs', {}).get('oid'))
    else:
        total = sum(f['size'] for f in parts)
        if output.exists() and output.stat().st_size == total:
            continue
        downloaded = []
        for f in parts:
            downloaded.append(download(
                f"{ENDPOINT}/HorizonRobotics/Uni3R/resolve/main/{f['path']}",
                ROOT/'checkpoints/parts'/Path(f['path']).name,
                f['size'], f.get('lfs', {}).get('oid')))
        import shutil
        tmp = output.with_suffix('.assembling')
        with tmp.open('wb') as out:
            for part in downloaded:
                with part.open('rb') as src:
                    shutil.copyfileobj(src, out, 8 << 20)
        if tmp.stat().st_size != total:
            raise RuntimeError(f'Checkpoint assembly failed: {tmp}')
        tmp.rename(output)
download(f'{ENDPOINT}/facebook/VGGT-1B/resolve/main/model.pt',
         ROOT/'cache/torch/hub/checkpoints/model.pt')
lseg = ROOT/'checkpoints/demo_e200.ckpt'
if not lseg.exists():
    import gdown
    out = gdown.download(id='1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7', output=str(lseg)+'.part', quiet=False, resume=True)
    if out is None:
        raise RuntimeError('LSeg checkpoint download failed')
    Path(out).rename(lseg)
print('Assets prepared:', ROOT, flush=True)
