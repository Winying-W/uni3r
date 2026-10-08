"""Verify official split files and atomically assemble a fixed-view checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--views', type=int, choices=(2, 8, 16), required=True)
p.add_argument('--wait_seconds', type=int, default=0)
args = p.parse_args()
root = Path(os.environ.get('UNI3R_STORAGE', '/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R'))
records = sorted([f for f in json.loads((root/'checkpoints/manifest.json').read_text())
                  if f['path'].startswith(f'{args.views}views.pth.part_')], key=lambda f: f['path'])
assert records, 'No official split files found'
paths = [root/'checkpoints/parts'/f['path'] for f in records]
deadline = time.monotonic()+args.wait_seconds
while not all(path.exists() for path in paths):
    if time.monotonic() >= deadline:
        raise RuntimeError('Missing verified checkpoint parts')
    print('Waiting for', ', '.join(path.name for path in paths if not path.exists()), flush=True)
    time.sleep(min(30, max(1, deadline-time.monotonic())))
output = root/f'checkpoints/{args.views}views.pth'
report = output.with_suffix('.manifest.json')
expected = sum(f['size'] for f in records)
if output.exists() and report.exists() and output.stat().st_size == expected:
    print('Already assembled', output, flush=True)
    raise SystemExit(0)
temp = output.with_suffix('.assembling')
combined = hashlib.sha256()
with temp.open('wb') as target:
    for source, record in zip(paths, records):
        assert source.stat().st_size == record['size'], source
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
                combined.update(block)
                target.write(block)
        assert digest.hexdigest() == record['lfs']['oid'], source
        print('Appended and verified', source.name, flush=True)
    target.flush()
    os.fsync(target.fileno())
assert temp.stat().st_size == expected
with zipfile.ZipFile(temp) as archive:
    names = archive.namelist()
    assert any(name.endswith('/data.pkl') for name in names)
temp.rename(output)
report.write_text(json.dumps({'path': str(output), 'size': expected, 'sha256': combined.hexdigest(),
                              'parts': records, 'zip_entries': len(names)}, indent=2))
print('Checkpoint ready:', output, flush=True)
