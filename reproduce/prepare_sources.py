"""Fetch source archives when GitHub's git endpoint is unavailable."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

repo = Path(__file__).resolve().parents[1]
storage = Path(os.environ['UNI3R_STORAGE'])
sources = storage/'sources'
sources.mkdir(exist_ok=True)
archives = storage/'cache/source_archives'
archives.mkdir(parents=True, exist_ok=True)
manifest = {}
items = [
    ('vggt', 'facebookresearch/vggt', '059c288227aa00faaf1a4ecf3f4f3be9b40aa467', repo/'submodules/vggt'),
    ('lang_seg', 'isl-org/lang-seg', '65c0a0978fa2e562cda695afed1554c63ec35cb9', repo/'submodules/lang_seg'),
    ('simple-knn', 'camenduru/simple-knn', 'main', sources/'simple-knn'),
    ('CLIP', 'openai/CLIP', 'main', sources/'CLIP'),
]
for name, project, ref, target in items:
    marker = target/'.uni3r_source_ready'
    archive = archives/f'{name}-{ref}.tar.gz'
    url = f'https://codeload.github.com/{project}/tar.gz/{ref}'
    if marker.exists():
        manifest[name] = {'source': marker.read_text().strip()}
        continue
    if not marker.exists():
        if not archive.exists():
            temp = archive.with_suffix('.part')
            subprocess.run(['curl', '--http1.1', '-fL', '--retry-all-errors', '--connect-timeout', '20', '--max-time', '3600',
                            '--retry', '3', url, '-o', str(temp)], check=True)
            temp.rename(archive)
        unpack = archives/f'{name}.unpack'
        unpack.mkdir(exist_ok=True)
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not (unpack/member.name).resolve().is_relative_to(unpack.resolve()) or member.issym() or member.islnk():
                    raise ValueError(member.name)
            tf.extractall(unpack)
        contents = list(unpack.iterdir())
        if len(contents) != 1:
            raise RuntimeError(f'Unexpected archive layout: {contents}')
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(contents[0], target, dirs_exist_ok=True)
        marker.write_text(url)
        shutil.rmtree(unpack)
    manifest[name] = {'url': url, 'sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(archive.read_bytes()).hexdigest()}
(sources/'manifest.json').write_text(json.dumps(manifest, indent=2))
