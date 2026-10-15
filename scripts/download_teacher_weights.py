"""Download the official teacher models in formats used by teachers_2d.py."""
import concurrent.futures
import json
from pathlib import Path
from download_reproduction import api, download, ROOT


def main():
    root = ROOT / 'checkpoints/teachers'
    sources = [
        ('facebook/sam2.1-hiera-large', 'sam2.1',
         lambda name: name == 'sam2.1_hiera_large.pt'),
        ('IDEA-Research/grounding-dino-base', 'grounding-dino-base',
         lambda name: name.endswith(('.json', '.txt', '.safetensors'))),
    ]
    jobs = []
    for repo, directory, select in sources:
        manifest = api(f'models/{repo}/tree/main')
        dest = root / directory
        dest.mkdir(parents=True, exist_ok=True)
        files = [x for x in manifest if x['type'] == 'file' and select(x['path'])]
        if not files:
            raise RuntimeError(f'No matching official files: {repo}')
        (dest / 'source_manifest.json').write_text(json.dumps(
            {'repository': repo, 'files': files}, indent=2))
        jobs.extend((repo, 'models', item, dest / item['path']) for item in files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lambda job: download(*job), jobs):
            print('Verified teacher asset:', result, flush=True)


if __name__ == '__main__':
    main()
