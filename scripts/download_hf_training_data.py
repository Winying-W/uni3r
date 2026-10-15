"""Fetch public RGB-D mirrors; retain manifests and verified source archives."""
import concurrent.futures
import json
from pathlib import Path
import os
from download_reproduction import api, download, ROOT

DATA = Path(os.environ.get('DATA_ROOT', ROOT / 'data'))


def main():
    sources = [
        ('ZzZZCHS/processed_scannet', 'scannet', lambda n: n == 'scannet_2d.tar.gz'),
        ('vctvct321/scannetpp-cut3r-processed', 'scannetpp', lambda n: n.endswith('.tar')),
    ]
    jobs = []
    for repo, folder, select in sources:
        manifest = api(f'datasets/{repo}/tree/main?limit=1000')
        dest = DATA / folder / 'archives'
        dest.mkdir(parents=True, exist_ok=True)
        files = [x for x in manifest if x['type'] == 'file' and select(x['path'])]
        report = {'repository': repo, 'files': files, 'total_bytes': sum(x['size'] for x in files)}
        (dest.parent / 'source_manifest.json').write_text(json.dumps(report, indent=2))
        print(repo, len(files), 'files;', report['total_bytes'], 'bytes', flush=True)
        jobs.extend((repo, 'datasets', x, dest / x['path']) for x in files)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download, *job): job[2]['path'] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append({'file': futures[future], 'error': str(exc)})
                print('FAILED', failures[-1], flush=True)
    (DATA / 'hf_training_download_result.json').write_text(json.dumps({'complete': not failures, 'failed': failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
