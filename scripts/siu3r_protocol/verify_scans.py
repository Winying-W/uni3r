"""Verify extracted SIU3R val scans against their archive listing; report bad / missing.

A scan is OK iff every member path of archives/<scan>.tar.gz exists with the same size.
Usage: python verify_scans.py [--fix]   (--fix deletes bad dirs' .extracted marker so the
downloader re-extracts them)
"""
import json
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

ROOT = '/mnt/workspace/Jensen/project/3DGS/data/siu3r_val_full'
LIST = '/mnt/workspace/Jensen/project/3DGS/data/splits/test_siu3r_val.txt'
for i, a in enumerate(sys.argv):  # optional: --root DIR --list FILE
    if a == '--root':
        ROOT = sys.argv[i + 1]
    if a == '--list':
        LIST = sys.argv[i + 1]
SCANS = open(LIST).read().split()


def check(scan):
    arc = f'{ROOT}/archives/{scan}.tar.gz'
    if not os.path.exists(f'{ROOT}/{scan}/.extracted'):
        return scan, 'not_extracted'
    try:
        with tarfile.open(arc) as t:
            for m in t:
                if not m.isfile():
                    continue
                p = f'{ROOT}/{m.name}'
                if not os.path.exists(p) or os.path.getsize(p) != m.size:
                    return scan, f'bad:{m.name}'
    except Exception as e:  # corrupt archive
        return scan, f'archive_error:{e}'
    return scan, 'ok'


if __name__ == '__main__':
    with ThreadPoolExecutor(16) as ex:
        res = dict(ex.map(check, SCANS))
    bad = {s: r for s, r in res.items() if r != 'ok'}
    print(json.dumps({'ok': sum(r == 'ok' for r in res.values()), 'total': len(SCANS),
                      'not_ok': bad}, indent=1))
    if '--fix' in sys.argv:
        for s, r in bad.items():
            if r.startswith('bad') or r.startswith('archive_error'):
                m = f'{ROOT}/{s}/.extracted'
                if os.path.exists(m):
                    os.remove(m)
