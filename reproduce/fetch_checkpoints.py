"""Resolve official links through the local proxy; resume CDN transfers and verify hashes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import requests

ROOT = Path(os.environ.get('UNI3R_STORAGE', '/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R'))
PROXY = os.environ.get('UNI3R_PROXY', 'http://127.0.0.1:37890')
FILES = [f for f in json.loads((ROOT/'checkpoints/manifest.json').read_text())
         if f['path'].startswith(('2views.pth.part_', '8views.pth.part_', '16views.pth.part_'))]
FILES.sort(key=lambda f: ((2,8,16).index(int(f['path'].split('views')[0])), f['path']))
parts = ROOT/'checkpoints/parts'
parts.mkdir(exist_ok=True)

def verify(path, record):
    if not path.exists() or path.stat().st_size != record['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest() == record['lfs']['oid']

def fetch(record):
    output = parts/record['path']
    if verify(output, record):
        print('Already verified', output.name, flush=True)
        return
    temp = output.with_suffix(output.suffix+'.part')
    for attempt in range(12):
        try:
            if verify(temp, record):
                temp.rename(output)
                print('Verified', output.name, flush=True)
                return
            cache_dir = ROOT/'cache/resolved'
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir/(output.name+'.json')
            cached = json.loads(cache_file.read_text()) if cache_file.exists() else {}
            if cached.get('expires', 0) > time.time()+60:
                url = cached['url']
            else:
                response = requests.head('https://huggingface.co/HorizonRobotics/Uni3R/resolve/main/'+record['path'],
                                         proxies={'https': PROXY}, allow_redirects=False, timeout=(20,40))
                response.raise_for_status()
                url = response.headers['Location']
                expires = int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)['Expires'][0])
                cache_file.write_text(json.dumps({'url': url, 'expires': expires}))
            with (ROOT/'logs'/('download_'+output.name+'.log')).open('a') as log:
                proxy_args = ['--proxy', PROXY] if output.name.endswith('part_aa') and attempt % 2 else ['--noproxy', '*']
                result = subprocess.run(['curl', *proxy_args, '-fL', '--connect-timeout', '20',
                                         '--speed-limit', '1024', '--speed-time', '120',
                                         '-C', '-', url, '-o', str(temp)], stdout=log, stderr=log)
            if result.returncode == 0:
                if not verify(temp, record):
                    raise RuntimeError('Downloaded file failed size/SHA256 verification')
                temp.rename(output)
                print('Verified', output.name, flush=True)
                return
            if result.returncode == 22:
                cache_file.unlink(missing_ok=True)
            print('Retry', output.name, attempt+1, 'curl exit', result.returncode, flush=True)
        except (requests.RequestException, KeyError) as error:
            print('Resolve retry', output.name, attempt+1, type(error).__name__, flush=True)
        time.sleep(5)
    raise RuntimeError('Download attempts exhausted: '+output.name)

if __name__ == '__main__':
    # Finish one view setting before starting the next, so inference can start early.
    for views in (2, 8, 16):
        group = [record for record in FILES if record['path'].startswith(f'{views}views.')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(fetch, group))
        print(f'{views}-view checkpoint parts ready', flush=True)
    print('All checkpoint parts verified', flush=True)
