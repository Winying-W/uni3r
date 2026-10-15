"""Read camera calibration from public ZIPs using HTTP ranges, not full ZIPs."""
import concurrent.futures
import argparse
import bisect
import io
import json
from pathlib import Path
import os
import struct
import time
import urllib.request
import zipfile
import zlib
from download_reproduction import api, ROOT

REPO = 'maksimko123/scannet'
DEST = Path(os.environ.get('DATA_ROOT', ROOT / 'data')) / 'scannet/calibration'


def fetch(url, start, length):
    req = urllib.request.Request(url + f'?range_start={start}&range_size={length}',
                                 headers={'Range': f'bytes={start}-{start + length - 1}'})
    for attempt in range(4):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError(f'Range request returned {response.status}')
                result = response.read()
            if len(result) != length:
                raise RuntimeError(f'Short range: {len(result)} != {length}')
            return result
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


class RemoteFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        if size <= 0:
            return b''
        data = fetch(self.url, self.pos, size)
        self.pos += len(data)
        return data


def extract_bytes(url, size, info):
    blob = fetch(url, info.header_offset, min(info.compress_size + 4096, size - info.header_offset))
    if blob[:4] != b'PK\x03\x04':
        raise RuntimeError(f'Bad ZIP header: {info.filename}')
    n, extra = struct.unpack_from('<HH', blob, 26)
    start = 30 + n + extra
    compressed = blob[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise RuntimeError(f'Unsupported ZIP compression: {info.compress_type}')
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise RuntimeError(f'ZIP CRC mismatch: {info.filename}')
    return data


def extract_info(url, size, info):
    target = DEST / info.filename
    if target.exists():
        return info.filename
    data = extract_bytes(url, size, info)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return info.filename


def image_size(url, size, info):
    from PIL import Image
    target = DEST / info.filename.split('/')[0] / 'image_size.json'
    if target.exists():
        return
    with Image.open(io.BytesIO(extract_bytes(url, size, info))) as image:
        dimensions = image.size
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'width': dimensions[0], 'height': dimensions[1],
                                 'source_image': info.filename}))


def supplement(scene):
    """Read a missing scene's calibration from a second, split-ZIP mirror."""
    from PIL import Image
    repo = 'the-Lin/scannet_hf'
    parts = sorted([x for x in api(f'datasets/{repo}/tree/main')
                    if x['path'].startswith('scans_frame.zip.part_')], key=lambda x: x['path'])
    offsets = [0]
    for part in parts:
        offsets.append(offsets[-1] + part['size'])

    class SplitRemoteFile(RemoteFile):
        def __init__(self):
            super().__init__('', offsets[-1])

        def read(self, size=-1):
            size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
            blocks = []
            while size > 0:
                idx = bisect.bisect_right(offsets, self.pos) - 1
                length = min(size, offsets[idx + 1] - self.pos)
                url = f'https://huggingface.co/datasets/{repo}/resolve/main/{parts[idx]["path"]}'
                blocks.append(fetch(url, self.pos - offsets[idx], length))
                self.pos += length
                size -= length
            return b''.join(blocks)

    with zipfile.ZipFile(SplitRemoteFile()) as archive:
        files = [i for i in archive.infolist() if f'/{scene}/' in i.filename]
        intrinsic = next(i for i in files if i.filename.endswith('/intrinsic.txt'))
        sample = next(i for i in files if i.filename.lower().endswith('.jpg'))
        calibration = archive.read(intrinsic)
        with Image.open(io.BytesIO(archive.read(sample))) as image:
            width, height = image.size
    dest = DEST / scene
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'intrinsic.txt').write_bytes(calibration)
    (dest / 'image_size.json').write_text(json.dumps({'width': width, 'height': height,
        'source_image': sample.filename, 'repository': repo}))
    (dest / 'supplement_source.json').write_text(json.dumps({'repository': repo,
        'intrinsic_file': intrinsic.filename, 'zip_crc_verified': True}, indent=2))
    print('Supplemented', scene, width, height, calibration.decode(), flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    report = []
    for item in api(f'datasets/{REPO}/tree/main'):
        if not item['path'].endswith('.zip'):
            continue
        url = f'https://huggingface.co/datasets/{REPO}/resolve/main/{item["path"]}'
        print('Reading ZIP directory:', item['path'], flush=True)
        with zipfile.ZipFile(RemoteFile(url, item['size'])) as archive:
            entries = archive.infolist()
            infos = [i for i in entries if i.filename.endswith('/intrinsic.txt')]
            first_images = {}
            for info in entries:
                if info.filename.lower().endswith('.jpg'):
                    first_images.setdefault(info.filename.split('/')[0], info)
        print('Calibration files:', len(infos), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(extract_info, url, item['size'], i) for i in infos]
            for f in concurrent.futures.as_completed(futures):
                report.append({'archive': item['path'], 'file': f.result()})
        (DEST / 'source.json').write_text(json.dumps({'repository': REPO, 'files': report}, indent=2))
        print('Checking native image sizes:', len(first_images), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(image_size, url, item['size'], info) for info in first_images.values()]
            for f in concurrent.futures.as_completed(futures):
                f.result()
    print('Fetched calibration files:', len(report), flush=True)
    if not (DEST / 'scene0533_00/intrinsic.txt').exists():
        supplement('scene0533_00')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--supplement-scene')
    args = parser.parse_args()
    supplement(args.supplement_scene) if args.supplement_scene else main()
