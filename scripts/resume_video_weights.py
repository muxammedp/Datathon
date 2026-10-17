import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'models' / 'Qwen3-VL-4B-Instruct-bnb-4bit'
TARGET = DIRECTORY / 'model.safetensors'
SIZE = 3483890098
SHA256 = 'b57d1194942317e2909d90b152a9fa7b1054c2d9780a66a074c0476b5c78a0e4'
URL = 'https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-bnb-4bit/resolve/79cd853d0bc98bb0d67f865123eb49ef8985c2ec/model.safetensors'
CHUNK = 16 * 1024 * 1024
STAGING = DIRECTORY / 'download-parts'
STAGING.mkdir(exist_ok=True)


def fetch(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    output = STAGING / f'{index:04}.part'
    expected = end - start + 1
    if output.exists() and output.stat().st_size == expected:
        return expected
    for attempt in range(5):
        try:
            with requests.get(URL, params={'download': 'true', 'part': index, 'refresh': int(time.time())},
                              headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(20, 45)) as response:
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise RuntimeError('Incorrect range response')
                with output.with_suffix('.partial').open('wb') as handle:
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
            if output.with_suffix('.partial').stat().st_size != expected:
                raise RuntimeError('Incorrect chunk length')
            os.replace(output.with_suffix('.partial'), output)
            return expected
        except Exception as exc:
            print(f'chunk_retry={index} attempt={attempt + 1} error={type(exc).__name__}', flush=True)
            if attempt == 4:
                raise RuntimeError(f'Failed chunk {index}') from None
    return 0


if TARGET.exists() and hashlib.sha256(TARGET.read_bytes()).hexdigest() == SHA256:
    print('Weights already verified.')
    raise SystemExit(0)
began = time.monotonic()
count = (SIZE + CHUNK - 1) // CHUNK
complete = 0
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(fetch, i) for i in range(count)]
    for future in as_completed(futures):
        complete += future.result()
        print(f'downloaded_mb={complete // 1048576} total_mb={SIZE // 1048576} elapsed_sec={int(time.monotonic() - began)}', flush=True)
digest = hashlib.sha256()
assembled = DIRECTORY / 'model.safetensors.assembling'
with assembled.open('wb') as output:
    for i in range(count):
        with (STAGING / f'{i:04}.part').open('rb') as part:
            while block := part.read(8 * 1024 * 1024):
                output.write(block)
                digest.update(block)
if digest.hexdigest() != SHA256:
    raise RuntimeError('Downloaded weight SHA256 mismatch')
os.replace(assembled, TARGET)
for i in range(count):
    (STAGING / f'{i:04}.part').unlink()
STAGING.rmdir()
print('Complete: pinned weights SHA256 verified.', flush=True)
