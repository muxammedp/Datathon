import hashlib
import json
import os
from pathlib import Path

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'model-manifest.json').read_text(encoding='utf8'))
directory = ROOT / 'models' / 'InternVideo2_5_Chat_8B'
snapshot_download(manifest['repo'], revision=manifest['revision'], local_dir=directory,
                  allow_patterns=list(manifest['files']), max_workers=2)
for name, expected in manifest['files'].items():
    path = directory / name
    if path.stat().st_size != expected['size']:
        raise RuntimeError(f'Incorrect file size: {name}')
    if expected.get('sha256'):
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected['sha256']:
            raise RuntimeError(f'Incorrect SHA256: {name}')
print('Pinned model files downloaded and checked. No videos were uploaded.')

