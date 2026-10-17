import os
import argparse
from pathlib import Path

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--size', choices=('2B', '8B'), default='2B')
args = parser.parse_args()
name = f'Qwen3-VL-{args.size}-Instruct'
revisions = {'2B': '89644892e4d85e24eaac8bacfd4f463576704203', '8B': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'}
snapshot_download(
    'Qwen/' + name,
    revision=revisions[args.size],
    local_dir=root / 'models' / name,
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.md', 'LICENSE*'],
    max_workers=3,
)
print('Model downloaded. Video processing requires no network.')
