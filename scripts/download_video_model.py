import os
import subprocess
import sys
from pathlib import Path

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
snapshot_download(
    'unsloth/Qwen3-VL-4B-Instruct-bnb-4bit',
    revision='79cd853d0bc98bb0d67f865123eb49ef8985c2ec',
    local_dir=root / 'models' / 'Qwen3-VL-4B-Instruct-bnb-4bit',
    allow_patterns=['*.json', '*.txt', '*.jinja', '*.md', 'LICENSE*'],
    max_workers=3,
)
subprocess.run([sys.executable, str(root / 'scripts' / 'resume_video_weights.py')], check=True)
print('Pinned quantized video model downloaded. No dataset is sent to external services.')
