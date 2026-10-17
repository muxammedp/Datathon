from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
target = root / 'models' / 'r3d_18-b3b3357e.pth'
target.parent.mkdir(exist_ok=True)
if target.exists():
    import hashlib
    if hashlib.sha256(target.read_bytes()).hexdigest().startswith('b3b3357e'):
        print('Pretrained R3D weights already verified.')
        raise SystemExit(0)
torch.hub.download_url_to_file(
    'https://download.pytorch.org/models/r3d_18-b3b3357e.pth', str(target), hash_prefix='b3b3357e',
)
print('Pretrained weights verified. These are initialization weights, not the trained Forte model.')
