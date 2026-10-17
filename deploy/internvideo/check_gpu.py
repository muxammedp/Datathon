import json
import shutil
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Check the driver before downloading weights.')
gpu = torch.cuda.get_device_properties(0)
if gpu.total_memory < 40 * 1024 ** 3:
    raise SystemExit('This server configuration expects at least 40 GiB GPU memory.')
if shutil.disk_usage('.').free < 45 * 1024 ** 3:
    raise SystemExit('At least 45 GiB of free disk space is required.')
x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
report = {'gpu': gpu.name, 'vram_gib': round(gpu.total_memory / 1024 ** 3, 2),
          'torch': torch.__version__, 'cuda': torch.version.cuda, 'bf16_check': True}
Path('hardware.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report))

