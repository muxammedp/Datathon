import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.engine import LocalAnalyzer
from app.full_video import binary_logit
from scripts.train_temporal import dataset_items, metrics

torch.set_num_threads(4)
items = dataset_items(Path(sys.argv[1]))
directory = ROOT / 'data' / 'full-video-training'
analyzer = LocalAnalyzer()
analyzer.load()
records = []
with torch.inference_mode():
    for index, item in enumerate(items):
        began = time.monotonic()
        inputs = {k: v.cuda() for k, v in load_file(str(directory / (item['sha256'] + '.safetensors'))).items()}
        score = float(binary_logit(analyzer.model, inputs))
        record = {'name': item['name'], 'sha256': item['sha256'], 'label': item['label'],
                  'predicted': int(score >= 0), 'score': score, 'elapsed_sec': round(time.monotonic() - began, 2)}
        records.append(record)
        del inputs
        print(json.dumps({'completed': index + 1, 'label': item['label'], 'predicted': record['predicted'],
                          'elapsed_sec': record['elapsed_sec']}), flush=True)
report = {'method': '2B pretrained, native full-span video in one context, binary answer without generated narration, fixed zero logit threshold',
          'metrics': metrics([r['label'] for r in records], [r['predicted'] for r in records]), 'records': records}
(directory / 'baseline-2b.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(report['metrics']), flush=True)
