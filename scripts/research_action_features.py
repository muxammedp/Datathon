import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.action_model import make_model, prepare_clips, window_centers
from scripts.train_temporal import dataset_items, select, fit, scores, metrics


def main():
    torch.set_num_threads(4)
    directory = ROOT / 'data' / 'action-feature-research'
    directory.mkdir(exist_ok=True)
    items = dataset_items(Path(sys.argv[1]))
    model = make_model(pretrained=True).cuda().eval()
    model.fc = torch.nn.Identity()
    features, times = [], []
    for index, item in enumerate(items):
        output = directory / (item['sha256'] + '.npz')
        if not output.exists():
            with np.load(ROOT / 'data' / 'action-training' / (item['sha256'] + '.npz'), allow_pickle=False) as data:
                frames = data['frames'].copy()
                centers = window_centers(data['times'])
            vectors = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                for start in range(0, len(centers), 4):
                    x = prepare_clips(frames, centers[start:start + 4])
                    vectors.extend(model(x).float().cpu().numpy())
            vectors = np.asarray(vectors)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-8)
            np.savez_compressed(output, features=vectors, times=centers)
        with np.load(output, allow_pickle=False) as data:
            features.append(data['features'].copy())
            times.append(data['times'].copy())
        print(json.dumps({'features': index + 1}), flush=True)
    del model
    torch.cuda.empty_cache()
    y = np.array([i['label'] for i in items])
    indices = np.arange(len(items))
    predictions = np.zeros(len(items), dtype=int)
    folds = []
    for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(indices, y)):
        c, threshold, _ = select(items, features, times, train, 9000 + fold)
        model = fit(items, features, times, train, c)
        predictions[test] = scores(model, features, test) >= threshold
        folds.append(dict(fold=fold, C=c, threshold=threshold, metrics=metrics(y[test], predictions[test])))
        print(json.dumps(folds[-1]), flush=True)
    report = dict(method='Frozen R3D-18 motion features; same nested video cross-validation as temporal SVM',
                  metrics=metrics(y, predictions), folds=folds)
    (directory / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
