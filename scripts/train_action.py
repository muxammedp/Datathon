import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.action_model import VERSION, decode_video, make_model, prepare_clips, predict_windows
from scripts.train_temporal import dataset_items, metrics


def choose_threshold(y, scores):
    choices = []
    for threshold in np.unique(np.r_[.5, scores, scores.max() + 1e-5]):
        m = metrics(y, scores >= threshold)
        loss = (2 * m['false_positive_rate'] + 1 - m['recall']) / 3
        choices.append((loss, -m['recall'], float(threshold)))
    return min(choices)[2]


def evaluate(model, clips, indices):
    scores, timestamps = [], []
    for i in indices:
        centers, s = predict_windows(model, *clips[i])
        scores.append(float(s.max()))
        timestamps.append(float(centers[s.argmax()]))
    return np.asarray(scores), timestamps


def train(items, clips, train_indices, validation_indices, output, seed, epochs=24, fixed_threshold=None):
    import torch
    from safetensors.torch import save_file
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(pretrained=True).cuda()
    optimizer = torch.optim.AdamW([
        {'params': [p for name, p in model.named_parameters() if not name.startswith('fc.')], 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 3e-4},
    ], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.BCEWithLogitsLoss()
    best = None
    history = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        order = np.tile(train_indices, 4)
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), 4):
            rows = order[start:start + 4]
            batch = []
            for i in rows:
                frames, times = clips[i]
                if items[i]['label']:
                    center = float(rng.choice(items[i]['event_times'])) + rng.uniform(-1., 1.)
                else:
                    center = rng.uniform(0., float(times[-1]))
                tensor = prepare_clips(frames, [center])
                tensor = tensor * float(rng.uniform(.95, 1.05))
                batch.append(tensor)
            x = torch.cat(batch)
            y = torch.tensor([items[i]['label'] for i in rows], device='cuda', dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x).flatten()
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        record = {'epoch': epoch, 'loss': float(np.mean(losses)), 'elapsed_sec': round(time.monotonic() - started, 2)}
        if len(validation_indices) and epoch % 4 == 0:
            y = np.asarray([items[i]['label'] for i in validation_indices])
            scores, _ = evaluate(model, clips, validation_indices)
            threshold = choose_threshold(y, scores)
            m = metrics(y, scores >= threshold)
            objective = (2 * m['false_positive_rate'] + 1 - m['recall']) / 3
            record.update(validation=m, threshold=threshold)
            if best is None or objective < best['objective']:
                best = {'objective': objective, 'epoch': epoch, 'threshold': threshold}
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output))
        history.append(record)
        print(json.dumps({'training_run': output.stem, **record}), flush=True)
    if not len(validation_indices):
        best = {'epoch': epochs, 'threshold': fixed_threshold}
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output))
    output.with_suffix('.history.json').write_text(json.dumps({'best': best, 'epochs': history}, indent=2), encoding='utf8')
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(output)))
    return model.eval(), best


def main():
    import torch
    from sklearn.model_selection import StratifiedKFold, train_test_split
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    directory = ROOT / 'data' / 'action-training'
    directory.mkdir(parents=True, exist_ok=True)
    items = dataset_items(args.dataset)
    clips = []
    for i, item in enumerate(items):
        cache = directory / (item['sha256'] + '.npz')
        if not cache.exists():
            frames, times, _ = decode_video(item['path'])
            np.savez_compressed(cache, frames=frames, times=times)
        with np.load(cache, allow_pickle=False) as data:
            clips.append((data['frames'].copy(), data['times'].copy()))
        print(json.dumps({'decoded': i + 1, 'total': len(items)}), flush=True)
    y = np.array([i['label'] for i in items])
    all_indices = np.arange(len(items))
    records, fold_results = [], []
    for fold, (outer_train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(all_indices, y)):
        train_indices, calibration = train_test_split(outer_train, test_size=.25, stratify=y[outer_train], random_state=9100 + fold)
        output = directory / f'fold-{fold}.safetensors'
        if output.exists() and output.with_suffix('.history.json').exists():
            from safetensors.torch import load_file
            model = make_model().cuda()
            model.load_state_dict(load_file(str(output)))
            best = json.loads(output.with_suffix('.history.json').read_text())['best']
        else:
            model, best = train(items, clips, train_indices, calibration, output, 4200 + fold, args.epochs)
        scores, timestamps = evaluate(model, clips, test)
        predicted = scores >= best['threshold']
        m = metrics(y[test], predicted)
        fold_results.append({'fold': fold, 'best': best, 'metrics': m,
                             'train_sha256': [items[i]['sha256'] for i in train_indices],
                             'calibration_sha256': [items[i]['sha256'] for i in calibration],
                             'test_sha256': [items[i]['sha256'] for i in test]})
        for i, s, p, t in zip(test, scores, predicted, timestamps):
            records.append({'name': items[i]['name'], 'sha256': items[i]['sha256'], 'label': int(y[i]),
                            'predicted': int(p), 'score': float(s), 'threshold': best['threshold'],
                            'timecode_sec': t, 'fold': fold})
        (directory / 'oof_predictions.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({'finished_fold': fold, **m}), flush=True)
        del model
        torch.cuda.empty_cache()
    final_epochs = int(np.median([f['best']['epoch'] for f in fold_results]))
    final_threshold = float(np.median([f['best']['threshold'] for f in fold_results]))
    report = {'version': VERSION, 'method': '5 disjoint outer SHA256-deduplicated video folds. Inner 25% calibration for epoch and threshold. No outer labels used in fitting or calibration.',
              'unique_videos': len(items), 'metrics': metrics([r['label'] for r in records], [r['predicted'] for r in records]),
              'folds': fold_results, 'final_epochs': final_epochs, 'threshold': final_threshold,
              'full_backbone_finetuned': True, 'pretrained_source': 'torchvision R3D_18 KINETICS400_V1',
              'binary_labels_only': True, 'category_labels': False, 'actor_labels': False,
              'independent_camera_test': False, 'independent_dataset': False,
              'training_sha256': [i['sha256'] for i in items]}
    (directory / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({'out_of_fold_metrics': report['metrics']}), flush=True)
    model_dir = ROOT / 'models' / 'forte-action-v1'
    model_dir.mkdir(exist_ok=True)
    model, _ = train(items, clips, all_indices, [], model_dir / 'model.safetensors', 4300,
                     final_epochs, final_threshold)
    (model_dir / 'metadata.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    training_scores, _ = evaluate(model, clips, all_indices)
    (directory / 'training_fit.json').write_text(json.dumps({'NOT_VALIDATION': True,
        'metrics': metrics(y, training_scores >= final_threshold)}, indent=2), encoding='utf8')
    print(json.dumps({'trained_model': str(model_dir), 'final_epochs': final_epochs}), flush=True)


if __name__ == '__main__':
    main()
