import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.temporal import FEATURE_VERSION, extract_features
from scripts.evaluate import labels_from_xlsx, normalize


def dataset_items(dataset):
    videos = {normalize(p.stem): p for p in (dataset / 'videos').glob('*.mov')}
    groups = {}
    for row in labels_from_xlsx(dataset / 'open_labels.xlsx'):
        path = videos[normalize(row['name'])]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in groups:
            groups[digest] = {'name': row['name'], 'path': str(path), 'sha256': digest,
                              'label': int(row['violation']), 'event_times': []}
        item = groups[digest]
        if item['label'] != int(row['violation']):
            raise ValueError('Conflicting labels for identical videos')
        if row['timecode_sec'] is not None:
            item['event_times'].append(float(row['timecode_sec']))
    return list(groups.values())


def supervised_rows(items, features, times, indices):
    rows, labels, weights = [], [], []
    for i in indices:
        item, x, t = items[i], features[i], times[i]
        if item['label']:
            distances = np.min(np.abs(t[:, None] - np.array(item['event_times'])[None, :]), axis=1)
            mask = distances <= 4.1
            if not mask.any():
                mask[np.argmin(distances)] = True
            x = x[mask]
        rows.extend(x)
        labels.extend([item['label']] * len(x))
        weights.extend([1 / len(x)] * len(x))
    weights = np.asarray(weights)
    weights *= len(weights) / weights.sum()
    return np.asarray(rows), np.asarray(labels), weights


def fit(items, features, times, indices, c):
    from sklearn.svm import SVC
    x, y, weight = supervised_rows(items, features, times, indices)
    classifier = SVC(C=c, kernel='rbf', gamma='scale', class_weight='balanced')
    classifier.fit(x, y, sample_weight=weight)
    return classifier


def scores(classifier, features, indices):
    return np.asarray([classifier.decision_function(features[i]).max() for i in indices])


def metrics(y, predicted):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=float(accuracy_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                false_positive_rate=float(fp / max(1, tn + fp)))


def select(items, features, times, indices, seed):
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray([items[i]['label'] for i in indices])
    folds = list(StratifiedKFold(4, shuffle=True, random_state=seed).split(indices, y))
    choices = []
    for c in (.1, 1., 10.):
        out = np.zeros(len(indices))
        for tr, va in folds:
            classifier = fit(items, features, times, indices[tr], c)
            out[va] = scores(classifier, features, indices[va])
        for threshold in np.unique(np.r_[out, 0., out.max() + .001]):
            m = metrics(y, out >= threshold)
            loss = (2 * m['false_positive_rate'] + (1 - m['recall'])) / 3
            choices.append((loss, -m['recall'], c, float(threshold), m))
    chosen = min(choices, key=lambda x: x[:4])
    return chosen[2], chosen[3], chosen[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'data' / 'temporal-training'
    directory.mkdir(parents=True, exist_ok=True)
    items = dataset_items(args.dataset)
    features, times = [], []
    analyzer = None
    for index, item in enumerate(items):
        cache = directory / (item['sha256'] + '.npz')
        began = time.monotonic()
        if not cache.exists():
            if analyzer is None:
                from app.engine import LocalAnalyzer
                analyzer = LocalAnalyzer()
                analyzer.load()
            x, t, sampled = extract_features(analyzer, Path(item['path']))
            np.savez_compressed(cache, features=x, times=t, version=FEATURE_VERSION)
        with np.load(cache, allow_pickle=False) as data:
            if str(data['version']) != FEATURE_VERSION:
                raise ValueError('Feature cache version mismatch')
            features.append(data['features'].copy())
            times.append(data['times'].copy())
        print(json.dumps({'extracted': index + 1, 'total': len(items),
                          'windows': len(times[-1]), 'elapsed_sec': round(time.monotonic() - began, 2)}), flush=True)
    if analyzer:
        del analyzer
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    safe_items = [{k: v for k, v in item.items() if k != 'path'} for item in items]
    (directory / 'manifest.json').write_text(json.dumps(safe_items, ensure_ascii=False, indent=2), encoding='utf8')
    if args.extract_only:
        return
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray([item['label'] for item in items])
    indices = np.arange(len(items))
    predictions = np.zeros(len(items), dtype=int)
    records, folds = [], []
    for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(indices, y)):
        c, threshold, inner_metrics = select(items, features, times, train, 9000 + fold)
        classifier = fit(items, features, times, train, c)
        s = scores(classifier, features, test)
        predictions[test] = s >= threshold
        result = dict(fold=fold, C=c, threshold=threshold,
                      train_sha256=[items[i]['sha256'] for i in train],
                      test_sha256=[items[i]['sha256'] for i in test],
                      metrics=metrics(y[test], predictions[test]))
        folds.append(result)
        for i, score in zip(test, s):
            records.append(dict(safe_items[i], fold=fold, score=float(score), threshold=threshold,
                                predicted=int(predictions[i]),
                                predicted_time_sec=float(times[i][np.argmax(classifier.decision_function(features[i]))])))
        print(json.dumps({'outer_fold_completed': fold + 1, **result['metrics']}), flush=True)
    c, threshold, inner_metrics = select(items, features, times, indices, 9100)
    classifier = fit(items, features, times, indices, c)
    report = dict(method='Nested 5 outer / 4 inner stratified video folds; SHA256 duplicates grouped; event-window weak supervision; threshold and C selected on inner folds only',
                  feature_version=FEATURE_VERSION, unique_videos=len(items),
                  windows=sum(len(x) for x in features),
                  frozen_backbone='Qwen3-VL-2B-Instruct', trained_component='RBF support vector temporal-window classifier',
                  label_scope='binary + weak timestamp, no category or actor labels',
                  threshold_objective='minimize (2*false_positive_rate + false_negative_rate)/3',
                  independent_dataset=False, unseen_camera_test=False,
                  metrics=metrics(y, predictions), final_C=c, final_threshold=threshold,
                  final_inner_metrics=inner_metrics, folds=folds)
    (directory / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    (directory / 'oof_predictions.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')
    model_dir = ROOT / 'models' / 'temporal-head-v1'
    model_dir.mkdir(exist_ok=True)
    np.savez_compressed(model_dir / 'weights.npz', support_vectors=classifier.support_vectors_.astype(np.float32),
                        dual_coef=classifier.dual_coef_[0], intercept=classifier.intercept_, gamma=np.array([classifier._gamma]))
    metadata = {k: v for k, v in report.items() if k != 'folds'}
    metadata.update(threshold=threshold, training_sha256=[item['sha256'] for item in items],
                    trained_at=time.strftime('%Y-%m-%dT%H:%M:%S'), automatic_category_confirmation=False)
    (model_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    from app.temporal import TemporalHead
    exported = TemporalHead(model_dir)
    maximum_error = max(float(np.max(np.abs(exported.score(x) - classifier.decision_function(x)))) for x in features)
    assert maximum_error < 1e-3, maximum_error
    print(json.dumps({'final_nested_metrics': report['metrics'], 'export_max_error': maximum_error,
                      'model_dir': str(model_dir)}), flush=True)


if __name__ == '__main__':
    main()
