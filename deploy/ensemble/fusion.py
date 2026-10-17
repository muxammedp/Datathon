import argparse
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from deploy.ensemble.common import digest, exclusive_output, file_hash, load_payload, read_json, save_json
from deploy.internvideo.evaluate_dataset import check_dataset
from deploy.internvideo.training_protocol import make_folds, metrics

VERSION = 'ensemble_nested_fusion_v1'
ACTION_DIMENSIONS = 2048
FEATURE_NAMES = ['action_max', 'action_mean', 'context_max', 'context_mean', 'cash_max', 'cash_mean']


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Nonfinite or nonnumeric {name}')
    return float(value)


def _aligned_windows(action, qwen):
    identity = action.get('video_sha256')
    if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity) or qwen.get('video_sha256') != identity:
        raise ValueError('Experts must refer to the same video SHA256')
    actions, language = action.get('windows'), qwen.get('windows')
    if not isinstance(actions, list) or not actions or not isinstance(language, list) or len(actions) != len(language):
        raise ValueError('Every video needs complete paired expert windows')
    matrix, context, cash = [], [], []
    previous = -1.
    for index, (left, right) in enumerate(zip(actions, language)):
        if type(left.get('index')) is not int or left['index'] != index or right.get('index') != index:
            raise ValueError('Expert window indices are inconsistent')
        start, end = (_finite(left[key], key) for key in ('start_sec', 'end_sec'))
        if start < 0 or end <= start or start <= previous or right.get('start_sec') != start or right.get('end_sec') != end:
            raise ValueError('Expert window time bounds are inconsistent')
        previous = start
        if right.get('status') != 'done':
            raise ValueError('Failed specialist windows cannot be silently dropped or treated as normal')
        vector = np.asarray(left.get('features'), dtype=np.float64)
        if vector.shape != (ACTION_DIMENSIONS,) or not np.isfinite(vector).all():
            raise ValueError('Action features have wrong dimensions or nonfinite values')
        matrix.append(vector)
        context.append(_finite(right.get('context_score'), 'context score'))
        cash.append(_finite(right.get('cash_score'), 'cash score'))
    return np.stack(matrix), np.asarray(context), np.asarray(cash)


def _training_windows(record):
    row, windows = record['row'], record['action']['windows']
    if type(row.get('label')) is not int or row['label'] not in (0, 1):
        raise ValueError('Only binary labels are supported')
    if not row['label']:
        return list(range(len(windows)))
    times = row.get('event_times', [])
    if not times or any(_finite(t, 'event time') < 0 or t >= windows[-1]['end_sec'] for t in times):
        raise ValueError('Positive training video requires valid annotated event times')
    selected = [i for i, window in enumerate(windows)
                if any(window['start_sec'] <= time < window['end_sec'] for time in times)]
    if not selected:
        raise ValueError('No window contains a positive video annotation')
    return selected


def _fit_head(inputs, labels, weights, regularization):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    inputs = np.asarray(inputs, dtype=np.float64)
    labels, weights = np.asarray(labels), np.asarray(weights, dtype=np.float64)
    if inputs.ndim != 2 or len(inputs) != len(labels) or len(weights) != len(labels) or set(labels) != {0, 1}:
        raise ValueError('Classifier fitting requires paired examples from both classes')
    if not np.isfinite(inputs).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Nonfinite features or invalid example weights')
    scaler = StandardScaler().fit(inputs, sample_weight=weights)
    model = LogisticRegression(C=regularization, solver='lbfgs', max_iter=2000, random_state=123)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(inputs), labels, sample_weight=weights)
    if list(model.classes_) != [0, 1]:
        raise ValueError('Unexpected classifier class ordering')
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=model.coef_[0].tolist(),
                intercept=float(model.intercept_[0]), dimensions=inputs.shape[1], C=regularization,
                standardizer='weighted_train_only', classifier='l2_logistic')


def _head_score(head, inputs, dimensions):
    if head.get('dimensions') != dimensions:
        raise ValueError('Predictor dimensions do not match the declared feature schema')
    arrays = [np.asarray(head[name], dtype=np.float64) for name in ('mean', 'scale', 'coef')]
    if any(value.shape != (dimensions,) or not np.isfinite(value).all() for value in arrays) or np.any(arrays[1] <= 0):
        raise ValueError('Invalid serialized predictor coefficients or scales')
    intercept = _finite(head['intercept'], 'classifier intercept')
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != dimensions or not np.isfinite(values).all():
        raise ValueError('Prediction input has incompatible feature dimensions')
    logits = ((values - arrays[0]) / arrays[1]) @ arrays[2] + intercept
    if not np.isfinite(logits).all():
        raise ValueError('Classifier produced nonfinite logits')
    return np.exp(-np.logaddexp(0., -logits))


def fit_action(records, indices):
    features, labels, weights = [], [], []
    for index in indices:
        record = records[index]
        values, _, _ = _aligned_windows(record['action'], record['qwen'])
        selected = _training_windows(record)
        features.extend(values[selected])
        labels.extend([record['row']['label']] * len(selected))
        weights.extend([1. / len(selected)] * len(selected))
    return _fit_head(features, labels, weights, .01)


def _video_features(action_head, record):
    values, context, cash = _aligned_windows(record['action'], record['qwen'])
    action = _head_score(action_head, values, ACTION_DIMENSIONS)
    vector = [float(action.max()), float(action.mean()), float(context.max()), float(context.mean()),
              float(cash.max()), float(cash.mean())]
    return vector, action.tolist()


def _novelty_vector(action, qwen):
    matrix, _, _ = _aligned_windows(action, qwen)
    vector = matrix.mean(axis=0)
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-12 else None


def novelty_reference(records, indices):
    vectors, identities = [], []
    for index in indices:
        record = records[index]
        vector = _novelty_vector(record['action'], record['qwen'])
        if vector is not None:
            vectors.append(vector)
            identities.append(record['row']['sha256'])
    threshold = None
    if len(vectors) >= 2:
        matrix = np.stack(vectors)
        distances = np.clip(1. - matrix @ matrix.T, 0., 2.)
        np.fill_diagonal(distances, np.inf)
        threshold = float(np.quantile(distances.min(axis=1), .95))
    return dict(reference_sha256=identities, vectors=[value.tolist() for value in vectors], threshold=threshold,
                method='Nearest cosine distance to normalized mean action features of outer training videos; '
                       'threshold is the training leave-one-out 95th percentile.',
                heuristic_only=True, new_type_detection_guaranteed=False, automatic_abstention=False)


def _novelty(bundle, action, qwen):
    reference = bundle.get('novelty_reference', {})
    identities, values = reference.get('reference_sha256', []), reference.get('vectors', [])
    if len(identities) != len(values) or len(set(identities)) != len(identities):
        raise ValueError('Invalid novelty reference identities')
    if not set(identities).issubset(set(bundle.get('train_sha256', []))):
        raise ValueError('Novelty reference contains a video outside the training split')
    vector, threshold = _novelty_vector(action, qwen), reference.get('threshold')
    if vector is None or threshold is None or not values:
        return dict(novelty_distance=None, novelty_threshold=threshold, novelty_available=False,
                    distribution_shift_flag=False, novelty_heuristic_only=True)
    threshold = _finite(threshold, 'novelty threshold')
    matrix = np.asarray(values, dtype=np.float64)
    if (matrix.shape != (len(identities), ACTION_DIMENSIONS) or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1., rtol=1e-6, atol=1e-6)
            or not 0 <= threshold <= 2):
        raise ValueError('Invalid normalized novelty references')
    distance = float(np.clip(1. - matrix @ vector, 0., 2.).min())
    return dict(novelty_distance=distance, novelty_threshold=threshold, novelty_available=True,
                distribution_shift_flag=distance > threshold + 1e-12, novelty_heuristic_only=True)


def predict_record(bundle, actionpayload, qwenpayload):
    if bundle.get('version') != VERSION or bundle.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Unsupported fusion predictor feature schema')
    record = dict(action=actionpayload, qwen=qwenpayload)
    vector, action_scores = _video_features(bundle['action_model'], record)
    score = float(_head_score(bundle['meta_model'], [vector], len(FEATURE_NAMES))[0])
    threshold = _finite(bundle['threshold'], 'fusion threshold')
    identity = actionpayload['video_sha256']
    known = {name: identity in bundle.get(name + '_sha256', []) for name in ('train', 'calibration', 'test')}
    review = sorted({reason for window in qwenpayload['windows'] for reason in window.get('review_reasons', [])})
    return dict(score=score, prediction=int(score >= threshold), threshold=threshold,
                window_action_scores=action_scores, features=vector, review_reasons=review,
                specialist_review_required=any(bool(w.get('review_required')) for w in qwenpayload['windows']),
                trained_on_video=known['train'], calibrated_on_video=known['calibration'],
                held_out_in_selected_fold=known['test'], new_to_this_fold=not any(known.values()),
                **_novelty(bundle, actionpayload, qwenpayload),
                automatic_verdict_enabled=False, production_approved=False)


def choose_threshold(labels, scores):
    values = [_finite(score, 'calibration score') for score in scores]
    if not values or len(labels) != len(values):
        raise ValueError('Calibration requires paired labels and scores')
    candidates = sorted(set([0.] + values + [math.nextafter(max(values), math.inf)]))
    options = []
    for threshold in candidates:
        result = metrics(labels, values, threshold)
        if result['recall'] >= .5:
            key = ((result['false_positive_rate'] + 1 - result['recall']) / 2, -result['recall'], abs(threshold))
            options.append((key, threshold, result))
    _, threshold, result = min(options, key=lambda option: option[0])
    return threshold, result


def _candidate_window(action, window_scores):
    if len(window_scores) != len(action['windows']) or not window_scores:
        raise ValueError('Candidate selection requires a score for every action window')
    selected = max(range(len(window_scores)), key=lambda index: _finite(window_scores[index], 'action window score'))
    window = action['windows'][selected]
    return dict(index=selected, start_sec=window['start_sec'], end_sec=window['end_sec'],
                action_score=window_scores[selected], selection='highest_action_score_window',
                verified_violation=False)


def fit_fold(records, split, fold=0, seed=123):
    from sklearn.model_selection import StratifiedKFold
    groups = [set(split[name]) for name in ('train', 'calibration', 'test')]
    if (any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
            or set.union(*groups) != set(range(len(records)))):
        raise ValueError('Outer train, calibration and test indices must be disjoint and cover all videos')
    outer = list(split['train'])
    labels = [records[i]['row']['label'] for i in outer]
    if min(labels.count(0), labels.count(1)) < 3:
        raise ValueError('Inner three-fold stacking requires at least three training videos per class')
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold)
    oof, provenance = {}, []
    for inner_fold, (train_positions, held_positions) in enumerate(inner.split(np.zeros(len(outer)), labels)):
        train_indices = [outer[i] for i in train_positions]
        held_indices = [outer[i] for i in held_positions]
        action_head = fit_action(records, train_indices)
        for index in held_indices:
            if index in oof:
                raise ValueError('Inner out-of-fold feature duplicated')
            oof[index] = _video_features(action_head, records[index])[0]
        provenance.append(dict(fold=inner_fold,
                               train_sha256=[records[i]['row']['sha256'] for i in train_indices],
                               held_out_sha256=[records[i]['row']['sha256'] for i in held_indices]))
    if set(oof) != set(outer):
        raise ValueError('Inner out-of-fold predictions do not cover outer training videos')
    meta = _fit_head([oof[i] for i in outer], labels, [1.] * len(outer), .1)
    action_head = fit_action(records, outer)
    bundle = dict(version=VERSION, feature_names=FEATURE_NAMES, action_model=action_head, meta_model=meta,
                  threshold=0., fold=fold, seed=seed, inner_oof_splits=provenance,
                  **{name + '_sha256': [records[i]['row']['sha256'] for i in indices] for name, indices in split.items()},
                  training_policy='All normal windows; positive windows only when containing an annotation; equal weight per video.',
                  meta_policy='Three-fold video-grouped out-of-fold action predictions on outer training videos only.',
                  threshold_policy='Outer calibration only: minimize balanced error with recall >= 0.5; tie prefers recall.',
                  independent_new_dataset=False, new_types_validated=False, new_angles_validated=False,
                  automatic_verdict_enabled=False, production_approved=False)
    bundle['novelty_reference'] = novelty_reference(records, outer)
    calibration = [predict_record(bundle, records[i]['action'], records[i]['qwen']) for i in split['calibration']]
    cal_labels = [records[i]['row']['label'] for i in split['calibration']]
    threshold, cal_metrics = choose_threshold(cal_labels, [result['score'] for result in calibration])
    bundle['threshold'] = threshold
    component_columns = dict(action=0, context=2, cash=4)
    bundle['component_thresholds'] = {
        name: choose_threshold(cal_labels, [result['features'][column] for result in calibration])[0]
        for name, column in component_columns.items()}
    predictions = []
    for index in split['test']:
        row = records[index]['row']
        result = predict_record(bundle, records[index]['action'], records[index]['qwen'])
        candidate = _candidate_window(records[index]['action'], result['window_action_scores'])
        annotation_hit = bool(row['label']) and any(candidate['start_sec'] <= t < candidate['end_sec']
                                                   for t in row.get('event_times', []))
        components = {name: dict(score=result['features'][column], threshold=bundle['component_thresholds'][name],
                                prediction=int(result['features'][column] >= bundle['component_thresholds'][name]))
                      for name, column in component_columns.items()}
        predictions.append(dict(sha256=row['sha256'], label=row['label'], score=result['score'],
                                threshold=threshold, prediction=result['prediction'], components=components,
                                window_action_scores=result['window_action_scores'],
                                candidate_interval=candidate, candidate_window_contains_annotation=annotation_hit,
                                windows_processed=len(result['window_action_scores']),
                                **{key: result[key] for key in ('distribution_shift_flag', 'novelty_distance',
                                                               'novelty_threshold', 'novelty_available', 'novelty_heuristic_only')},
                                specialist_review_required=result['specialist_review_required'],
                                review_reasons=result['review_reasons']))
    report = dict(fold=fold, calibration_metrics=cal_metrics, predictions=predictions,
                  **{name + '_sha256': bundle[name + '_sha256'] for name in split})
    return bundle, report


def load_records(dataset, cache):
    dataset, cache = Path(dataset).resolve(), Path(cache).resolve()
    rows = check_dataset(dataset)['videos']
    protocol = read_json(cache / 'protocol.json')
    fingerprint = protocol.get('fingerprint')
    if not isinstance(fingerprint, str) or not re.fullmatch('[0-9a-f]{64}', fingerprint):
        raise ValueError('Expert cache needs a valid extraction fingerprint')
    records, hashes = [], []
    for row in rows:
        identity = row['sha256']
        if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity):
            raise ValueError('Invalid video identity')
        directory = (cache / 'videos' / identity).resolve()
        if not directory.is_relative_to(cache):
            raise ValueError('Video cache path escapes cache root')
        action_path, qwen_path = directory / 'action.json', directory / 'qwen.json'
        action, qwen = load_payload(action_path, fingerprint), load_payload(qwen_path, fingerprint)
        media_path = directory / 'media/index.json'
        media = read_json(media_path)
        if (action.get('video_sha256') != identity or qwen.get('video_sha256') != identity
                or media.get('video_sha256') != identity or media.get('coverage_complete') is not True):
            raise ValueError('Media and expert video identity or coverage mismatch')
        _aligned_windows(action, qwen)
        if len(media.get('windows', [])) != len(action['windows']):
            raise ValueError('Expert windows do not cover the complete media plan')
        for left, right in zip(media['windows'], action['windows']):
            if any(left.get(key) != right.get(key) for key in ('index', 'start_sec', 'end_sec')):
                raise ValueError('Expert window differs from cached media plan')
        record = dict(row=row, action=action, qwen=qwen)
        _training_windows(record)
        records.append(record)
        hashes.append(dict(sha256=identity, action=file_hash(action_path), qwen=file_hash(qwen_path), media=file_hash(media_path)))
    return records, dict(cache_fingerprint=fingerprint, extraction_protocol=protocol,
                         cache_protocol_sha256=file_hash(cache / 'protocol.json'),
                         cache_inputs_sha256=digest(hashes), dataset_manifest_sha256=file_hash(dataset / 'manifest.json'))


def _validate_report(report, bundle, rows, split, fold):
    if report.get('fold') != fold or bundle.get('fold') != fold:
        raise ValueError('Saved fold identity mismatch')
    for name, indices in split.items():
        expected = [rows[i]['sha256'] for i in indices]
        if report.get(name + '_sha256') != expected or bundle.get(name + '_sha256') != expected:
            raise ValueError('Saved split identity mismatch')
    predictions = report.get('predictions', [])
    expected = {rows[i]['sha256']: rows[i] for i in split['test']}
    if len(predictions) != len(expected) or {r.get('sha256') for r in predictions} != set(expected):
        raise ValueError('Saved test predictions missing or duplicated')
    for row in predictions:
        score, threshold = _finite(row['score'], 'test score'), _finite(row['threshold'], 'test threshold')
        if row['label'] != expected[row['sha256']]['label'] or threshold != bundle['threshold'] or row['prediction'] != int(score >= threshold):
            raise ValueError('Saved prediction conflicts with its label or calibrated threshold')
        values, candidate = row['window_action_scores'], row['candidate_interval']
        if (not values or row['windows_processed'] != len(values)
                or candidate['index'] != max(range(len(values)), key=lambda i: _finite(values[i], 'action window score'))
                or candidate['action_score'] != values[candidate['index']]):
            raise ValueError('Saved candidate is not the maximum action-score window')
        start, end = _finite(candidate['start_sec'], 'candidate start'), _finite(candidate['end_sec'], 'candidate end')
        if not 0 <= start < end:
            raise ValueError('Saved candidate bounds are invalid')
        expected_hit = bool(row['label']) and any(start <= t < end for t in expected[row['sha256']].get('event_times', []))
        if row['candidate_window_contains_annotation'] != expected_hit:
            raise ValueError('Saved interval annotation-hit diagnostic is inconsistent')
        for name in ('action', 'context', 'cash'):
            value = row['components'][name]
            score = _finite(value['score'], name + ' score')
            threshold = _finite(value['threshold'], name + ' threshold')
            if threshold != bundle['component_thresholds'][name] or value['prediction'] != int(score >= threshold):
                raise ValueError('Saved component threshold mismatch')


def summarize(reports, expected):
    predictions = [row for report in reports for row in report['predictions']]
    if len({row['sha256'] for row in predictions}) != len(predictions):
        raise ValueError('Duplicate video in held-out aggregate')
    def measured(name=None):
        scores = [(row if name is None else row['components'][name]) for row in predictions]
        return metrics([row['label'] for row in predictions], [s['score'] - s['threshold'] for s in scores], 0.)
    positives = sum(row['label'] == 1 for row in predictions)
    interval_hits = sum(row['label'] == 1 and row['prediction'] == 1 and row['candidate_window_contains_annotation']
                        for row in predictions)
    return dict(version=VERSION, complete=len(reports) == 5 and len(predictions) == expected,
                folds_completed=len(reports), expected_videos=expected, held_out_metrics=measured(),
                component_metrics={name: measured(name) for name in ('action', 'context', 'cash')},
                candidate_window_hit_recall=interval_hits / positives if positives else None,
                candidate_window_hit_positive_videos=interval_hits, candidate_window_positive_denominator=positives,
                candidate_window_metric_note='A positive held-out video must be predicted positive AND its highest-action-score '
                                             'window must contain an annotation. This is interval-hit recall, not exact timestamp '
                                             'or event-type accuracy. Annotations are never used to select the candidate.',
                errors=[dict(sha256=row['sha256'], label=row['label'], prediction=row['prediction'],
                             score=row['score'], threshold=row['threshold'],
                             outcome='fp' if row['prediction'] else 'fn')
                        for row in predictions if row['label'] != row['prediction']],
                independent_new_dataset=False, dataset_reused_during_development=True,
                new_types_validated=False, new_angles_validated=False, actor_roles_verified=False,
                automatic_verdict_enabled=False, production_approved=False,
                method='Nested video-level stacking: train-only action features and inner OOF meta inputs; calibration-only thresholds.',
                note='Repeated development on this dataset limits independence. New videos are required to establish generalization. '
                     'Five held-out predictors must not be ensembled for evaluating these same training videos.')


def main():
    import sklearn
    parser = argparse.ArgumentParser(description='Train and evaluate a nested fusion model from local expert caches')
    parser.add_argument('--dataset', type=Path, default=Path('../internvideo/open-dataset'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', '--outputdir', dest='output', type=Path, default=Path('results/fusion-v1'))
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-folds', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.max_folds <= 5:
        parser.error('max-folds must be between one and five')
    if args.dataset.is_file():
        if args.dataset.name != 'manifest.json':
            parser.error('Dataset file must be manifest.json')
        args.dataset = args.dataset.parent
    output, cache = args.output.resolve(), args.cache.resolve()
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        parser.error('Output and expert cache must be separate directories')
    records, provenance = load_records(args.dataset, cache)
    rows = [record['row'] for record in records]
    splits = make_folds(rows, args.seed)
    protocol = dict(version=VERSION, provenance=provenance, seed=args.seed, splits=splits,
                    feature_names=FEATURE_NAMES, action_dimensions=ACTION_DIMENSIONS,
                    action_C=.01, meta_C=.1, inner_folds=3, sklearn=sklearn.__version__, numpy=np.__version__,
                    code_sha256={name: file_hash(ROOT / name) for name in ('fusion.py', 'common.py')},
                    split_code_sha256=file_hash(ROOT.parent / 'internvideo/training_protocol.py'))
    with exclusive_output(output):
        path = output / 'protocol.json'
        if path.exists():
            if read_json(path) != protocol:
                raise ValueError('Fusion protocol changed; choose a new output directory')
        elif any(p.name != '.ensemble.lock' for p in output.iterdir()):
            raise ValueError('Nonempty fusion output has no protocol')
        else:
            save_json(path, protocol)
        reports = []
        for fold, split in enumerate(splits[:args.max_folds]):
            directory = output / f'fold-{fold}'
            result_path, predictor_path = directory / 'result.json', directory / 'predictor.json'
            if result_path.exists():
                report, bundle = read_json(result_path), read_json(predictor_path)
                if report.get('predictor_sha256') != file_hash(predictor_path):
                    raise ValueError('Saved fusion predictor checksum mismatch')
            else:
                bundle, report = fit_fold(records, split, fold, args.seed)
                bundle['fingerprint'] = provenance['cache_fingerprint']
                bundle['extraction_protocol'] = provenance['extraction_protocol']
                bundle['training_provenance'] = provenance
                save_json(predictor_path, bundle)
                report['predictor_sha256'] = file_hash(predictor_path)
                save_json(result_path, report)
            _validate_report(report, bundle, rows, split, fold)
            reports.append(report)
            summary = summarize(reports, len(rows))
            save_json(output / 'summary.json', summary)
            save_json(output / 'metrics.json', dict(summary, folds=reports))
            print(__import__('json').dumps(dict(stage='fusion_fold_complete', fold=fold + 1,
                                               held_out_metrics=summary['held_out_metrics'])), flush=True)
        print(__import__('json').dumps(dict(stage='finished', **summary)), flush=True)


if __name__ == '__main__':
    main()
