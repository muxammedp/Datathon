import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'deploy/ensemble/fusion.py'
spec = importlib.util.spec_from_file_location('ensemble_fusion', SOURCE)
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)


def records_fixture():
    records = []
    for index in range(10):
        identity, label = f'{index + 1:064x}', int(index < 5)
        action, qwen = [], []
        for window in range(2):
            features = np.ones(fusion.ACTION_DIMENSIONS)
            features[:4] = [label * 2 - 1 + index * .03, index * .1, window * .2 + .1, index % 3]
            start, end = (0., 16.) if window == 0 else (12., 20.)
            action.append(dict(index=window, start_sec=start, end_sec=end, features=features.tolist()))
            qwen.append(dict(index=window, start_sec=start, end_sec=end, status='done',
                             context_score=label * 2 - 1 + index * .05, cash_score=label - .5,
                             review_required=True, review_reasons=['cashier_role_unknown']))
        row = dict(sha256=identity, file=f'video-{index}.mov', label=label,
                   event_times=[.5] if label else [])
        records.append(dict(row=row, action=dict(video_sha256=identity, windows=action),
                            qwen=dict(video_sha256=identity, windows=qwen)))
    return records


def test_action_training_uses_only_annotated_positive_windows_and_equal_video_mass(monkeypatch):
    records = records_fixture()
    captured = {}

    def fit(inputs, labels, weights, regularization):
        captured.update(inputs=np.asarray(inputs), labels=labels, weights=weights, C=regularization)
        return {'captured': True}

    monkeypatch.setattr(fusion, '_fit_head', fit)
    assert fusion.fit_action(records, [0, 5]) == {'captured': True}
    assert captured['labels'] == [1, 0, 0]
    assert captured['weights'] == [1., .5, .5] and captured['C'] == .01
    np.testing.assert_array_equal(captured['inputs'][0], records[0]['action']['windows'][0]['features'])
    records[0]['row']['event_times'] = [100.]
    with pytest.raises(ValueError, match='event times'):
        fusion.fit_action(records, [0, 5])


def test_nested_fit_never_uses_outer_calibration_or_test_inputs_in_training(monkeypatch):
    records = records_fixture()
    split = fusion.make_folds([r['row'] for r in records])[0]
    calls, original = [], fusion.fit_action

    def tracked(records, indices):
        calls.append(list(indices))
        return original(records, indices)

    monkeypatch.setattr(fusion, 'fit_action', tracked)
    bundle, report = fusion.fit_fold(records, split)
    assert len(calls) == 4 and set(calls[-1]) == set(split['train'])
    assert all(set(indices) <= set(split['train']) for indices in calls)
    held = []
    for inner in bundle['inner_oof_splits']:
        assert not set(inner['train_sha256']) & set(inner['held_out_sha256'])
        assert set(inner['train_sha256'] + inner['held_out_sha256']) == set(bundle['train_sha256'])
        held.extend(inner['held_out_sha256'])
    assert sorted(held) == sorted(bundle['train_sha256'])
    assert set(bundle['novelty_reference']['reference_sha256']) <= set(bundle['train_sha256'])
    changed = copy.deepcopy(records)
    for index in split['calibration'] + split['test']:
        for window in changed[index]['action']['windows']:
            window['features'] = (np.arange(fusion.ACTION_DIMENSIONS) + 1000).tolist()
        for window in changed[index]['qwen']['windows']:
            window['context_score'] = 999.
            window['cash_score'] = -999.
    other, _ = fusion.fit_fold(changed, split)
    assert other['action_model'] == bundle['action_model']
    assert other['meta_model'] == bundle['meta_model']
    assert other['novelty_reference'] == bundle['novelty_reference']
    assert {r['sha256'] for r in report['predictions']} == set(bundle['test_sha256'])


def test_json_predictor_roundtrip_and_unseen_feature_distance(tmp_path):
    records = records_fixture()
    split = fusion.make_folds([r['row'] for r in records])[0]
    bundle, _ = fusion.fit_fold(records, split)
    path = tmp_path / 'predictor.json'
    fusion.save_json(path, bundle)
    loaded = fusion.read_json(path)
    record = records[split['test'][0]]
    assert fusion.predict_record(bundle, record['action'], record['qwen']) == fusion.predict_record(
        loaded, record['action'], record['qwen'])
    assert fusion.predict_record(bundle, record['action'], record['qwen'])['held_out_in_selected_fold']
    action, qwen = copy.deepcopy(record['action']), copy.deepcopy(record['qwen'])
    action['video_sha256'] = qwen['video_sha256'] = 'f' * 64
    for window in action['windows']:
        window['features'] = [-value for value in window['features']]
    result = fusion.predict_record(loaded, action, qwen)
    assert result['new_to_this_fold'] and result['distribution_shift_flag']
    assert result['novelty_available'] and result['novelty_heuristic_only']
    assert not result['automatic_verdict_enabled'] and result['specialist_review_required']
    assert len(result['window_action_scores']) == 2
    assert 0 <= result['score'] <= 1


def test_test_labels_cannot_change_fitted_predictors_or_calibration_threshold():
    records = records_fixture()
    split = fusion.make_folds([r['row'] for r in records])[0]
    before, _ = fusion.fit_fold(records, split)
    for index in split['test']:
        records[index]['row']['label'] = 1 - records[index]['row']['label']
    after, _ = fusion.fit_fold(records, split)
    assert after == before


@pytest.mark.parametrize('change', ['identity', 'failed', 'missing_window', 'nan', 'dimension', 'bounds'])
def test_missing_or_bad_expert_inputs_are_not_dropped(change):
    record = records_fixture()[0]
    if change == 'identity':
        record['qwen']['video_sha256'] = 'f' * 64
    elif change == 'failed':
        record['qwen']['windows'][0]['status'] = 'failed'
    elif change == 'missing_window':
        record['qwen']['windows'].pop()
    elif change == 'nan':
        record['qwen']['windows'][0]['cash_score'] = float('nan')
    elif change == 'dimension':
        record['action']['windows'][0]['features'].pop()
    elif change == 'bounds':
        record['qwen']['windows'][0]['end_sec'] += 1
    with pytest.raises(ValueError):
        fusion._aligned_windows(record['action'], record['qwen'])


def test_threshold_uses_balanced_error_with_recall_floor():
    threshold, measured = fusion.choose_threshold([0, 0, 1, 1], [.1, .2, .8, .9])
    assert measured['accuracy'] == 1 and measured['fp'] == measured['fn'] == 0
    assert .2 < threshold <= .8
    _, result = fusion.choose_threshold([0, 0, 1, 1], [.9, .8, .1, .2])
    assert result['recall'] >= .5


def test_split_overlap_and_reference_leakage_are_rejected():
    records = records_fixture()
    split = fusion.make_folds([r['row'] for r in records])[0]
    bad = copy.deepcopy(split)
    bad['train'].append(bad['test'][0])
    with pytest.raises(ValueError, match='disjoint'):
        fusion.fit_fold(records, bad)
    bundle, _ = fusion.fit_fold(records, split)
    bundle['novelty_reference']['reference_sha256'][0] = bundle['test_sha256'][0]
    with pytest.raises(ValueError, match='outside the training'):
        fusion.predict_record(bundle, records[0]['action'], records[0]['qwen'])


def test_full_outer_evaluation_counts_every_video_once():
    records = records_fixture()
    rows = [r['row'] for r in records]
    reports = []
    for fold, split in enumerate(fusion.make_folds(rows)):
        bundle, report = fusion.fit_fold(records, split, fold)
        fusion._validate_report(report, bundle, rows, split, fold)
        reports.append(report)
    summary = fusion.summarize(reports, len(records))
    assert summary['complete'] and summary['held_out_metrics']['total'] == 10
    assert summary['candidate_window_positive_denominator'] == 5
    assert set(summary['component_metrics']) == {'action', 'context', 'cash'}
    assert not summary['independent_new_dataset'] and not summary['new_types_validated']
    with pytest.raises(ValueError, match='Duplicate'):
        fusion.summarize(reports + [reports[0]], len(records))


def test_candidate_interval_hit_requires_positive_decision_and_correct_window():
    action = records_fixture()[0]['action']
    selected = fusion._candidate_window(action, [.1, .9])
    assert selected['index'] == 1 and selected['start_sec'] == 12.
    predictions = []
    for index, (label, decision, interval_hit) in enumerate([(1, 1, False), (1, 0, True), (1, 1, True), (0, 0, False)]):
        score = .9 if decision else .1
        value = dict(score=score, threshold=.5, prediction=decision)
        predictions.append(dict(sha256=f'{index:064x}', label=label, **value,
                                candidate_window_contains_annotation=interval_hit,
                                components={name: value for name in ('action', 'context', 'cash')}))
    result = fusion.summarize([{'predictions': predictions}], 4)
    assert result['candidate_window_positive_denominator'] == 3
    assert result['candidate_window_hit_positive_videos'] == 1
    assert result['candidate_window_hit_recall'] == pytest.approx(1 / 3)


def test_cache_requires_matching_envelopes_and_all_video_inputs(tmp_path, monkeypatch):
    from deploy.ensemble.common import save_payload
    records = records_fixture()
    rows = [r['row'] for r in records]
    dataset, cache = tmp_path / 'dataset', tmp_path / 'cache'
    dataset.mkdir()
    fusion.save_json(dataset / 'manifest.json', {'videos': rows})
    fingerprint = 'a' * 64
    fusion.save_json(cache / 'protocol.json', {'fingerprint': fingerprint, 'settings': {}})
    for record in records:
        directory = cache / 'videos' / record['row']['sha256']
        save_payload(directory / 'action.json', record['action'], fingerprint)
        save_payload(directory / 'qwen.json', record['qwen'], fingerprint)
        fusion.save_json(directory / 'media/index.json', dict(video_sha256=record['row']['sha256'],
                         coverage_complete=True, windows=[{key: w[key] for key in ('index', 'start_sec', 'end_sec')}
                                                          for w in record['action']['windows']]))
    monkeypatch.setattr(fusion, 'check_dataset', lambda path: {'videos': rows})
    loaded, provenance = fusion.load_records(dataset, cache)
    assert loaded == records and provenance['cache_fingerprint'] == fingerprint
    path = cache / 'videos' / rows[-1]['sha256'] / 'qwen.json'
    value = fusion.read_json(path)
    value['payload']['windows'][0]['cash_score'] += 1
    fusion.save_json(path, value)
    with pytest.raises(ValueError, match='checksum'):
        fusion.load_records(dataset, cache)


def test_cli_saves_json_predictor_resumes_and_rejects_changed_weights(tmp_path, monkeypatch, capsys):
    records = records_fixture()
    fingerprint = 'a' * 64
    provenance = dict(cache_fingerprint=fingerprint, extraction_protocol={'fingerprint': fingerprint, 'settings': {}},
                      cache_protocol_sha256='b' * 64, cache_inputs_sha256='c' * 64,
                      dataset_manifest_sha256='d' * 64)
    monkeypatch.setattr(fusion, 'load_records', lambda dataset, cache: (records, provenance))
    output = tmp_path / 'output'
    monkeypatch.setattr(fusion.sys, 'argv', ['fusion.py', '--dataset', str(tmp_path / 'dataset'),
                                          '--cache', str(tmp_path / 'cache'), '--output', str(output), '--max-folds', '1'])
    fusion.main()
    saved = fusion.read_json(output / 'fold-0/predictor.json')
    assert saved['fingerprint'] == fingerprint and saved['extraction_protocol'] == provenance['extraction_protocol']
    assert 'action_model' in saved and 'meta_model' in saved and not saved['production_approved']
    before = (output / 'fold-0/predictor.json').read_bytes()
    fusion.main()
    assert (output / 'fold-0/predictor.json').read_bytes() == before
    assert not fusion.read_json(output / 'summary.json')['complete']
    assert not (output / '.ensemble.lock').exists()
    saved['action_model']['coef'][0] += .1
    fusion.save_json(output / 'fold-0/predictor.json', saved)
    with pytest.raises(ValueError, match='checksum'):
        fusion.main()
    assert not (output / '.ensemble.lock').exists()
    assert 'finished' in capsys.readouterr().out


def test_serialized_head_matches_weighted_sklearn_predictions():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    values = np.asarray([[0., 1.], [1., 3.], [4., 2.], [5., 6.]])
    labels, weights = [0, 0, 1, 1], [1., .5, .5, 1.]
    head = fusion._fit_head(values, labels, weights, .01)
    scaler = StandardScaler().fit(values, sample_weight=weights)
    model = LogisticRegression(C=.01, solver='lbfgs', max_iter=2000, random_state=123)
    model.fit(scaler.transform(values), labels, sample_weight=weights)
    expected = model.predict_proba(scaler.transform(values))[:, 1]
    np.testing.assert_allclose(fusion._head_score(head, values, 2), expected, rtol=1e-12, atol=1e-12)
