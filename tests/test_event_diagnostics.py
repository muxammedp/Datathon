import copy
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'deploy/internvideo'
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('event_diagnostics', SOURCE / 'event_diagnostics.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def rows_fixture():
    return [dict(sha256=f'{i + 1:064x}', file=f'video-{i}.mov', aliases=[f'alias-{i}.mov'],
                 label=int(i < 5), event_times=[.5] if i < 5 else []) for i in range(10)]


@pytest.fixture
def cache(tmp_path):
    rows = rows_fixture()
    root, dataset, run = (tmp_path / name for name in ('code', 'dataset', 'run'))
    for directory in (root, dataset, run):
        directory.mkdir()
    directory = run / 'window-cache'
    directory.mkdir()
    for name in diagnostics.EXTRACTOR_FILES:
        (root / name).write_text(name)
    write_json(root / 'model-manifest.json', {'revision': 'pinned'})
    write_json(dataset / 'manifest.json', {'videos': rows})
    settings = dict(seed=123, window_sec=16., overlap_sec=4., fps=4., detail_frames=96, history_frames=32)
    protocol = dict(version='internvideo_events_v1', settings=settings,
                    dataset_manifest_sha256=diagnostics.file_hash(dataset / 'manifest.json'),
                    model_revision='pinned', splits=diagnostics.make_folds(rows),
                    code_sha256={n: diagnostics.file_hash(root / n) for n in diagnostics.EXTRACTOR_FILES})
    write_json(run / 'protocol.json', protocol)
    windows = []
    for row in rows:
        window = dict(index=0, start_frame=0, end_frame_exclusive=4, start_sec=0., end_sec=1.,
                      frame_indices=[0, 1, 2, 3], detail_indices=[0, 1, 2, 3], history_indices=[],
                      motion_peaks=[], cache_key=row['sha256'] + '-0')
        windows.append([window])
        path = directory / (window['cache_key'] + '.safetensors')
        header = json.dumps({'embeds': {'dtype': 'F32', 'shape': [1, 80, 4],
                                       'data_offsets': [0, 1280]}}).encode()
        path.write_bytes(struct.pack('<Q', len(header)) + header + bytes(1280))
        write_json(path.with_suffix('.json'), dict(cache_sha256=diagnostics.file_hash(path),
                   answer_ids=[300, 312], frame_indices=window['frame_indices']))
    write_json(directory / 'index.json', dict(windows=windows, answer_ids=[300, 312]))
    return run, dataset, rows, root


def test_verified_cache_reuses_only_exact_frozen_inputs_without_importing_gpu(cache):
    run, dataset, rows, root = cache
    before = {p: diagnostics.file_hash(p) for p in run.rglob('*') if p.is_file()}
    directory, windows, answer_ids, provenance = diagnostics.load_verified_cache(run, dataset, rows, root)
    assert directory == run / 'window-cache'
    assert len(windows) == 10 and answer_ids == [300, 312]
    assert provenance['windows'] == 10 and provenance['hidden_size'] == 4
    assert len(provenance['input_files_sha256']) == 64
    assert before == {p: diagnostics.file_hash(p) for p in run.rglob('*') if p.is_file()}


def test_cache_accepts_real_planner_history_and_duplicate_padding(cache):
    from video_windows import add_visual_context
    run, dataset, rows, root = cache
    protocol_path = run / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['settings'].update(window_sec=1., overlap_sec=.5)
    write_json(protocol_path, protocol)
    directory = run / 'window-cache'
    index_path = directory / 'index.json'
    index = json.loads(index_path.read_text())
    windows = add_visual_context(diagnostics.plan_windows(7, 4., seconds=1., overlap=.5), [0.] * 7, 4.)
    assert windows[1]['history_indices'] == [0, 1, 1, 1]
    for row_number, row in enumerate(rows):
        template = directory / (row['sha256'] + '-0.safetensors')
        plan = copy.deepcopy(windows)
        for window in plan:
            window['cache_key'] = row['sha256'] + '-' + str(window['index'])
            path = directory / (window['cache_key'] + '.safetensors')
            path.write_bytes(template.read_bytes())
            write_json(path.with_suffix('.json'), dict(answer_ids=[300, 312],
                       frame_indices=window['frame_indices'], cache_sha256=diagnostics.file_hash(path)))
        index['windows'][row_number] = plan
    write_json(index_path, index)
    _, verified, _, provenance = diagnostics.load_verified_cache(run, dataset, rows, root)
    assert provenance['windows'] == 30 and verified[0][1]['history_indices'] == [0, 1, 1, 1]


@pytest.mark.parametrize('change', ['extractor', 'revision', 'manifest', 'split', 'answer_ids', 'checksum',
                                   'bounds', 'ordering', 'path', 'header', 'missing_video'])
def test_cache_rejects_stale_corrupt_or_misaligned_inputs(cache, change):
    run, dataset, rows, root = cache
    index_path = run / 'window-cache/index.json'
    index = json.loads(index_path.read_text())
    window = index['windows'][0][0]
    tensor = run / 'window-cache' / (window['cache_key'] + '.safetensors')
    if change == 'extractor':
        (root / 'video_windows.py').write_text('changed code')
    elif change == 'revision':
        write_json(root / 'model-manifest.json', {'revision': 'different'})
    elif change == 'manifest':
        rows = list(reversed(rows))
    elif change == 'split':
        path = run / 'protocol.json'
        value = json.loads(path.read_text())
        value['splits'][0]['test'][0] = value['splits'][0]['train'][0]
        write_json(path, value)
    elif change == 'answer_ids':
        path = tensor.with_suffix('.json')
        value = json.loads(path.read_text())
        value['answer_ids'] = [300, 313]
        write_json(path, value)
    elif change == 'checksum':
        with tensor.open('ab') as file:
            file.write(b'wrong')
    elif change == 'header':
        tensor.write_bytes(struct.pack('<Q', 2) + b'{}')
        path = tensor.with_suffix('.json')
        value = json.loads(path.read_text())
        value['cache_sha256'] = diagnostics.file_hash(tensor)
        write_json(path, value)
    else:
        if change == 'bounds':
            window['end_sec'] = .9
            window['detail_indices'][-1] = 9
        elif change == 'ordering':
            window['detail_indices'] = [3, 2, 1, 0]
        elif change == 'path':
            window['cache_key'] = '../../outside'
        elif change == 'missing_video':
            index['windows'].pop()
        write_json(index_path, index)
    with pytest.raises(ValueError):
        diagnostics.load_verified_cache(run, dataset, rows, root)


def reports_fixture(rows):
    reports = []
    for fold, split in enumerate(diagnostics.make_folds(rows)):
        predictions = []
        for index in split['test']:
            row = rows[index]
            decision = 1 - row['label'] if index in (0, 5) else row['label']
            score = 1. if decision else -1.
            candidates = [dict(start_sec=0., end_sec=1., score=score)] if decision else []
            predictions.append(dict(sha256=row['sha256'], label=row['label'], prediction=decision,
                                    score=score, threshold=0., windows_processed=1,
                                    candidate_intervals=candidates, window_scores=[score]))
        reports.append(dict(fold=fold, best=dict(epoch=fold % 3, threshold=0.), predictions=predictions,
                            **{name + '_sha256': [rows[i]['sha256'] for i in indices]
                               for name, indices in split.items()}))
    return reports


def test_audit_joins_aliases_and_events_without_creating_new_predictions():
    rows = rows_fixture()
    reports = reports_fixture(rows)
    before = copy.deepcopy(reports)
    result = diagnostics.audit_reports(rows, reports)
    assert result['complete'] and result['videos_evaluated'] == 10
    assert result['counts'] == dict(tn=4, fp=1, fn=1, tp=4)
    assert result['metrics']['accuracy'] == .8
    assert {r['file'] for r in result['errors']} == {'video-0.mov', 'video-5.mov'}
    assert {r['outcome'] for r in result['errors']} == {'fp', 'fn'}
    assert all(r['aliases'] for r in result['records'])
    assert result['per_window_scores_available']
    assert reports == before
    assert not diagnostics.audit_reports(rows, reports[:1])['complete']


def test_audit_supports_new_traces_and_rejects_inconsistent_candidates():
    rows = rows_fixture()
    reports = reports_fixture(rows)
    for report in reports:
        for prediction in report['predictions']:
            score = prediction['score']
            prediction['window_traces'] = [dict(index=0, start_sec=0., end_sec=1., score=score)]
            prediction.update(frozen_score=score, frozen_threshold=0., frozen_window_scores=[score])
    audit = diagnostics.audit_reports(rows, reports)
    assert all('window_traces' in r and 'frozen_window_scores' in r for r in audit['records'])
    candidate = next(p for report in reports for p in report['predictions'] if p['prediction'])
    candidate['candidate_intervals'][0]['end_sec'] = 2.
    with pytest.raises(ValueError, match='traces disagree'):
        diagnostics.audit_reports(rows, reports)


@pytest.mark.parametrize('change', ['duplicate_fold', 'split', 'duplicate_prediction', 'label',
                                   'decision', 'candidate', 'nan', 'window_scores', 'dataset_path'])
def test_audit_rejects_mixed_splits_and_inconsistent_predictions(change):
    rows = rows_fixture()
    reports = reports_fixture(rows)
    prediction = reports[0]['predictions'][0]
    if change == 'duplicate_fold':
        reports.append(reports[0])
    elif change == 'split':
        reports[0]['test_sha256'][0] = reports[0]['train_sha256'][0]
    elif change == 'duplicate_prediction':
        reports[0]['predictions'][1] = copy.deepcopy(prediction)
    elif change == 'label':
        prediction['label'] = 1 - prediction['label']
    elif change == 'decision':
        prediction['prediction'] = 1 - prediction['prediction']
    elif change == 'candidate':
        prediction['candidate_intervals'] = [dict(start_sec=1., end_sec=0., score=1.)]
    elif change == 'nan':
        prediction['score'] = float('nan')
    elif change == 'window_scores':
        prediction['window_scores'] = [999.]
    elif change == 'dataset_path':
        rows[0]['file'] = '../outside.mov'
    with pytest.raises(ValueError):
        diagnostics.audit_reports(rows, reports)


def test_audit_preserves_source_and_verifies_adapter_provenance(cache, tmp_path, monkeypatch):
    run, dataset, rows, root = cache
    reports = reports_fixture(rows)
    for report in reports:
        directory = run / f'fold-{report["fold"]}'
        directory.mkdir()
        adapter = directory / 'adapter.safetensors'
        adapter.write_bytes(b'opaque adapter test fixture')
        report['adapter_sha256'] = diagnostics.file_hash(adapter)
        write_json(directory / 'result.json', report)
    monkeypatch.setattr(diagnostics, 'check_dataset', lambda path: {'videos': rows})
    with pytest.raises(ValueError, match='separate'):
        diagnostics.audit_existing_run(run, dataset, run / 'audit', root)
    before = {p: diagnostics.file_hash(p) for p in run.rglob('*') if p.is_file()}
    output = tmp_path / 'audit'
    result = diagnostics.audit_existing_run(run, dataset, output, root)
    assert result['complete'] and (output / 'errors.json').exists()
    assert before == {p: diagnostics.file_hash(p) for p in run.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='not empty'):
        diagnostics.audit_existing_run(run, dataset, output, root)
    (run / 'fold-0/adapter.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        diagnostics.audit_existing_run(run, dataset, tmp_path / 'other-audit', root)
