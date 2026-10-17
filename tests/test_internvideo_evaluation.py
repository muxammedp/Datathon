import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1] / 'deploy/internvideo/evaluate_dataset.py'
spec = importlib.util.spec_from_file_location('internvideo_evaluation', script)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_unknown_and_failure_are_not_correct_normal_predictions():
    rows = [{'label': label, 'status': 'done', 'result': {'prediction': prediction}}
            for label, prediction in [(0, 'no_violation'), (0, 'violation'),
                                      (1, 'violation'), (1, 'no_violation'), (0, 'inconclusive')]]
    rows.append({'label': 1, 'status': 'failed'})
    result = evaluation.summarize(rows, 6)
    assert [result[k] for k in ('tn', 'fp', 'fn', 'tp', 'inconclusive', 'failed')] == [1] * 6
    assert result['accuracy_decided'] == .5
    assert result['correct_fraction_all_attempted'] == 2 / 6
    assert result['decision_coverage'] == 4 / 6
    assert result['recall_all_positive_videos'] == 1 / 3


def test_dataset_rejects_duplicate_video_identity(tmp_path):
    path = tmp_path / 'fixture.mov'
    path.write_bytes(b'synthetic-only')
    row = {'file': path.name, 'label': 0, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / 'manifest.json').write_text(json.dumps({'videos': [row, row]}))
    with pytest.raises(ValueError, match='Duplicate'):
        evaluation.check_dataset(tmp_path)


def test_dataset_rejects_paths_outside_dataset(tmp_path):
    row = {'file': '../outside.mov', 'label': 0, 'sha256': '0' * 64}
    (tmp_path / 'manifest.json').write_text(json.dumps({'videos': [row]}))
    with pytest.raises(ValueError, match='Invalid'):
        evaluation.check_dataset(tmp_path)


def test_retry_preserves_success_and_exposes_original_failure(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    rows = []
    for index in range(3):
        path = dataset / f'{index}.mov'
        path.write_bytes(f'synthetic-{index}'.encode())
        rows.append({'file': path.name, 'label': index % 2,
                     'sha256': evaluation.file_hash(path), 'aliases': []})
    (dataset / 'manifest.json').write_text(json.dumps({'videos': rows}))
    output = tmp_path / 'results'
    calls = []

    def failing_analyze(model, tokenizer, path, fps, maximum):
        calls.append(path.name)
        if path.name == '1.mov':
            raise RuntimeError('synthetic allocation failure')
        return {'prediction': 'no_violation'}

    stub = types.SimpleNamespace(load_model=lambda: (None, None), analyze=failing_analyze)
    monkeypatch.setitem(sys.modules, 'analyze', stub)
    arguments = ['evaluate_dataset.py', '--dataset', str(dataset), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', arguments)
    with pytest.raises(RuntimeError, match='synthetic allocation failure'):
        evaluation.main()
    records = json.loads((output / 'records.json').read_text())
    assert [row['status'] for row in records] == ['done', 'failed']
    assert 'synthetic allocation failure' in (output / records[1]['error_log']).read_text()
    first_result = records[0]

    def successful_analyze(model, tokenizer, path, fps, maximum):
        calls.append(path.name)
        return {'prediction': 'violation'}

    stub.analyze = successful_analyze
    monkeypatch.setattr(sys, 'argv', arguments + ['--retry-failed'])
    evaluation.main()
    records = json.loads((output / 'records.json').read_text())
    assert calls == ['0.mov', '1.mov', '1.mov', '2.mov']
    assert records[0] == first_result
    assert len(records) == 3
    assert all(row['status'] == 'done' for row in records)
    assert json.loads((output / 'metrics.json').read_text())['failed'] == 0
    assert list(output.glob('error-*.log'))
