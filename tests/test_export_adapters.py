import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

SOURCE = Path(__file__).resolve().parents[1] / 'deploy/internvideo'
sys.path.insert(0, str(SOURCE))
SPEC = importlib.util.spec_from_file_location('export_adapters', SOURCE / 'export_adapters.py')
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def adapter_tensors():
    return {f'model.layers.31.attention.{projection}.{parameter}': np.zeros(shape, dtype=np.float32)
            for projection in ('wqkv', 'wo') for parameter, shape in
            [('a', (1, 4096)), ('b', (6144 if projection == 'wqkv' else 4096, 1))]}


@pytest.fixture
def bundle(tmp_path):
    run, dataset, root = (tmp_path / name for name in ('run', 'dataset', 'code'))
    for directory in (run, dataset, root):
        directory.mkdir()
    rows = []
    for i in range(10):
        video = dataset / f'video-{i}.mov'
        video.write_bytes(f'opaque fixture {i}'.encode())
        rows.append(dict(sha256=exporter.file_hash(video), file=video.name, label=int(i < 5),
                         event_times=[.5] if i < 5 else []))
    write_json(dataset / 'manifest.json', {'videos': rows})
    write_json(root / 'model-manifest.json', dict(repo=exporter.BASE_MODEL, revision='pinned'))
    for name in ('lora_binary.py', 'event_learning.py'):
        (root / name).write_text('fixture code')
    settings = dict(seed=123, rank=1, last_layers=1, window_sec=16., overlap_sec=4., fps=4.,
                    detail_frames=96, history_frames=32)
    protocol = dict(version='internvideo_events_v1', settings=settings, model_revision='pinned',
                    splits=exporter.make_folds(rows), dataset_manifest_sha256=exporter.file_hash(dataset / 'manifest.json'),
                    code_sha256={name: exporter.file_hash(root / name) for name in ('lora_binary.py', 'event_learning.py')})
    write_json(run / 'protocol.json', protocol)
    reports = []
    for fold, split in enumerate(protocol['splits']):
        directory = run / f'fold-{fold}'
        directory.mkdir()
        adapter = directory / 'adapter.safetensors'
        save_file(adapter_tensors(), str(adapter))
        predictions = []
        for index in split['test']:
            row = rows[index]
            decision = 1 - row['label'] if index in (0, 5) else row['label']
            score = 1. if decision else -1.
            candidates = [dict(start_sec=0., end_sec=1., score=score)] if decision else []
            predictions.append(dict(sha256=row['sha256'], label=row['label'], prediction=decision,
                                    score=score, threshold=0., frozen_score=score, frozen_threshold=0.,
                                    windows_processed=1, candidate_intervals=candidates))
        result = dict(fold=fold, best=dict(epoch=fold % 3, threshold=0.), predictions=predictions,
                      adapter_sha256=exporter.file_hash(adapter),
                      **{name + '_sha256': [rows[i]['sha256'] for i in indices] for name, indices in split.items()})
        reports.append(result)
        write_json(directory / 'result.json', result)
    audit = exporter.audit_reports(rows, reports)
    summary = dict(complete=True, completed_folds=5, dataset_total=len(rows), unique_test_videos=len(rows),
                   held_out_metrics=audit['metrics'], matched_frozen_metrics=audit['metrics'],
                   epochs_with_selected_training=[r['best']['epoch'] for r in reports], timestamp_hit_recall=.8,
                   production_approved=False, automatic_verdict_enabled=False)
    write_json(run / 'summary.json', summary)
    write_json(run / 'metrics.json', dict(summary, folds=reports))
    (run / 'unrelated.key').write_text('excluded fixture')
    (run / 'window-cache').mkdir()
    (run / 'window-cache/private-cache.npz').write_bytes(b'excluded fixture')
    return run, dataset, root


@pytest.mark.parametrize('version', ['internvideo_events_v1', 'internvideo_events_v3'])
def test_export_all_five_adapters_with_honest_metadata_and_no_training_data(bundle, tmp_path, version):
    run, dataset, root = bundle
    protocol = exporter._json(run / 'protocol.json')
    protocol['version'] = version
    if version.endswith('v3'):
        protocol['input_cache'] = dict(source_settings=dict(protocol['settings']))
    write_json(run / 'protocol.json', protocol)
    before = {str(p.relative_to(run)): exporter.file_hash(p) for p in run.rglob('*') if p.is_file()}
    output = tmp_path / 'exports/adapters.zip'
    result = exporter.build_bundle(run, dataset, output, root)
    assert result['adapters'] == 5 and result['held_out_metrics']['accuracy'] == .8
    assert result['sha256'] == exporter.file_hash(output)
    assert not result['production_approved'] and not result['uploaded'] and not result['single_final_model']
    with zipfile.ZipFile(output) as archive:
        expected = {'protocol.json', 'metrics.json', 'summary.json', 'MODEL_INFO.json', 'README.txt'}
        expected.update(f'fold-{i}/{name}' for i in range(5) for name in ('adapter.safetensors', 'result.json'))
        assert set(archive.namelist()) == expected
        info = json.loads(archive.read('MODEL_INFO.json'))
        assert info['held_out_metrics']['accuracy'] == .8
        assert not info['single_final_model'] and not info['ensemble_evaluated']
        assert not info['base_model']['weights_included']
        assert [r['fold'] for r in info['folds']] == list(range(5))
        assert info['folds'][0]['selected_epoch'] == 0 and not info['folds'][0]['selected_training']
        assert info['loading']['export_code_matches_training']
        if version.endswith('v3'):
            assert info['loading']['binary_head'] == 'FP32 event_learning.binary_head_logits'
            assert not info['loading']['v3_supported_by_legacy_analyze_windows']
            assert b'does not support V3' in archive.read('README.txt')
        for name, entry in info['contents'].items():
            assert exporter.hashlib.sha256(archive.read(name)).hexdigest() == entry['sha256']
    assert before == {str(p.relative_to(run)): exporter.file_hash(p) for p in run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['incomplete', 'summary_metrics', 'summary_only', 'split', 'label',
                                   'adapter_hash', 'missing_adapter', 'revision', 'dataset', 'model_type',
                                   'production', 'epochs', 'timestamp', 'baseline', 'code_path'])
def test_export_rejects_corrupt_or_misleading_run(bundle, tmp_path, change):
    run, dataset, root = bundle
    protocol = exporter._json(run / 'protocol.json')
    summary = exporter._json(run / 'summary.json')
    metrics = exporter._json(run / 'metrics.json')
    report = exporter._json(run / 'fold-0/result.json')
    if change == 'incomplete':
        summary['complete'] = False
    elif change == 'summary_metrics':
        summary['held_out_metrics']['accuracy'] = .99
    elif change == 'summary_only':
        metrics['held_out_metrics']['accuracy'] = .99
    elif change == 'split':
        protocol['splits'][0]['test'] = protocol['splits'][0]['train']
    elif change == 'label':
        report['predictions'][0]['label'] = 1 - report['predictions'][0]['label']
    elif change == 'adapter_hash':
        with (run / 'fold-0/adapter.safetensors').open('ab') as file:
            file.write(b'changed')
    elif change == 'missing_adapter':
        (run / 'fold-0/adapter.safetensors').unlink()
    elif change == 'revision':
        protocol['model_revision'] = 'different'
    elif change == 'dataset':
        with (dataset / 'video-0.mov').open('ab') as file:
            file.write(b'changed')
    elif change == 'model_type':
        protocol['version'] = 'unknown_model'
    elif change == 'production':
        summary['production_approved'] = True
    elif change == 'epochs':
        summary['epochs_with_selected_training'][0] = 99
    elif change == 'timestamp':
        summary['timestamp_hit_recall'] = 1.
    elif change == 'baseline':
        summary['matched_frozen_metrics']['accuracy'] = .99
    elif change == 'code_path':
        protocol['code_sha256'] = {'../outside.py': 'a' * 64}
    if change != 'summary_only':
        metrics.update(summary)
    write_json(run / 'protocol.json', protocol)
    write_json(run / 'summary.json', summary)
    write_json(run / 'metrics.json', metrics)
    write_json(run / 'fold-0/result.json', report)
    with pytest.raises((ValueError, FileNotFoundError)):
        exporter.build_bundle(run, dataset, tmp_path / 'bad.zip', root)
    assert not (tmp_path / 'bad.zip').exists()


@pytest.mark.parametrize('change', ['unexpected_parameter', 'nonfinite', 'dtype', 'dimensions', 'malformed'])
def test_adapter_payload_must_be_valid_native_lora_even_with_matching_checksum(bundle, tmp_path, change):
    run, dataset, root = bundle
    path = run / 'fold-0/adapter.safetensors'
    values = adapter_tensors()
    key = next(iter(values))
    if change == 'unexpected_parameter':
        values['model.base.weight'] = np.zeros((1, 1), dtype=np.float32)
    elif change == 'nonfinite':
        values[key][0, 0] = float('nan')
    elif change == 'dtype':
        values[key] = values[key].astype(np.float16)
    elif change == 'dimensions':
        values[key] = np.zeros((1, 2), dtype=np.float32)
    if change == 'malformed':
        path.write_bytes(b'not safetensors')
    else:
        save_file(values, str(path))
    report = exporter._json(run / 'fold-0/result.json')
    report['adapter_sha256'] = exporter.file_hash(path)
    write_json(run / 'fold-0/result.json', report)
    with pytest.raises(ValueError):
        exporter.build_bundle(run, dataset, tmp_path / 'bad-payload.zip', root)


def test_export_does_not_require_current_code_to_equal_old_training_but_records_mismatch(bundle, tmp_path):
    run, dataset, root = bundle
    (root / 'lora_binary.py').write_text('newer checkout')
    output = tmp_path / 'older-run.zip'
    exporter.build_bundle(run, dataset, output, root)
    with zipfile.ZipFile(output) as archive:
        info = json.loads(archive.read('MODEL_INFO.json'))
        assert not info['loading']['export_code_matches_training']
        assert info['loading']['requires_matching_training_code']


def test_export_preserves_existing_output_and_never_writes_inside_sources(bundle, tmp_path):
    run, dataset, root = bundle
    output = tmp_path / 'existing.zip'
    output.write_bytes(b'preserve')
    for bad in (output, run / 'bundle.zip', dataset / 'bundle.zip', tmp_path / 'not-a-zip.txt'):
        with pytest.raises(ValueError):
            exporter.build_bundle(run, dataset, bad, root)
    assert output.read_bytes() == b'preserve'
    assert not (run / 'bundle.zip').exists() and not (dataset / 'bundle.zip').exists()
