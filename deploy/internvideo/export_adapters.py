import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path

from event_diagnostics import audit_reports
from evaluate_dataset import check_dataset, file_hash
from training_protocol import make_folds, metrics

ROOT = Path(__file__).resolve().parent
BASE_MODEL = 'OpenGVLab/InternVideo2_5_Chat_8B'
VERSIONS = {'internvideo_events_v1', 'internvideo_events_v3'}


def _inside(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError('Source path escapes the run directory')
    return path


def _json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(path.read_text(encoding='utf8'), parse_constant=reject)


def _same_metrics(actual, expected, name):
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        raise ValueError(name + ' metric fields mismatch')
    for key, value in expected.items():
        candidate = actual[key]
        if (isinstance(candidate, bool) or not isinstance(candidate, (int, float))
                or not math.isfinite(candidate)
                or not math.isclose(candidate, value, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(name + ' metric value mismatch: ' + key)


def _verify_adapter(path, expected_hash, settings):
    import numpy as np
    from safetensors import safe_open
    if path.suffix != '.safetensors' or file_hash(path) != expected_hash:
        raise ValueError('Adapter checksum mismatch')
    rank, layers = settings.get('rank'), settings.get('last_layers')
    if type(rank) is not int or not 1 <= rank <= 64 or type(layers) is not int or not 1 <= layers <= 32:
        raise ValueError('Invalid adapter rank or layer settings')
    expected = {f'model.layers.{layer}.attention.{projection}.{parameter}'
                for layer in range(32 - layers, 32) for projection in ('wqkv', 'wo')
                for parameter in ('a', 'b')}
    try:
        with safe_open(str(path), framework='numpy') as adapter:
            if set(adapter.keys()) != expected:
                raise ValueError('Adapter contains unexpected parameters; only LoRA A/B matrices are allowed')
            for key in sorted(expected):
                tensor = adapter.get_tensor(key)
                if (tensor.dtype != np.float32 or tensor.ndim != 2 or not np.isfinite(tensor).all()
                        or tensor.shape[0 if key.endswith('.a') else 1] != rank):
                    raise ValueError('Invalid adapter tensor dtype, dimensions or values')
                expected_width = 4096 if '.wo.' in key or key.endswith('.a') else 6144
                if tensor.shape[1 if key.endswith('.a') else 0] != expected_width:
                    raise ValueError('Adapter dimensions do not match the pinned InternVideo language model')
    except ValueError:
        raise
    except Exception as error:
        raise ValueError('Unreadable safetensors adapter') from error


def _verify_summary(summary, saved_metrics, reports, audit, rows):
    if summary != {key: value for key, value in saved_metrics.items() if key != 'folds'}:
        raise ValueError('Summary and metrics reports disagree')
    if saved_metrics.get('folds') != reports:
        raise ValueError('Metrics fold reports do not match saved results')
    if (summary.get('complete') is not True or not audit['complete']
            or summary.get('completed_folds') != 5 or summary.get('unique_test_videos') != len(rows)
            or summary.get('dataset_total') != len(rows)):
        raise ValueError('Only a completed five-fold run can be exported')
    if summary.get('production_approved') is not False or summary.get('automatic_verdict_enabled') is not False:
        raise ValueError('Experimental adapters must not claim production approval')
    _same_metrics(summary.get('held_out_metrics'), audit['metrics'], 'Held-out')
    predictions = [prediction for report in reports for prediction in report['predictions']]
    scores = [prediction['frozen_score'] - prediction['frozen_threshold'] for prediction in predictions]
    frozen = metrics([prediction['label'] for prediction in predictions], scores, 0.)
    _same_metrics(summary.get('matched_frozen_metrics'), frozen, 'Frozen baseline')
    expected_epochs = [report['best']['epoch'] for report in reports]
    if summary.get('epochs_with_selected_training') != expected_epochs:
        raise ValueError('Selected epoch summary mismatch')
    positives = [record for record in audit['records'] if record['label']]
    hit_recall = sum(record['annotation_hit'] for record in positives) / len(positives)
    reported_hit = summary.get('timestamp_hit_recall')
    if (isinstance(reported_hit, bool) or not isinstance(reported_hit, (int, float))
            or not math.isfinite(reported_hit) or not math.isclose(reported_hit, hit_recall, abs_tol=1e-12)):
        raise ValueError('Timestamp metric mismatch')


def build_bundle(run, dataset, output, root=ROOT):
    run, dataset, output, root = (Path(path).resolve() for path in (run, dataset, output, root))
    if output.is_relative_to(run) or output.is_relative_to(dataset):
        raise ValueError('Output must be outside the source run and dataset')
    if output.suffix.lower() != '.zip' or output.exists():
        raise ValueError('Choose a new output .zip file; existing files are never overwritten')
    protocol_path = _inside(run, 'protocol.json')
    protocol = _json(protocol_path)
    if protocol.get('version') not in VERSIONS:
        raise ValueError('Unsupported event training protocol')
    manifest = _json(root / 'model-manifest.json')
    if manifest.get('repo') != BASE_MODEL or protocol.get('model_revision') != manifest.get('revision'):
        raise ValueError('Pinned base model or revision mismatch')
    rows = check_dataset(dataset)['videos']
    if protocol.get('dataset_manifest_sha256') != file_hash(dataset / 'manifest.json'):
        raise ValueError('Dataset manifest checksum mismatch')
    seed = protocol['settings']['seed']
    if type(seed) is not int or protocol.get('splits') != make_folds(rows, seed):
        raise ValueError('Protocol split identities do not match the verified dataset')
    sources = {'protocol.json': protocol_path, 'summary.json': _inside(run, 'summary.json'),
               'metrics.json': _inside(run, 'metrics.json')}
    reports = []
    for fold in range(5):
        report_name, adapter_name = f'fold-{fold}/result.json', f'fold-{fold}/adapter.safetensors'
        sources[report_name], sources[adapter_name] = _inside(run, report_name), _inside(run, adapter_name)
        report = _json(sources[report_name])
        if report.get('fold') != fold:
            raise ValueError('Fold directory and report identity mismatch')
        _verify_adapter(sources[adapter_name], report.get('adapter_sha256'), protocol['settings'])
        reports.append(report)
    audit = audit_reports(rows, reports, seed)
    summary, saved_metrics = _json(sources['summary.json']), _json(sources['metrics.json'])
    _verify_summary(summary, saved_metrics, reports, audit, rows)
    recorded_code = protocol.get('code_sha256')
    if not isinstance(recorded_code, dict) or not recorded_code:
        raise ValueError('Training code checksums are missing')
    current_code = {}
    for name, digest in recorded_code.items():
        if (Path(name).name != name or not name.endswith('.py')
                or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError('Invalid training code checksum entry')
        path = _inside(root, name)
        current_code[name] = file_hash(path) if path.is_file() else None
    contents = {name: dict(sha256=file_hash(path), bytes=path.stat().st_size) for name, path in sources.items()}
    v3 = protocol['version'] == 'internvideo_events_v3'
    sampling = protocol.get('input_cache', {}).get('source_settings') if v3 else protocol['settings']
    if not isinstance(sampling, dict) or any(key not in sampling for key in
                                            ('window_sec', 'overlap_sec', 'fps', 'detail_frames', 'history_frames')):
        raise ValueError('Locked video sampling settings are missing')
    info = dict(bundle_format='forte_internvideo_five_fold_adapters_v1',
                base_model=dict(repo=BASE_MODEL, revision=protocol['model_revision'], weights_included=False),
                training_protocol=protocol['version'], experimental=True, production_approved=False,
                automatic_verdict_enabled=False, independent_new_dataset=False,
                single_final_model=False, ensemble_evaluated=False, adapters=5,
                held_out_metrics=audit['metrics'], matched_frozen_metrics=summary['matched_frozen_metrics'],
                timestamp_hit_recall=summary['timestamp_hit_recall'],
                settings=protocol['settings'], sampling_settings={key: sampling[key] for key in
                    ('window_sec', 'overlap_sec', 'fps', 'detail_frames', 'history_frames')},
                folds=[dict(fold=report['fold'], selected_epoch=report['best']['epoch'],
                            threshold=report['best']['threshold'], adapter_sha256=report['adapter_sha256'],
                            selected_training=report['best']['epoch'] > 0) for report in reports],
                loading=dict(adapter_format='native lora_binary.py A/B tensors, not a PEFT adapter',
                             requires_exact_pinned_base=True, requires_matching_training_code=True,
                             binary_head='FP32 event_learning.binary_head_logits' if v3 else 'lora_binary.binary_logits',
                             v3_supported_by_legacy_analyze_windows=False if v3 else None,
                             training_code_sha256=recorded_code, export_checkout_code_sha256=current_code,
                             export_code_matches_training=current_code == recorded_code), contents=contents)
    readme = (
        'ForteVision experimental five-fold adapter archive\n\n'
        f'Base model: {BASE_MODEL}\nPinned revision: {protocol["model_revision"]}\n'
        f'Held-out accuracy: {audit["metrics"]["accuracy"]:.6f}\n'
        f'TN/FP/FN/TP: {audit["counts"]}\n\n'
        'This contains five separately trained cross-validation adapters, not one final model.\n'
        'Every fold uses its own selected epoch and calibration threshold from MODEL_INFO.json.\n'
        'Do not choose a fold using its test accuracy. No five-model ensemble has been evaluated.\n'
        'Epoch zero means the unchanged initial adapter was selected.\n'
        'This is internal evaluation on reused data; production approval is false.\n\n'
        'Loading requirements:\n'
        '1. Install the deployment requirements and obtain the exact pinned base model separately.\n'
        '2. Use deployment Python files matching protocol.json code_sha256 (included as hashes, not code).\n'
        '3. Load the base model; attach_adapters(language_model, last_layers, rank) using saved settings.\n'
        '4. Load one fold adapter.safetensors on CPU and apply lora_binary.restore_adapter. Set eval mode.\n'
        '5. Reproduce the saved sampling, prompt, answer-token IDs and that fold calibration threshold.\n'
        + ('6. V3 requires event_learning.binary_head_logits in FP32, as used by train_events_v3.window_score.\n'
           '   The legacy analyze_windows.py adapter loader does not support V3. Do not use its BF16 head.\n'
           if v3 else
           '6. V1 uses lora_binary.binary_logits and the matching analyze_windows.py --adapter-run loader.\n')
        + '\nIncluded: five adapters, five fold reports, protocol, metrics, summary, MODEL_INFO and this README.\n'
        'Excluded: base weights, videos, frames, embeddings, caches, credentials and training environments.\n'
        'Export is local only. This script does not publish or upload anything.\n'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix='forte-adapters-', suffix='.zip.part', dir=output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in sources.items():
                archive.write(path, name)
            archive.writestr('MODEL_INFO.json', json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False))
            archive.writestr('README.txt', readme)
        with zipfile.ZipFile(temporary) as archive:
            for name, expected in contents.items():
                digest = hashlib.sha256()
                with archive.open(name) as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != expected['sha256']:
                    raise ValueError('Source changed during export: ' + name)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(output=str(output), sha256=file_hash(output), bytes=output.stat().st_size,
                adapters=5, held_out_metrics=audit['metrics'], production_approved=False,
                single_final_model=False, uploaded=False)


def main():
    parser = argparse.ArgumentParser(description='Export all five verified experimental event adapters; no upload')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('open-dataset'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_bundle(args.run, args.dataset, args.output)
    print(json.dumps(dict(stage='adapters_exported', **result), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
