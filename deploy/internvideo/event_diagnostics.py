import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path, PurePosixPath

from evaluate_dataset import check_dataset, file_hash, save_json
from training_protocol import make_folds, metrics
from video_windows import annotated_windows, plan_windows

ROOT = Path(__file__).resolve().parent
EXTRACTOR_FILES = ('analyze.py', 'lora_binary.py', 'event_cache.py', 'video_windows.py')


def _json(path):
    return json.loads(path.read_text(encoding='utf8'))


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid finite number: {name}')
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'Invalid integer: {name}')
    return value


def _inside(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Path escapes the expected directory')
    return path


def _rows(rows):
    identities = set()
    for row in rows:
        identity = row.get('sha256')
        name = row.get('file', '')
        relative = PurePosixPath(name.replace('\\', '/'))
        if (not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity)
                or identity in identities or type(row.get('label')) is not int or row['label'] not in (0, 1)
                or not name or relative.is_absolute() or '..' in relative.parts or ':' in name):
            raise ValueError('Invalid or duplicate dataset identity/path')
        identities.add(identity)


def _header(path):
    with path.open('rb') as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError('Truncated safetensors header')
        size = struct.unpack('<Q', raw)[0]
        if not 2 <= size <= 1024 * 1024:
            raise ValueError('Invalid safetensors header size')
        header = json.loads(source.read(size))
    if set(header) - {'__metadata__'} != {'embeds'}:
        raise ValueError('Cache must contain only embeds')
    entry = header['embeds']
    shape = entry.get('shape', [])
    width = {'BF16': 2, 'F16': 2, 'F32': 4}.get(entry.get('dtype'))
    if (not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1
            or any(type(i) is not int or i <= 0 for i in shape) or width is None):
        raise ValueError('Invalid cached embedding shape or dtype')
    payload = math.prod(shape) * width
    if entry.get('data_offsets') != [0, payload] or path.stat().st_size != 8 + size + payload:
        raise ValueError('Truncated or malformed cached embeddings')
    return shape


def _validate_windows(row, windows, settings):
    if not isinstance(windows, list) or not windows:
        raise ValueError('Every video needs a nonempty window plan')
    end_frame = _integer(windows[-1]['end_frame_exclusive'], 'last frame', 1)
    end_sec = _number(windows[-1]['end_sec'], 'last time')
    if end_sec <= 0:
        raise ValueError('Invalid video duration')
    fps = end_frame / end_sec
    detail_budget = _integer(settings['detail_frames'], 'detail budget', 4)
    history_budget = _integer(settings['history_frames'], 'history budget')
    if detail_budget % 4 or history_budget % 4 or detail_budget + history_budget > 256:
        raise ValueError('Invalid cached frame budgets')
    expected = plan_windows(end_frame, fps, settings['window_sec'], settings['overlap_sec'],
                            settings['fps'], detail_budget)
    if len(expected) != len(windows):
        raise ValueError('Window count does not match locked sampling settings')
    for number, (window, original) in enumerate(zip(windows, expected)):
        key = f'{row["sha256"]}-{number}'
        if window.get('cache_key') != key:
            raise ValueError('Cache key does not match video identity and window index')
        for field in ('index', 'start_frame', 'end_frame_exclusive'):
            if _integer(window[field], field) != original[field]:
                raise ValueError('Window bounds do not match locked sampling settings')
        for field in ('start_sec', 'end_sec'):
            if not math.isclose(_number(window[field], field), original[field], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError('Window times do not match frame bounds')
        history, detail = window.get('history_indices'), window.get('detail_indices')
        if not isinstance(history, list) or not isinstance(detail, list) or not detail:
            raise ValueError('Missing history or detail frame plan')
        for indices, budget in ((history, history_budget), (detail, detail_budget)):
            if (len(indices) > budget or len(indices) % 4
                    or any(type(i) is not int or i < 0 for i in indices) or indices != sorted(indices)):
                raise ValueError('Invalid chronological selected frame indices')
        if (any(i >= original['start_frame'] for i in history)
                or any(not original['start_frame'] <= i < original['end_frame_exclusive'] for i in detail)
                or window.get('frame_indices') != history + detail
                or not set(original['frame_indices']).issubset(detail)):
            raise ValueError('Selected frame indices violate locked window bounds or sampling density')
    annotated_windows(row, windows)


def load_verified_cache(run, dataset, rows, root=ROOT):
    run, dataset, root = Path(run).resolve(), Path(dataset).resolve(), Path(root).resolve()
    protocol_path = _inside(run, 'protocol.json')
    protocol = _json(protocol_path)
    if protocol.get('version') != 'internvideo_events_v1':
        raise ValueError('Expected an internvideo_events_v1 source run')
    if protocol.get('dataset_manifest_sha256') != file_hash(dataset / 'manifest.json'):
        raise ValueError('Dataset manifest checksum mismatch')
    if _json(dataset / 'manifest.json').get('videos') != rows:
        raise ValueError('Rows do not match the ordered dataset manifest')
    _rows(rows)
    revision = _json(root / 'model-manifest.json')['revision']
    if protocol.get('model_revision') != revision:
        raise ValueError('Model revision mismatch')
    current = {name: file_hash(root / name) for name in EXTRACTOR_FILES}
    if any(protocol.get('code_sha256', {}).get(name) != digest for name, digest in current.items()):
        raise ValueError('Frozen input extractor changed; existing cache cannot be reused')
    settings = protocol['settings']
    if protocol.get('splits') != make_folds(rows, settings['seed']):
        raise ValueError('Source protocol split identities do not match the dataset')
    cache = _inside(run, 'window-cache')
    index_path = _inside(cache, 'index.json')
    index = _json(index_path)
    windows, answer_ids = index.get('windows'), index.get('answer_ids')
    if (not isinstance(windows, list) or len(windows) != len(rows)
            or not isinstance(answer_ids, list) or len(answer_ids) != 2
            or any(type(i) is not int or not 0 <= i < 1000000 for i in answer_ids)
            or answer_ids[0] == answer_ids[1]):
        raise ValueError('Invalid cache index or answer token IDs')
    digests, hidden_size, total = [], None, 0
    for row, plan in zip(rows, windows):
        _validate_windows(row, plan, settings)
        for window in plan:
            path = _inside(cache, window['cache_key'] + '.safetensors')
            metadata_path = _inside(cache, window['cache_key'] + '.json')
            metadata = _json(metadata_path)
            digest = file_hash(path)
            if (metadata.get('cache_sha256') != digest or metadata.get('answer_ids') != answer_ids
                    or metadata.get('frame_indices') != window['frame_indices']):
                raise ValueError('Cache checksum, answer IDs or selected frames mismatch')
            shape = _header(path)
            if hidden_size is not None and hidden_size != shape[-1]:
                raise ValueError('Cache embedding dimensions are inconsistent')
            hidden_size = shape[-1]
            digests.append((window['cache_key'], digest, file_hash(metadata_path)))
            total += 1
    combined = hashlib.sha256(json.dumps(digests, separators=(',', ':')).encode()).hexdigest()
    provenance = dict(source_protocol_sha256=file_hash(protocol_path), source_index_sha256=file_hash(index_path),
                      dataset_manifest_sha256=protocol['dataset_manifest_sha256'], model_revision=revision,
                      extractor_sha256=current, input_files_sha256=combined, windows=total,
                      videos=len(rows), hidden_size=hidden_size, source_settings=settings)
    return cache, windows, answer_ids, provenance


def audit_reports(rows, reports, seed=123):
    _rows(rows)
    splits = make_folds(rows, seed)
    lookup = {row['sha256']: row for row in rows}
    folds, seen, records, selected_epochs = set(), set(), [], {}
    for report in reports:
        fold = _integer(report['fold'], 'fold')
        if fold >= len(splits) or fold in folds:
            raise ValueError('Duplicate or invalid fold')
        folds.add(fold)
        split = splits[fold]
        for name, indices in split.items():
            if report.get(name + '_sha256') != [rows[i]['sha256'] for i in indices]:
                raise ValueError('Report split identities do not match the expected held-out split')
        selected = _integer(report['best']['epoch'], 'selected epoch')
        selected_epochs[str(fold)] = selected
        predictions = report['predictions']
        expected = {rows[i]['sha256'] for i in split['test']}
        if len(predictions) != len(expected) or {p.get('sha256') for p in predictions} != expected:
            raise ValueError('Predictions do not cover the held-out test split exactly once')
        for prediction in predictions:
            identity = prediction['sha256']
            if identity in seen:
                raise ValueError('Duplicate held-out prediction')
            seen.add(identity)
            row = lookup[identity]
            score = _number(prediction['score'], 'score')
            threshold = _number(prediction['threshold'], 'threshold')
            label, decision = prediction['label'], prediction['prediction']
            if (type(label) is not int or type(decision) is not int or label != row['label']
                    or decision != int(score >= threshold)):
                raise ValueError('Prediction label or threshold decision mismatch')
            if threshold != _number(report['best']['threshold'], 'best threshold'):
                raise ValueError('Prediction threshold does not match selected calibration threshold')
            count = _integer(prediction['windows_processed'], 'window count', 1)
            intervals = []
            for interval in prediction['candidate_intervals']:
                start = _number(interval['start_sec'], 'candidate start')
                end = _number(interval['end_sec'], 'candidate end')
                value = _number(interval['score'], 'candidate score')
                if start < 0 or end <= start or value < threshold:
                    raise ValueError('Invalid candidate interval or score')
                intervals.append(dict(start_sec=start, end_sec=end, score=value))
            if len(intervals) > count or bool(intervals) != bool(decision):
                raise ValueError('Candidate intervals disagree with video decision')
            outcome = ('tp' if decision else 'fn') if label else ('fp' if decision else 'tn')
            times = [_number(t, 'event time') for t in row.get('event_times', [])]
            record = dict(sha256=identity, file=row['file'], aliases=row.get('aliases', []), fold=fold,
                          selected_epoch=selected, label=label, prediction=decision, outcome=outcome,
                          score=score, threshold=threshold, margin=score - threshold,
                          event_times=times, candidate_intervals=intervals, windows_processed=count,
                          annotation_hit=any(c['start_sec'] <= t < c['end_sec'] for c in intervals for t in times))
            if 'window_scores' in prediction:
                values = prediction['window_scores']
                if (not isinstance(values, list) or len(values) != count
                        or max(_number(v, 'window score') for v in values) != score):
                    raise ValueError('Per-window scores disagree with video score')
                record['window_scores'] = values
            if 'frozen_score' in prediction:
                record['frozen_score'] = _number(prediction['frozen_score'], 'frozen score')
                record['frozen_threshold'] = _number(prediction['frozen_threshold'], 'frozen threshold')
                record['frozen_prediction'] = int(record['frozen_score'] >= record['frozen_threshold'])
            if 'frozen_window_scores' in prediction:
                values = prediction['frozen_window_scores']
                if (not isinstance(values, list) or len(values) != count
                        or max(_number(v, 'frozen window score') for v in values) != record.get('frozen_score')):
                    raise ValueError('Frozen window scores disagree with frozen video score')
                record['frozen_window_scores'] = values
            if 'window_traces' in prediction:
                traces = prediction['window_traces']
                if not isinstance(traces, list) or len(traces) != count:
                    raise ValueError('Window traces do not match processed count')
                clean = []
                for number, trace in enumerate(traces):
                    start, end, value = (_number(trace[k], k) for k in ('start_sec', 'end_sec', 'score'))
                    if start < 0 or end <= start or trace.get('index', number) != number:
                        raise ValueError('Invalid window trace')
                    clean.append(dict(index=number, start_sec=start, end_sec=end, score=value))
                if max(t['score'] for t in clean) != score:
                    raise ValueError('Window traces disagree with video score')
                if ('window_scores' in record and [t['score'] for t in clean] != record['window_scores']
                        or intervals != [{k: t[k] for k in ('start_sec', 'end_sec', 'score')}
                                         for t in clean if t['score'] >= threshold]):
                    raise ValueError('Window traces disagree with candidates or window scores')
                record['window_traces'] = clean
            records.append(record)
    records.sort(key=lambda r: (r['fold'], r['sha256']))
    counts = {key: sum(r['outcome'] == key for r in records) for key in ('tn', 'fp', 'fn', 'tp')}
    result = metrics([r['label'] for r in records], [r['margin'] for r in records], 0.) if records else None
    return dict(complete=len(seen) == len(rows) and len(folds) == 5, videos_evaluated=len(records),
                expected_videos=len(rows), counts=counts, metrics=result, selected_epochs=selected_epochs,
                records=records, errors=[r for r in records if r['outcome'] in ('fp', 'fn')],
                per_window_scores_available=bool(records) and all('window_scores' in r or 'window_traces' in r for r in records),
                production_approved=False, independent_new_dataset=False,
                note='Numeric held-out diagnostic only. File aliases are display identifiers, never training features. '
                     'Inspecting these errors informs development and does not create a new independent test set.')


def audit_existing_run(run, dataset, output, root=ROOT):
    run, dataset, output = Path(run).resolve(), Path(dataset).resolve(), Path(output).resolve()
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError('Audit output must be separate from the source run')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Audit output is not empty; choose a new output directory')
    rows = check_dataset(dataset)['videos']
    _, _, _, provenance = load_verified_cache(run, dataset, rows, root)
    reports = []
    for fold in range(5):
        path = _inside(run, f'fold-{fold}/result.json')
        if path.exists():
            saved = _json(path)
            if saved.get('fold') != fold:
                raise ValueError('Fold directory and report identity mismatch')
            adapter = _inside(run, f'fold-{fold}/adapter.safetensors')
            if saved.get('adapter_sha256') != file_hash(adapter):
                raise ValueError('Adapter checksum mismatch')
            reports.append(saved)
    result = audit_reports(rows, reports, provenance['source_settings']['seed'])
    result['source_cache'] = provenance
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'audit.json', result)
    save_json(output / 'errors.json', result['errors'])
    return result


def main():
    parser = argparse.ArgumentParser(description='Verify an event cache and export numeric held-out error diagnostics')
    parser.add_argument('--dataset', type=Path, default=Path('open-dataset'))
    parser.add_argument('--run', type=Path, default=Path('results/events-v2'))
    parser.add_argument('--output', type=Path, default=Path('results/events-v2-audit'))
    args = parser.parse_args()
    result = audit_existing_run(args.run, args.dataset, args.output)
    print(json.dumps(dict(stage='audit_complete', complete=result['complete'], counts=result['counts'],
                          metrics=result['metrics'], output=str(args.output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
