import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.simple_video.analyzer import DEFAULTS, VERSION, SimpleVideoAnalyzer, prepare_video


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def select_inputs(args):
    if args.video:
        path = args.video.resolve()
        if not path.is_file():
            raise ValueError('Video does not exist')
        return [dict(path=path, sha256=file_hash(path), label=args.expected_label,
                     event_times=args.expected_time or [])]
    dataset = args.dataset.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf8'))
    rows = manifest['videos']
    if len({row['sha256'] for row in rows}) != len(rows):
        raise ValueError('Dataset manifest contains duplicate identities')
    selected = []
    for row in rows[:args.limit]:
        path = (dataset / row['file']).resolve()
        if not path.is_relative_to(dataset) or type(row['label']) is not int or row['label'] not in (0, 1):
            raise ValueError('Invalid dataset record')
        if file_hash(path) != row['sha256']:
            raise ValueError('Video differs from dataset manifest')
        selected.append(dict(path=path, sha256=row['sha256'], label=row['label'],
                             event_times=row.get('event_times', [])))
    if not selected:
        raise ValueError('No videos selected')
    return selected


def combine(clips):
    positives = [i for i, item in enumerate(clips)
                 if item.get('status') == 'done' and item.get('prediction') == 'violation']
    if positives:
        return 'violation', positives[0]
    if not clips or any(item.get('status') != 'done' or item.get('prediction') != 'no_violation' for item in clips):
        return 'insufficient_data', None
    return 'no_violation', None


def comparison(prediction, label, evidence_time, annotations):
    if label is not None and (type(label) is not int or label not in (0, 1)):
        raise ValueError('Expected label must be zero or one')
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in annotations):
        raise ValueError('Invalid annotation timestamp')
    binary = {'violation': 1, 'no_violation': 0}.get(prediction)
    return dict(expected_label=label, predicted_label=binary,
                label_matches=None if label is None or binary is None else label == binary,
                annotated_times_sec=annotations,
                nearest_annotation_error_sec=min(abs(evidence_time - t) for t in annotations)
                if label == 1 and binary == 1 and evidence_time is not None and annotations else None,
                annotation_used_for_inference=False, exact_event_verified=False)


def run(args):
    began = time.perf_counter()
    output, model_path = args.output.resolve(), args.model.resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; previous results are preserved')
    if output.is_relative_to(model_path) or model_path.is_relative_to(output):
        raise ValueError('Output and model directories must be separate')
    rows = select_inputs(args)
    if any(row['path'].is_relative_to(output) for row in rows):
        raise ValueError('Output must be separate from input videos')
    settings = dict(DEFAULTS, sample_fps=args.sample_fps, max_new_tokens=args.max_new_tokens)
    learned = None
    if getattr(args, 'head', None):
        from deploy.simple_video.head import protocol as feature_protocol
        learned = json.loads(args.head.read_text(encoding='utf8'))
        if learned.get('extraction_protocol') != feature_protocol(model_path, settings):
            raise ValueError('Classifier differs from the installed model, source or preparation settings')
    source_hashes = {name: file_hash(ROOT / name) for name in
                     ('deploy/simple_video/run.py', 'deploy/simple_video/analyzer.py', 'app/rules.py')}
    output.mkdir(parents=True)
    model = SimpleVideoAnalyzer(model_path, device=args.device, max_new_tokens=args.max_new_tokens)
    loaded_at = time.perf_counter()
    model.load()
    load_sec = time.perf_counter() - loaded_at
    print(json.dumps(dict(stage='simple_model_loaded', load_sec=round(load_sec, 3))), flush=True)
    records = []
    for ordinal, row in enumerate(rows, 1):
        started = time.perf_counter()
        directory = output / row['sha256']
        directory.mkdir()
        try:
            clips, metadata = prepare_video(row['path'], settings)
            prepare_sec = time.perf_counter() - started
            results = []
            classification = None
            if learned is not None:
                from deploy.simple_video.head import predict
                vector, feature_profile = model.extract_features(clips, metadata)
                classification = dict(predict(learned, vector), profile=feature_profile,
                                      trained_on_video=row['sha256'] in learned.get('train_sha256', []),
                                      calibrated_on_video=row['sha256'] in learned.get('calibration_sha256', []),
                                      held_out_in_exported_fold=row['sha256'] in learned.get('test_sha256', []))
            inspect_clips = clips if classification is None or (
                getattr(args, 'locate', False) and classification['prediction'] == 'violation') else []
            for clip in inspect_clips:
                try:
                    result = model.analyze_clip(clip, metadata)
                except Exception as exc:
                    result = dict(status='failed', prediction='insufficient_data',
                                  start_sec=clip['start_sec'], end_sec=clip['end_sec'],
                                  error_type=type(exc).__name__, source_frame_index=None, source_time_sec=None)
                results.append(result)
                print(json.dumps(dict(stage='simple_clip', video=ordinal, clip=len(results), total_clips=len(clips),
                                      status=result.get('status'), prediction=result.get('prediction'),
                                      profile=result.get('profile'))), flush=True)
                if result.get('error_type') in ('OutOfMemoryError', 'RuntimeError'):
                    if len(results) < len(inspect_clips):
                        results.append(dict(status='failed', prediction='insufficient_data',
                                            error_type='RemainingClipsNotProcessed'))
                    break
            prediction, selected = combine(results)
            if classification is not None:
                prediction = classification['prediction']
            evidence_time = None
            if selected is not None:
                item = results[selected]
                evidence_time = item.get('source_time_sec')
                index = item.get('source_frame_index')
                if index is not None and index in clips[selected]['source_indices']:
                    import cv2
                    position = clips[selected]['source_indices'].index(index)
                    target = directory / 'candidate-frame.jpg'
                    if not cv2.imwrite(str(target), cv2.cvtColor(clips[selected]['rgb'][position], cv2.COLOR_RGB2BGR)):
                        raise ValueError('Could not save the candidate frame')
                    item['candidate_frame_file'] = target.name
            elapsed = time.perf_counter() - started
            measured = comparison(prediction, row['label'], evidence_time, row['event_times'])
            record = dict(video_sha256=row['sha256'], prediction=prediction, coverage=metadata,
                          preparation_sec=round(prepare_sec, 3), elapsed_sec=round(elapsed, 3),
                          processing_under_60_sec=elapsed <= 60, clips=results, comparison=measured,
                          classification=classification, localization_requested=getattr(args, 'locate', False),
                          experimental=True, production_approved=False)
        except Exception as exc:
            record = dict(video_sha256=row['sha256'], prediction='insufficient_data',
                          error_type=type(exc).__name__, elapsed_sec=round(time.perf_counter() - started, 3),
                          comparison=comparison('insufficient_data', row['label'], None, row['event_times']),
                          experimental=True, production_approved=False)
        save_json(directory / 'result.json', record)
        records.append(record)
        print(json.dumps(dict(stage='simple_video_complete', video=ordinal, total=len(rows),
                              video_sha256=row['sha256'], prediction=record['prediction'],
                              elapsed_sec=record['elapsed_sec'], comparison=record['comparison'],
                              error_type=record.get('error_type'))), flush=True)
    known = [row for row in records if row['comparison']['expected_label'] is not None]
    summary = dict(version=VERSION, complete=len(records) == len(rows), videos=len(records),
                   model_path=str(model_path), model_config_sha256=file_hash(model_path / 'config.json'),
                   model_weights_rehashed=learned is not None, device=args.device, settings=settings, source_hashes=source_hashes,
                   model_load_sec=round(load_sec, 3), total_elapsed_sec=round(time.perf_counter() - began, 3),
                   labelled_videos=len(known), correct=sum(row['comparison']['label_matches'] is True for row in known),
                   wrong=sum(row['comparison']['label_matches'] is False for row in known),
                   undecided=sum(row['prediction'] == 'insufficient_data' for row in records),
                   failed=sum('error_type' in row or any(c.get('status') == 'failed' for c in row.get('clips', []))
                              for row in records),
                   raw_responses_exported=False, independent_new_dataset=False,
                   training_performed=False, production_approved=False,
                   note='Single-model smoke/evaluation. One video does not establish accuracy. '
                        'Per-video processing excludes initial model loading; total time includes it.')
    save_json(output / 'summary.json', summary)
    print(json.dumps(dict(stage='simple_finished', **summary)), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description='One local Qwen model, native video input, compact verdict; no ensemble training.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--dataset', type=Path)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--model', type=Path, default=Path('../ensemble/models/qwen'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--head', type=Path, help='Learned predictor.json from train_head.py')
    parser.add_argument('--locate', action='store_true', help='After a positive learned verdict, ask the same Qwen for a candidate time')
    parser.add_argument('--sample-fps', type=float, default=DEFAULTS['sample_fps'])
    parser.add_argument('--max-new-tokens', type=int, default=DEFAULTS['max_new_tokens'])
    parser.add_argument('--expected-label', type=int, choices=(0, 1))
    parser.add_argument('--expected-time', type=float, action='append')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or not .5 <= args.sample_fps <= 2 or not 64 <= args.max_new_tokens <= 160:
        parser.error('Use limit >= 1, sample-fps 0.5..2 and max-new-tokens 64..160')
    if args.dataset and (args.expected_label is not None or args.expected_time):
        parser.error('Dataset labels come from its manifest')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps(dict(stage='simple_failed', error_type=type(exc).__name__)), flush=True)
        raise SystemExit(1) from None
    if result['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
