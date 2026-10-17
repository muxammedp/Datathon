import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.ensemble.common import (HERE, digest, exclusive_output, file_hash, load_payload, model_manifest,
                                    offline, read_json, release_gpu, save_json, save_payload, verify_models)


def protocol(settings, camera):
    from deploy.ensemble.media_tracking import validate_camera
    value = dict(version='forte_specialists_v1', settings=settings, camera=validate_camera(camera),
                 models=model_manifest(), source_hashes={name: file_hash(HERE / name) for name in
                 ('common.py', 'run.py', 'media_tracking.py', 'action_expert.py', 'qwen_expert.py')})
    return dict(value, fingerprint=digest(value))


def event(stage, **details):
    print(json.dumps(dict(stage=stage, **details), ensure_ascii=False, allow_nan=False), flush=True)


def check_windows(payload, media):
    if payload.get('video_sha256') != media['video_sha256'] or len(payload.get('windows', [])) != len(media['windows']):
        raise ValueError('Expert result identity or window count differs from media')
    for actual, expected in zip(payload['windows'], media['windows']):
        if any(actual.get(key) != expected[key] for key in ('index', 'start_sec', 'end_sec')):
            raise ValueError('Expert result intervals differ from media')
        if actual.get('status', 'done') != 'done':
            raise ValueError('A specialist window failed; retry extraction before training')


def extract(rows, dataset, output, models, settings, camera=None):
    if not rows or len({row['sha256'] for row in rows}) != len(rows):
        raise ValueError('Extraction requires nonempty SHA-unique video inputs')
    offline()
    output, models, dataset = Path(output).resolve(), Path(models).resolve(), Path(dataset).resolve()
    fingerprint = protocol(settings, camera)
    identity = fingerprint['fingerprint']
    with exclusive_output(output):
        if (output / 'protocol.json').exists() and read_json(output / 'protocol.json') != fingerprint:
            raise ValueError('Source/model/settings changed. Choose a new cache directory; old caches are preserved.')
        (output / 'complete.json').unlink(missing_ok=True)
        save_json(output / 'protocol.json', fingerprint)
        began = time.monotonic()
        verify_models(models)
        event('models_verified', elapsed_sec=round(time.monotonic() - began, 2))
        from deploy.ensemble.media_tracking import prepare_video
        media = {}
        timings_path = output / 'timings.json'
        timings = read_json(timings_path) if timings_path.exists() else {}
        for ordinal, row in enumerate(rows, 1):
            start = time.monotonic()
            path = (dataset / row['file']).resolve()
            if not path.is_relative_to(dataset) or file_hash(path) != row['sha256']:
                raise ValueError('Input path/checksum changed')
            directory = output / 'videos' / row['sha256']
            media_settings = {key: value for key, value in settings.items() if not key.startswith('qwen_')}
            resumed = (directory / 'media' / 'complete.json').exists()
            media[row['sha256']] = prepare_video(path, directory / 'media', settings=media_settings,
                                               camera=fingerprint['camera'], detector_path=models / 'detector' / 'yolo11n.pt')
            if not resumed:
                timings.setdefault(row['sha256'], {})['media_sec'] = time.monotonic() - start
            save_json(timings_path, timings)
            event('media_ready', video=ordinal, total=len(rows), reused=resumed,
                  windows=len(media[row['sha256']]['windows']), elapsed_sec=round(time.monotonic() - start, 2))
        release_gpu()
        action = None
        try:
            for ordinal, row in enumerate(rows, 1):
                directory = output / 'videos' / row['sha256']
                target = directory / 'action.json'
                start = time.monotonic()
                if target.exists():
                    result = load_payload(target, identity)
                    reused = True
                else:
                    if action is None:
                        from deploy.ensemble.action_expert import ActionExpert
                        action = ActionExpert(models / 'action', device=settings['device'])
                    result = action.extract(media[row['sha256']], directory / 'media')
                    check_windows(result, media[row['sha256']])
                    save_payload(target, result, identity)
                    timings.setdefault(row['sha256'], {})['action_sec'] = time.monotonic() - start
                    save_json(timings_path, timings)
                    reused = False
                check_windows(result, media[row['sha256']])
                event('action_ready', video=ordinal, total=len(rows), reused=reused,
                      elapsed_sec=round(time.monotonic() - start, 2))
        finally:
            if action is not None:
                action.release()
            del action
            release_gpu()
        qwen = None
        try:
            for ordinal, row in enumerate(rows, 1):
                directory = output / 'videos' / row['sha256']
                target, partial = directory / 'qwen.json', directory / 'qwen.partial.json'
                start = time.monotonic()
                if target.exists():
                    result = load_payload(target, identity)
                    reused = True
                else:
                    if qwen is None:
                        from deploy.ensemble.qwen_expert import QwenExpert
                        qwen = QwenExpert(models / 'qwen', device=settings['device'],
                                          max_frames=settings['qwen_frames'], max_new_tokens=settings['qwen_new_tokens'])
                    previous = load_payload(partial, identity) if partial.exists() else None

                    def checkpoint(payload):
                        save_payload(partial, payload, identity)
                        event('qwen_window', video=ordinal, total=len(rows), window=len(payload['windows']),
                              total_windows=len(media[row['sha256']]['windows']), status=payload['windows'][-1]['status'],
                              elapsed_sec=round(time.monotonic() - start, 2))

                    result = qwen.analyze(media[row['sha256']], directory / 'media', previous_results=previous,
                                          on_window=checkpoint)
                    check_windows(result, media[row['sha256']])
                    save_payload(target, result, identity)
                    timings.setdefault(row['sha256'], {})['qwen_sec'] = time.monotonic() - start
                    timings[row['sha256']]['qwen_resumed'] = previous is not None
                    save_json(timings_path, timings)
                    reused = False
                check_windows(result, media[row['sha256']])
                event('qwen_ready', video=ordinal, total=len(rows), reused=reused,
                      elapsed_sec=round(time.monotonic() - start, 2))
        finally:
            del qwen
            release_gpu()
        save_json(output / 'complete.json', dict(fingerprint=identity, videos=[row['sha256'] for row in rows],
                                                complete=True, elapsed_this_run_sec=time.monotonic() - began))
        event('extraction_complete', videos=len(rows), training_started=False)
    return fingerprint


def main():
    parser = argparse.ArgumentParser(description='Self-hosted specialist extraction. All models run on this machine.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--video', type=Path)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--qwen-frames', type=int, default=16)
    parser.add_argument('--qwen-new-tokens', type=int, default=384)
    args = parser.parse_args()
    if not 2 <= args.qwen_frames <= 32 or not 64 <= args.qwen_new_tokens <= 768:
        parser.error('Use --qwen-frames 2..32 and --qwen-new-tokens 64..768')
    if args.dataset:
        from deploy.internvideo.evaluate_dataset import check_dataset
        rows = check_dataset(args.dataset)['videos']
        dataset = args.dataset
    else:
        video = args.video.resolve()
        dataset = video.parent
        rows = [dict(file=video.name, sha256=file_hash(video))]
    settings = dict(window_sec=16., overlap_sec=4., sample_fps=4., max_frames_per_window=64,
                    max_edge=848, device=args.device, qwen_frames=args.qwen_frames, qwen_new_tokens=args.qwen_new_tokens)
    extract(rows, dataset, args.output, args.models, settings, read_json(args.camera) if args.camera else None)


if __name__ == '__main__':
    main()
