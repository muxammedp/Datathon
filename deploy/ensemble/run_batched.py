import argparse
import math
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.ensemble.common import (HERE, digest, exclusive_output, file_hash, load_payload, offline,
                                    read_json, release_gpu, save_json, save_payload, verify_models)
from deploy.ensemble.run import check_windows, event, protocol as original_protocol


def protocol(settings, camera):
    value = original_protocol(settings, camera)
    value.pop('fingerprint')
    value['version'] = 'forte_specialists_batched_v2'
    value['execution'] = dict(qwen_version='qwen_batched_v2', view_batch_size=2,
                              observations_version='observations_v2', max_observation_events=64)
    value['source_hashes'].update({name: file_hash(HERE / name) for name in
                                   ('qwen_batched.py', 'run_batched.py', 'observation_schema.py')})
    return dict(value, fingerprint=digest(value))


def _reuse_source(source, output, settings, camera):
    if source is None:
        return None, None
    source = Path(source).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Reuse source and output must be separate non-nested cache directories')
    expected = original_protocol(settings, camera)
    if not (source / 'protocol.json').is_file() or read_json(source / 'protocol.json') != expected:
        raise ValueError('Reuse cache must exactly match the original v1 source, models, settings and camera protocol')
    return source, expected['fingerprint']


def _contained(root, *parts):
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Cache path escapes its root')
    return path


def _copy_media(source, destination):
    if not (source / 'complete.json').is_file():
        return False
    if destination.exists():
        return False
    for path in [source, *source.rglob('*')]:
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError('Media import cannot follow filesystem links')
    completion = read_json(source / 'complete.json')
    if completion.get('index_sha256') != file_hash(source / 'index.json'):
        raise ValueError('Reuse media index checksum mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + '.importing')
    if staging.exists():
        raise ValueError('Interrupted media import exists; use another output directory')
    shutil.copytree(source, staging, copy_function=shutil.copy2)
    if read_json(source / 'complete.json') != completion or file_hash(staging / 'index.json') != completion['index_sha256']:
        raise ValueError('Reuse media changed during copying')
    staging.replace(destination)
    return True


def _check_action(payload, media):
    from deploy.ensemble.action_expert import FEATURE_SIZE, MODEL_ID, VERSION, clip_indices
    check_windows(payload, media)
    if (payload.get('version') != VERSION or payload.get('model_identifier') != MODEL_ID
            or payload.get('feature_dimensions') != FEATURE_SIZE or payload.get('all_windows_processed') is not True
            or payload.get('temporal_frames_per_clip') != 8 or payload.get('temporal_stride') != 4):
        raise ValueError('Action cache has incompatible model or feature metadata')
    for actual, expected in zip(payload['windows'], media['windows']):
        features = actual.get('features')
        if (not isinstance(features, list) or len(features) != FEATURE_SIZE
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in features)):
            raise ValueError('Action cache has invalid feature values')
        count = len(expected['frame_paths'])
        clips = clip_indices(count)
        if (actual.get('selected_times') != expected['times'] or actual.get('sampled_frames') != count
                or actual.get('crop_used') is not (expected.get('counter_crop') is not None)
                or actual.get('all_supplied_samples_covered') is not True
                or actual.get('covered_sample_indices') != list(range(count))
                or actual.get('clips_processed') != len(clips)
                or [clip.get('sample_indices') for clip in actual.get('clips', [])] != clips):
            raise ValueError('Action cache does not cover the validated media samples')


def _check_qwen(payload, media, settings):
    from deploy.ensemble.qwen_batched import VERSION
    from deploy.ensemble.qwen_expert import MAX_PIXELS
    from deploy.ensemble import observation_schema as observations
    check_windows(payload, media)
    if payload.get('version') != VERSION:
        raise ValueError('Qwen cache does not belong to the batched backend')
    expected = dict(max_frames=settings['qwen_frames'], max_new_tokens=settings['qwen_new_tokens'],
                    max_pixels=MAX_PIXELS, view_batch_size=2, execution='batched_views_v2',
                    observations_version=observations.VERSION, max_observation_events=observations.MAX_EVENTS)
    if payload.get('settings') != expected:
        raise ValueError('Qwen cache has incompatible batch settings')


def extract(rows, dataset, output, models, settings, camera=None, reuse_from=None):
    if not rows or len({row['sha256'] for row in rows}) != len(rows):
        raise ValueError('Extraction requires nonempty SHA-unique video inputs')
    offline()
    output, models, dataset = Path(output).resolve(), Path(models).resolve(), Path(dataset).resolve()
    source, source_identity = _reuse_source(reuse_from, output, settings, camera)
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
        source_timings = read_json(source / 'timings.json') if source and (source / 'timings.json').exists() else {}
        for ordinal, row in enumerate(rows, 1):
            start = time.monotonic()
            path = (dataset / row['file']).resolve()
            if not path.is_relative_to(dataset) or file_hash(path) != row['sha256']:
                raise ValueError('Input path/checksum changed')
            directory = _contained(output, 'videos', row['sha256'])
            media_settings = {key: value for key, value in settings.items() if not key.startswith('qwen_')}
            imported = _copy_media(_contained(source, 'videos', row['sha256'], 'media'), directory / 'media') if source else False
            resumed = (directory / 'media' / 'complete.json').exists()
            media[row['sha256']] = prepare_video(path, directory / 'media', settings=media_settings,
                                               camera=fingerprint['camera'], detector_path=models / 'detector' / 'yolo11n.pt')
            if not resumed:
                timings.setdefault(row['sha256'], {})['media_sec'] = time.monotonic() - start
            elif imported:
                timings.setdefault(row['sha256'], {})['media_imported'] = True
                if 'media_sec' in source_timings.get(row['sha256'], {}):
                    timings[row['sha256']]['media_sec'] = source_timings[row['sha256']]['media_sec']
            save_json(timings_path, timings)
            event('media_ready', video=ordinal, total=len(rows), reused=resumed, imported=imported,
                  windows=len(media[row['sha256']]['windows']), elapsed_sec=round(time.monotonic() - start, 2))
        release_gpu()
        action = None
        try:
            for ordinal, row in enumerate(rows, 1):
                directory = _contained(output, 'videos', row['sha256'])
                target = directory / 'action.json'
                start = time.monotonic()
                imported = False
                original = _contained(source, 'videos', row['sha256'], 'action.json') if source else None
                if target.exists():
                    result = load_payload(target, identity)
                    reused = True
                elif original is not None and original.is_file():
                    result = load_payload(original, source_identity)
                    _check_action(result, media[row['sha256']])
                    save_payload(target, result, identity)
                    reused = imported = True
                    timings.setdefault(row['sha256'], {})['action_imported'] = True
                    if 'action_sec' in source_timings.get(row['sha256'], {}):
                        timings[row['sha256']]['action_sec'] = source_timings[row['sha256']]['action_sec']
                    save_json(timings_path, timings)
                else:
                    if action is None:
                        from deploy.ensemble.action_expert import ActionExpert
                        action = ActionExpert(models / 'action', device=settings['device'])
                    result = action.extract(media[row['sha256']], directory / 'media')
                    _check_action(result, media[row['sha256']])
                    save_payload(target, result, identity)
                    timings.setdefault(row['sha256'], {})['action_sec'] = time.monotonic() - start
                    save_json(timings_path, timings)
                    reused = False
                _check_action(result, media[row['sha256']])
                event('action_ready', video=ordinal, total=len(rows), reused=reused, imported=imported,
                      elapsed_sec=round(time.monotonic() - start, 2))
        finally:
            if action is not None:
                action.release()
            del action
            release_gpu()
        qwen = None
        try:
            for ordinal, row in enumerate(rows, 1):
                directory = _contained(output, 'videos', row['sha256'])
                target, partial = directory / 'qwen.json', directory / 'qwen.partial.json'
                if not target.exists() and qwen is None:
                    from deploy.ensemble.qwen_batched import BatchedQwenExpert
                    load_start = time.monotonic()
                    qwen = BatchedQwenExpert(models / 'qwen', device=settings['device'],
                                            max_frames=settings['qwen_frames'], max_new_tokens=settings['qwen_new_tokens'])
                    qwen._load()
                    event('qwen_model_loaded', elapsed_sec=round(time.monotonic() - load_start, 2))
                start = time.monotonic()
                if target.exists():
                    result = load_payload(target, identity)
                    reused = True
                else:
                    previous = load_payload(partial, identity) if partial.exists() else None
                    reused_prefix = 0
                    if previous is not None:
                        for window in previous.get('windows', []):
                            if window.get('index') != reused_prefix or window.get('status') != 'done':
                                break
                            reused_prefix += 1

                    def checkpoint(payload):
                        save_payload(partial, payload, identity)
                        window = payload['windows'][-1]
                        reused_window = window['index'] < reused_prefix
                        event('qwen_window', video=ordinal, total=len(rows), window=len(payload['windows']),
                              total_windows=len(media[row['sha256']]['windows']), status=window['status'],
                              elapsed_sec=0. if reused_window else window.get('elapsed_sec'),
                              cached_elapsed_sec=window.get('elapsed_sec') if reused_window else None,
                              reused=reused_window, profile=window.get('profile'),
                              video_elapsed_sec=round(time.monotonic() - start, 2))

                    result = qwen.analyze(media[row['sha256']], directory / 'media', previous_results=previous,
                                          on_window=checkpoint)
                    _check_qwen(result, media[row['sha256']], settings)
                    save_payload(target, result, identity)
                    timings.setdefault(row['sha256'], {})['qwen_sec'] = time.monotonic() - start
                    timings[row['sha256']]['qwen_resumed'] = previous is not None
                    save_json(timings_path, timings)
                    reused = False
                _check_qwen(result, media[row['sha256']], settings)
                event('qwen_ready', video=ordinal, total=len(rows), reused=reused,
                      elapsed_sec=round(time.monotonic() - start, 2), model_load_included=False)
        finally:
            if qwen is not None:
                qwen.release()
            del qwen
            release_gpu()
        save_json(output / 'complete.json', dict(fingerprint=identity, videos=[row['sha256'] for row in rows],
                                                complete=True, elapsed_this_run_sec=time.monotonic() - began))
        event('extraction_complete', videos=len(rows), training_started=False)
    return fingerprint


def main():
    parser = argparse.ArgumentParser(description='Batched local specialists with optional validated media/action reuse.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--video', type=Path)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-from', type=Path)
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
    extract(rows, dataset, args.output, args.models, settings, read_json(args.camera) if args.camera else None,
            reuse_from=args.reuse_from)


if __name__ == '__main__':
    main()
