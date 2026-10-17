import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.ensemble.common import HERE, file_hash, load_payload, read_json, save_json


def candidate_evidence(media, action_scores, qwen, media_root):
    if len(action_scores) != len(media['windows']):
        raise ValueError('Action localization scores do not match media windows')
    chosen = max(range(len(action_scores)), key=lambda i: action_scores[i])
    window = media['windows'][chosen]
    middle = (window['start_sec'] + window['end_sec']) / 2
    observation = None
    facts = qwen['windows'][chosen].get('observations', {})
    events = [event for view in ('context', 'cash') for event in facts.get(view, {}).get('events', [])
              if window['start_sec'] <= event['time_sec'] < window['end_sec']]
    if events:
        observation = min(events, key=lambda item: abs(item['time_sec'] - middle))
        middle = observation['time_sec']
    position = min(range(len(window['times'])), key=lambda i: abs(window['times'][i] - middle))
    frame_path = (Path(media_root) / window['frame_paths'][position]).resolve()
    if (not frame_path.is_relative_to(Path(media_root).resolve())
            or file_hash(frame_path) != window['frame_sha256'][position]):
        raise ValueError('Evidence frame is missing or modified')
    return dict(window_index=chosen, start_sec=window['start_sec'], end_sec=window['end_sec'],
                frame_index=window['frame_indices'][position], time_sec=window['times'][position],
                frame_path=str(frame_path), observed_event=observation, verified_violation=False,
                timing_method='frame_ordinal_over_reported_fps', timestamps_verified_pts=False,
                selection='highest_action_score_window_then_observation_or_midpoint',
                note='Review candidate only. This frame does not establish the exact violation time or type.')


def infer(args):
    began = time.monotonic()
    video = args.video.resolve()
    bundle = read_json(args.bundle)
    config = bundle['extraction_protocol']
    settings = config['settings']
    defaults = dict(window_sec=16., overlap_sec=4., sample_fps=4., max_frames_per_window=64, max_edge=848)
    if any(settings.get(key) != value for key, value in defaults.items()) or set(settings) != set(defaults) | {'device', 'qwen_frames', 'qwen_new_tokens'}:
        raise ValueError('Predict CLI does not support this custom extraction settings protocol')
    if bundle.get('fingerprint') != config['fingerprint']:
        raise ValueError('Predictor and extraction fingerprint differ')
    if any(config['camera'].get(key) is not None for key in ('cashier_zone', 'counter_crop')) and not args.same_camera:
        raise ValueError('This bundle uses configured camera zones. Use --same-camera only for the same unchanged viewpoint; new angles need a full-scene bundle.')
    from deploy.ensemble.run import protocol
    if protocol(settings, config['camera'])['fingerprint'] != config['fingerprint']:
        raise ValueError('Predictor was trained with different code/models/settings; use its matching source revision')
    args.cache.mkdir(parents=True, exist_ok=True)
    camera = args.cache / ('predict-camera-' + uuid.uuid4().hex + '.json')
    save_json(camera, config['camera'])
    remaining = args.timeout - (time.monotonic() - began)
    if remaining <= 0:
        raise TimeoutError('Processing time budget exhausted before extraction')
    command = [sys.executable, str(HERE / 'run.py'), '--video', str(video), '--models', str(args.models.resolve()),
               '--output', str(args.cache.resolve()), '--camera', str(camera.resolve()), '--device', settings['device'],
               '--qwen-frames', str(settings['qwen_frames']), '--qwen-new-tokens', str(settings['qwen_new_tokens'])]
    try:
        with subprocess.Popen(command) as process:
            try:
                code = process.wait(timeout=remaining)
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                lock = args.cache / '.ensemble.lock'
                if lock.exists() and lock.read_text().strip() == str(process.pid):
                    lock.unlink()
                raise
    finally:
        camera.unlink(missing_ok=True)
    identity = file_hash(video)
    directory = args.cache / 'videos' / identity
    action = load_payload(directory / 'action.json', config['fingerprint'])
    qwen = load_payload(directory / 'qwen.json', config['fingerprint'])
    media = read_json(directory / 'media' / 'index.json')
    if (media.get('video_sha256') != identity or media.get('coverage_complete') is not True
            or media.get('decoded_frames') != media.get('source_frames')):
        raise ValueError('Media identity or full decoding coverage mismatch')
    from deploy.ensemble.run import check_windows
    check_windows(action, media)
    check_windows(qwen, media)
    from deploy.ensemble.fusion import predict_record
    scores = predict_record(bundle, action, qwen)
    reasons = sorted({reason for row in qwen['windows'] for reason in row.get('review_reasons', [])})
    if scores.get('distribution_shift_flag'):
        reasons.append('embedding_distribution_shift_heuristic')
    evidence = candidate_evidence(media, scores['window_action_scores'], qwen, directory / 'media')
    elapsed = time.monotonic() - began
    if elapsed > args.timeout:
        raise TimeoutError('Processing exceeded the configured time budget')
    seen = set(bundle.get('train_sha256', []) + bundle.get('calibration_sha256', []) + bundle.get('test_sha256', []))
    return dict(complete=True, experimental=True, production_approved=False,
                prediction='violation' if scores['prediction'] else 'no_violation',
                category=None, video_sha256=identity, scores=scores, elapsed_sec=round(elapsed, 3),
                within_time_budget=True, time_budget_sec=args.timeout,
                seen_in_development=identity in seen, independent_new_dataset=False,
                review_reasons=reasons, unknown_activity_is_not_automatically_a_violation=True,
                candidate_evidence=evidence,
                coverage=dict(decoded_frames=media['decoded_frames'], sampled_frames=media['sampled_frames'],
                              windows=len(media['windows']), max_sample_gap_sec=media['max_sample_gap_sec']),
                actor_roles_verified=False, exact_event_localization_validated=False,
                new_violation_types_validated=False, camera_angle_generalization_validated=False,
                note='Experimental ensemble CLI output; this is not the jury API schema or a verified legal finding.')


def main():
    parser = argparse.ArgumentParser(description='Run the frozen specialists and a trained fusion model on one local video.')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--cache', type=Path, default=Path('results/predict-cache'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--same-camera', action='store_true', help='Assert that configured zones still match this unchanged camera view')
    args = parser.parse_args()
    target = args.output.resolve()
    if target in (args.video.resolve(), args.bundle.resolve()) or target.is_relative_to(args.models.resolve()):
        parser.error('--output must not overwrite the input video, trained bundle, or base model files')
    if not 1 <= args.timeout <= 300:
        parser.error('--timeout must be between 1 and 300 seconds')
    began = time.monotonic()
    try:
        result = infer(args)
    except Exception as exc:
        result = dict(complete=False, prediction='insufficient_data', experimental=True, production_approved=False,
                      error_type=type(exc).__name__, error=str(exc), elapsed_sec=round(time.monotonic() - began, 3),
                      within_time_budget=False if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else None,
                      time_budget_sec=args.timeout, reason='Processing failed or did not finish; no normal verdict inferred.')
    save_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'scores'}, ensure_ascii=False), flush=True)
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
