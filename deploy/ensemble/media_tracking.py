import bisect
import hashlib
import json
import math
import os
from pathlib import Path


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def normalized_box(box, name='box'):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f'{name} must be normalized [x1, y1, x2, y2]')
    values = [_finite(value, name) for value in box]
    if not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
        raise ValueError(f'{name} must have positive area inside [0, 1]')
    return values


def validate_camera(camera=None):
    if camera is None:
        camera = {}
    if not isinstance(camera, dict):
        raise ValueError('Camera configuration must be an object')
    result = {}
    for key in ('cashier_zone', 'counter_crop'):
        result[key] = normalized_box(camera[key], key) if camera.get(key) is not None else None
    return result


def _settings(settings=None):
    settings = settings or {}
    if not isinstance(settings, dict):
        raise ValueError('Media settings must be an object')
    result = dict(window_sec=settings.get('window_sec', 16.), overlap_sec=settings.get('overlap_sec', 4.),
                  sample_fps=settings.get('sample_fps', settings.get('fps', 4.)),
                  max_frames_per_window=settings.get('max_frames_per_window', settings.get('max_frames', 64)),
                  max_edge=settings.get('max_edge', 848), image_format=settings.get('image_format', 'jpg'),
                  jpeg_quality=settings.get('jpeg_quality', 92), device=str(settings.get('device', 'cpu')),
                  detector_conf=settings.get('detector_conf', .15), detector_iou=settings.get('detector_iou', .5),
                  detector_imgsz=settings.get('detector_imgsz', 640),
                  track_buffer_frames=settings.get('track_buffer_frames', 12))
    for key in ('window_sec', 'overlap_sec', 'sample_fps', 'detector_conf', 'detector_iou'):
        result[key] = _finite(result[key], key)
    for key in ('max_frames_per_window', 'max_edge', 'jpeg_quality', 'detector_imgsz', 'track_buffer_frames'):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if (result['window_sec'] <= 0 or not 0 <= result['overlap_sec'] < result['window_sec']
            or result['sample_fps'] <= 0 or not 0 < result['detector_conf'] < 1
            or not 0 < result['detector_iou'] <= 1 or not 1 <= result['jpeg_quality'] <= 100
            or result['image_format'] not in ('jpg', 'png') or result['max_edge'] > 4096
            or result['detector_imgsz'] > 4096 or result['max_frames_per_window'] > 256):
        raise ValueError('Invalid media sampling or detector settings')
    required = math.ceil(result['window_sec'] * result['sample_fps'] - 1e-9)
    if required > result['max_frames_per_window']:
        raise ValueError('Frame budget is too small for the requested sampling density; no thinning is performed')
    return result


def plan_video(source_frames, fps, settings=None):
    settings = _settings(settings)
    fps = _finite(fps, 'source FPS')
    if type(source_frames) is not int or source_frames <= 0 or fps <= 0:
        raise ValueError('Video needs positive frame count and FPS')
    duration = source_frames / fps
    rate = min(settings['sample_fps'], fps)
    slots = max(1, math.ceil(duration * rate - 1e-9))
    sampled, nominal_times = [], []
    for slot in range(slots):
        index = min(source_frames - 1, round(slot * fps / rate))
        if not sampled or index != sampled[-1]:
            sampled.append(index)
            nominal_times.append(slot / rate)
    windows, number = [], 0
    stride = settings['window_sec'] - settings['overlap_sec']
    while True:
        start = number * stride
        end = min(duration, start + settings['window_sec'])
        left = bisect.bisect_left(nominal_times, start - 1e-9)
        right = bisect.bisect_left(nominal_times, end - 1e-9)
        selected = sampled[left:right]
        if not selected:
            selected = [min(source_frames - 1, max(0, round(start * fps)))]
        if len(selected) > settings['max_frames_per_window']:
            raise ValueError('Window exceeds configured frame budget; no thinning is performed')
        first = max(0, math.floor(start * fps + 1e-9))
        last = min(source_frames, math.ceil(end * fps - 1e-9))
        windows.append(dict(index=number, start_sec=first / fps, end_sec=last / fps,
                            planned_start_sec=start, planned_end_sec=end,
                            start_frame=first, end_frame_exclusive=last,
                            frame_indices=selected, times=[i / fps for i in selected]))
        if end >= duration:
            break
        number += 1
    selected = sorted({index for window in windows for index in window['frame_indices']})
    return windows, selected


class _PersonTracker:
    def __init__(self, detector_path, output, settings):
        detector_path = Path(detector_path).resolve()
        if not detector_path.is_file() or detector_path.suffix.lower() != '.pt':
            raise ValueError('Person detector requires an existing local .pt model; downloading is disabled')
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_VERBOSE'] = 'false'
        config = output / 'ultralytics-config'
        config.mkdir(exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(config)
        from ultralytics import YOLO, settings as yolo_settings
        from ultralytics import utils
        utils.ONLINE = False
        utils.AUTOINSTALL = False
        disabled = {key: False for key in ('sync', 'hub', 'wandb', 'mlflow', 'clearml', 'comet', 'dvc',
                                          'neptune', 'raytune', 'tensorboard') if key in yolo_settings}
        yolo_settings.update(disabled)
        self.model = YOLO(str(detector_path), task='detect')
        self.settings = settings
        self.config_path = output / 'bytetrack.yaml'
        self.config_path.write_text(
            'tracker_type: bytetrack\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.1\n'
            'new_track_thresh: 0.25\nmatch_thresh: 0.8\nfuse_score: true\n'
            f'track_buffer: {settings["track_buffer_frames"]}\n', encoding='utf8')

    def update(self, frame):
        result = self.model.track(source=frame, persist=True, tracker=str(self.config_path), classes=[0],
                                  conf=self.settings['detector_conf'], iou=self.settings['detector_iou'],
                                  imgsz=self.settings['detector_imgsz'], device=self.settings['device'],
                                  verbose=False, save=False, show=False, save_txt=False, stream=False)[0]
        boxes = result.boxes
        if boxes is None:
            return dict(tracks=[], detected_people=0, untracked_people=0)
        if boxes.id is None:
            return dict(tracks=[], detected_people=len(boxes), untracked_people=len(boxes))
        height, width = frame.shape[:2]
        records = []
        for identity, box, confidence in zip(boxes.id.cpu().tolist(), boxes.xyxy.cpu().tolist(),
                                             boxes.conf.cpu().tolist()):
            if not math.isfinite(identity) or identity < 0 or int(identity) != identity:
                raise ValueError('Tracker returned an invalid persistent ID')
            box = [max(0., min(1., _finite(float(value), 'detector coordinate') / scale))
                   for value, scale in zip(box, (width, height, width, height))]
            records.append(dict(track_id=int(identity), box=normalized_box(box),
                                confidence=_finite(float(confidence), 'detector confidence')))
        return dict(tracks=records, detected_people=None, untracked_people=None)


def _in_zone(box, zone):
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]


def _candidate_established_at(samples, zone):
    if zone is None:
        return None
    inside = 0
    for count, sample in enumerate(samples, 1):
        inside += bool(sample.get('in_cashier_zone', False))
        if count >= 3 and inside / count >= .8:
            return sample['time_sec']
    return None


def _track_record(identity, samples, zone, candidate_since=None):
    fraction = sum(sample.get('in_cashier_zone', False) for sample in samples) / len(samples) if zone else None
    candidate = zone is not None and candidate_since is not None and candidate_since <= samples[-1]['time_sec']
    return dict(track_id=identity, first_sec=samples[0]['time_sec'], last_sec=samples[-1]['time_sec'], samples=samples,
                role='cashier_candidate' if candidate else 'unknown',
                role_source='configured_zone' if candidate else 'unknown',
                role_established_sec=candidate_since if candidate else None,
                zone_observation_fraction=fraction, identity_verified=False, role_verified=False)


def _safe_cached_path(output, name):
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Invalid cached frame path')
    path = (output / name).resolve()
    if not path.is_relative_to(output):
        raise ValueError('Cached frame path escapes media directory')
    return path


def _resume(output, protocol):
    path = output / 'index.json'
    if not path.exists() or not (output / 'complete.json').exists():
        return None
    completion = json.loads((output / 'complete.json').read_text(encoding='utf8'))
    if completion.get('index_sha256') != _hash(path):
        raise ValueError('Completed media index checksum mismatch')
    value = json.loads(path.read_text(encoding='utf8'))
    if not value.get('coverage_complete') or value.get('protocol') != protocol:
        raise ValueError('Completed media cache does not match current inputs')
    frames = {record['frame_index']: record for record in value['frames']}
    if len(frames) != value['sampled_frames']:
        raise ValueError('Duplicate or missing cached frame metadata')
    for frame in frames.values():
        if _hash(_safe_cached_path(output, frame['path'])) != frame['sha256']:
            raise ValueError('Cached frame checksum mismatch')
    for window in value['windows']:
        if window['frame_paths'] != [frames[i]['path'] for i in window['frame_indices']]:
            raise ValueError('Window frame paths do not match cache metadata')
    return value


def prepare_video(video, output, settings=None, camera=None, detector_path=None):
    import cv2
    video, output = Path(video).resolve(), Path(output).resolve()
    settings, camera = _settings(settings), validate_camera(camera)
    if not video.is_file():
        raise ValueError('Source video must be an existing local file')
    if detector_path is not None:
        detector_path = Path(detector_path).resolve()
        if not detector_path.is_file() or detector_path.suffix.lower() != '.pt':
            raise ValueError('Detector must be an existing local .pt file')
    protocol = dict(version='ensemble_media_v1', source_sha256=_hash(video), settings=settings, camera=camera,
                    detector_sha256=_hash(detector_path) if detector_path is not None else None,
                    code_sha256=_hash(Path(__file__)))
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf8')) != protocol:
            raise ValueError('Media inputs or settings changed; use a new output directory')
        resumed = _resume(output, protocol)
        if resumed is not None:
            return resumed
    elif any(output.iterdir()):
        raise ValueError('Media output is not empty and has no protocol')
    else:
        _save(protocol_path, protocol)
    frame_directory = output / 'frames'
    frame_directory.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    decoded, records, tracks = 0, [], {}
    try:
        if not capture.isOpened():
            raise ValueError('Could not open source video')
        fps = _finite(capture.get(cv2.CAP_PROP_FPS), 'source FPS')
        reported = _finite(capture.get(cv2.CAP_PROP_FRAME_COUNT), 'source frame count')
        if fps <= 0 or reported <= 0 or int(reported) != reported:
            raise ValueError('Invalid source FPS or frame count')
        total = int(reported)
        windows, selected = plan_video(total, fps, settings)
        selected_set = set(selected)
        tracker = _PersonTracker(detector_path, output, settings) if detector_path is not None else None
        source_shape = None
        for index in range(total):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f'Sequential decoding stopped at frame {index} of {total}; no frame substituted')
            decoded += 1
            if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
                raise ValueError(f'Invalid source image at frame {index}')
            if source_shape is None:
                source_shape = list(frame.shape[:2])
            elif list(frame.shape[:2]) != source_shape:
                raise ValueError('Video dimensions changed; configured normalized zones need review')
            if index not in selected_set:
                continue
            height, width = frame.shape[:2]
            factor = min(1., settings['max_edge'] / max(height, width))
            if factor < 1:
                frame = cv2.resize(frame, (max(1, round(width * factor)), max(1, round(height * factor))),
                                   interpolation=cv2.INTER_AREA)
            relative = f'frames/{index:012d}.{settings["image_format"]}'
            parameters = [cv2.IMWRITE_JPEG_QUALITY, settings['jpeg_quality']] if settings['image_format'] == 'jpg' else []
            ok, encoded = cv2.imencode('.' + settings['image_format'], frame, parameters)
            if not ok:
                raise ValueError(f'Could not encode selected frame {index}')
            payload = encoded.tobytes()
            path = output / relative
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_bytes(payload)
            temporary.replace(path)
            detected = tracker.update(frame) if tracker else dict(tracks=[], detected_people=None, untracked_people=None)
            identities = set()
            for person in detected['tracks']:
                identity = person['track_id']
                if type(identity) is not int or identity < 0 or identity in identities:
                    raise ValueError('Duplicate or invalid person tracking ID')
                identities.add(identity)
                box = normalized_box(person['box'], 'tracked person box')
                confidence = _finite(person['confidence'], 'detector confidence')
                if not 0 <= confidence <= 1:
                    raise ValueError('Detector confidence is outside [0, 1]')
                sample = dict(frame_index=index, time_sec=index / fps, box=box, confidence=confidence)
                if camera['cashier_zone'] is not None:
                    sample['in_cashier_zone'] = _in_zone(box, camera['cashier_zone'])
                tracks.setdefault(identity, []).append(sample)
            records.append(dict(frame_index=index, time_sec=index / fps, path=relative,
                                sha256=hashlib.sha256(payload).hexdigest(), width=frame.shape[1], height=frame.shape[0],
                                track_ids=sorted(identities), detected_people=detected['detected_people'],
                                untracked_people=detected['untracked_people']))
        extra_ok, extra_frame = capture.read()
        if extra_ok or extra_frame is not None:
            raise ValueError('Decoded frames exceed reported frame count; refusing incomplete timeline metadata')
        if len(records) != len(selected):
            raise ValueError('Selected frame cache is incomplete')
        by_index = {record['frame_index']: record for record in records}
        candidate_since = {identity: _candidate_established_at(samples, camera['cashier_zone'])
                           for identity, samples in tracks.items()}
        for window in windows:
            indices = set(window['frame_indices'])
            window['frame_paths'] = [by_index[i]['path'] for i in window['frame_indices']]
            window['frame_sha256'] = [by_index[i]['sha256'] for i in window['frame_indices']]
            window['tracks'] = []
            for identity, samples in sorted(tracks.items()):
                observed = [sample for sample in samples if sample['frame_index'] in indices]
                if observed:
                    window['tracks'].append(_track_record(identity, observed, camera['cashier_zone'],
                                                          candidate_since[identity]))
            window['counter_crop'] = camera['counter_crop']
        times = [record['time_sec'] for record in records]
        result = dict(protocol=protocol, video_sha256=protocol['source_sha256'], duration_sec=total / fps,
                      fps=fps, source_frames=total, decoded_frames=decoded, sampled_frames=len(records),
                      source_width=source_shape[1], source_height=source_shape[0], frames=records,
                      windows=windows, global_tracks=[_track_record(identity, samples, camera['cashier_zone'],
                                                                    candidate_since[identity])
                                                     for identity, samples in sorted(tracks.items())],
                      coverage_complete=True, tracking_enabled=tracker is not None,
                      cashier_role_status='configured_zone_candidates' if camera['cashier_zone'] else 'unknown',
                      sample_fps_requested=settings['sample_fps'], sample_fps_effective=min(settings['sample_fps'], fps),
                      max_sample_gap_sec=max((b - a for a, b in zip(times, times[1:])), default=0.),
                      first_sample_sec=times[0], last_sample_sec=times[-1],
                      tail_unsampled_sec=total / fps - times[-1], timestamps_verified_pts=False,
                      actor_roles_verified=False, person_identity_verified=False,
                      missing_track_is_departure=False, automatic_verdict_enabled=False,
                      notes=['Every original frame was decoded; only sampled frames enter detection and video models.',
                             'Timing is frame ordinal / reported FPS. Variable-frame-rate presentation timestamps are not verified.',
                             'Track IDs are local motion-based associations within this video, not biometric identities.',
                             'Tracking can omit unconfirmed detections; tracked people are not a complete people count.',
                             'A configured zone creates only a cashier candidate; missing tracks do not prove departure.',
                             'A candidate association persists for its video track after three or more observations with '
                             'at least 80 percent inside the configured zone; role_established_sec marks its first evidence.',
                             'Images keep their aspect ratio; rounding may move planned window bounds by one source frame.'])
        _save(output / 'index.json', result)
        _save(output / 'complete.json', dict(index_sha256=_hash(output / 'index.json')))
        return result
    except Exception as exc:
        _save(output / 'failure.json', dict(coverage_complete=False, decoded_frames=decoded,
                                           sampled_frames=len(records), error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        capture.release()
