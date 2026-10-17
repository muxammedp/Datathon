import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'deploy/ensemble/media_tracking.py'
spec = importlib.util.spec_from_file_location('ensemble_media_tracking', SOURCE)
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)


class Capture:
    def __init__(self, total=540, fps=30., actual=None, shape=(24, 64, 3)):
        self.total, self.fps = total, fps
        self.actual = total if actual is None else actual
        self.shape, self.position, self.released = shape, 0, False

    def isOpened(self):
        return True

    def set(self, *args):
        pytest.fail('Sequential decoder must never seek')

    def get(self, property_id):
        import cv2
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.total
        pytest.fail('Decoder position metadata must not assign frame identity')

    def read(self):
        import numpy as np
        if self.position >= self.actual:
            return False, None
        frame = np.full(self.shape, self.position % 256, dtype=np.uint8)
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


def install_capture(tmp_path, monkeypatch, capture):
    import cv2
    path = tmp_path / 'source.mov'
    path.write_bytes(b'synthetic video placeholder; decoder supplied by test')
    monkeypatch.setattr(cv2, 'VideoCapture', lambda video: capture)
    return path


class Tracker:
    instances = []

    def __init__(self, *args):
        self.calls = 0
        self.instances.append(self)

    def update(self, frame):
        self.calls += 1
        return dict(tracks=[dict(track_id=7, box=[.1, .1, .3, .8], confidence=.8)],
                    detected_people=1, untracked_people=0)


def install_tracker(tmp_path, monkeypatch):
    Tracker.instances = []
    monkeypatch.setattr(media, '_PersonTracker', Tracker)
    path = tmp_path / 'detector.pt'
    path.write_bytes(b'local fixture weights')
    return path


@pytest.mark.parametrize('total,fps', [(1, 30.), (480, 30.), (481, 30.), (510, 29.97), (4273, 29.97),
                                       (108000, 30.), (37, 2.)])
def test_uniform_windows_have_bounded_inputs_and_full_timeline_coverage(total, fps):
    windows, sampled = media.plan_video(total, fps)
    assert windows[0]['start_sec'] == 0
    assert windows[-1]['end_sec'] == total / fps
    assert sampled == sorted(set(sampled))
    for window in windows:
        assert 1 <= len(window['frame_indices']) <= 64
        assert window['times'] == [i / fps for i in window['frame_indices']]
        assert all(window['start_frame'] <= i < window['end_frame_exclusive'] for i in window['frame_indices'])
    for first, second in zip(windows, windows[1:]):
        assert second['start_sec'] < first['end_sec']
        assert set(first['frame_indices']) & set(second['frame_indices'])
    if total > 100000:
        assert len(windows) > 200 and len(sampled) > 10000


def test_frame_budget_never_silently_reduces_sampling_density():
    with pytest.raises(ValueError, match='no thinning'):
        media.plan_video(900, 30., dict(sample_fps=8., max_frames_per_window=64))
    with pytest.raises(ValueError):
        media.plan_video(900, 30., dict(overlap_sec=16.))


def test_sequential_decode_tracks_once_per_shared_frame_and_preserves_unknown_roles(tmp_path, monkeypatch):
    capture = Capture()
    video = install_capture(tmp_path, monkeypatch, capture)
    detector = install_tracker(tmp_path, monkeypatch)
    output = tmp_path / 'media'
    result = media.prepare_video(video, output, detector_path=detector)
    assert capture.released and capture.position == 540
    assert result['decoded_frames'] == result['source_frames'] == 540
    assert result['sampled_frames'] == Tracker.instances[0].calls == 72
    assert len(result['windows']) == 2 and result['coverage_complete']
    first, second = result['windows']
    common = set(first['frame_indices']) & set(second['frame_indices'])
    assert len(common) == 16
    assert first['tracks'][0]['track_id'] == second['tracks'][0]['track_id'] == 7
    assert all(t['role'] == 'unknown' and t['role_source'] == 'unknown' for t in result['global_tracks'])
    assert not result['missing_track_is_departure'] and not result['actor_roles_verified']
    assert result['cashier_role_status'] == 'unknown' and not result['timestamps_verified_pts']
    assert all(not Path(f['path']).is_absolute() and (output / f['path']).exists() for f in result['frames'])
    before_calls = Tracker.instances[0].calls
    resumed = media.prepare_video(video, output, detector_path=detector)
    assert resumed == result and len(Tracker.instances) == 1 and Tracker.instances[0].calls == before_calls


def test_each_video_starts_a_fresh_tracker_and_zone_only_creates_candidate(tmp_path, monkeypatch):
    import cv2
    detector = install_tracker(tmp_path, monkeypatch)
    video = tmp_path / 'source.mov'
    video.write_bytes(b'fixture')
    captures = []

    def create(path):
        capture = Capture(total=60)
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, 'VideoCapture', create)
    camera = dict(cashier_zone=[0., 0., .5, 1.], counter_crop=[.2, .4, .8, .9])
    first = media.prepare_video(video, tmp_path / 'first', camera=camera, detector_path=detector)
    second = media.prepare_video(video, tmp_path / 'second', detector_path=detector)
    assert len(Tracker.instances) == 2 and all(c.released for c in captures)
    assert first['global_tracks'][0]['role'] == 'cashier_candidate'
    assert first['global_tracks'][0]['role_source'] == 'configured_zone'
    assert first['windows'][0]['counter_crop'] == camera['counter_crop']
    assert second['global_tracks'][0]['role'] == 'unknown'


def test_real_video_cache_keeps_aspect_ratio_and_detects_tampering(tmp_path):
    import cv2
    import numpy as np
    video = tmp_path / 'synthetic.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 120))
    assert writer.isOpened()
    for index in range(30):
        frame = np.zeros((120, 320, 3), dtype=np.uint8)
        frame[10:30, index:index + 15] = 200
        writer.write(frame)
    writer.release()
    output = tmp_path / 'media'
    result = media.prepare_video(video, output, dict(max_edge=160, image_format='png'))
    assert result['decoded_frames'] == 30 and result['sampled_frames'] == 12
    assert not result['tracking_enabled'] and result['global_tracks'] == []
    assert all(frame['detected_people'] is None for frame in result['frames'])
    frame = result['frames'][0]
    assert (frame['width'], frame['height']) == (160, 60)
    path = output / frame['path']
    assert cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR).shape == (60, 160, 3)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        media.prepare_video(video, output, dict(max_edge=160, image_format='png'))


@pytest.mark.parametrize('actual', [3, 11])
def test_failed_or_underreported_decode_is_never_complete(tmp_path, monkeypatch, actual):
    capture = Capture(total=10, actual=actual)
    video = install_capture(tmp_path, monkeypatch, capture)
    output = tmp_path / 'media'
    with pytest.raises(ValueError, match='decoding stopped|exceed reported'):
        media.prepare_video(video, output)
    failure = json.loads((output / 'failure.json').read_text())
    assert not failure['coverage_complete'] and not (output / 'index.json').exists()
    assert failure['decoded_frames'] == min(actual, 10)
    assert capture.released


@pytest.mark.parametrize('fps', [float('nan'), float('inf'), 0., -30.])
def test_invalid_fps_fails_before_model_or_frame_work(tmp_path, monkeypatch, fps):
    capture = Capture(fps=fps)
    video = install_capture(tmp_path, monkeypatch, capture)
    with pytest.raises(ValueError, match='FPS'):
        media.prepare_video(video, tmp_path / 'media')
    assert capture.position == 0 and capture.released


@pytest.mark.parametrize('box', [[0, 0, 1.1, 1], [-1, 0, 1, 1], [.5, 0, .5, 1], [0, 1, 1, 0],
                               [0, 0, float('nan'), 1], [0, 0, 1], [False, 0, 1, 1]])
def test_normalized_camera_boxes_are_validated(box):
    for key in ('cashier_zone', 'counter_crop'):
        with pytest.raises(ValueError):
            media.validate_camera({key: box})


def test_unchanged_inputs_required_for_cache_resume(tmp_path, monkeypatch):
    video = install_capture(tmp_path, monkeypatch, Capture(total=10))
    output = tmp_path / 'media'
    media.prepare_video(video, output)
    with pytest.raises(ValueError, match='changed'):
        media.prepare_video(video, output, camera={'cashier_zone': [0, 0, 1, 1]})
    video.write_bytes(b'new source')
    with pytest.raises(ValueError, match='changed'):
        media.prepare_video(video, output)


def test_tracker_uses_local_yolo_bytetrack_offline_and_persistent_ids(tmp_path, monkeypatch):
    import numpy as np
    updates, models = [], []
    yolo_settings = {'sync': True, 'hub': True, 'wandb': True}

    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    class Boxes:
        id, xyxy, conf = Tensor([4.]), Tensor([[10, 5, 30, 45]]), Tensor([.8])

        def __len__(self):
            return 1

    class YOLO:
        def __init__(self, path, task):
            assert Path(path).is_file() and task == 'detect'
            models.append(self)

        def track(self, **kwargs):
            updates.append(kwargs)
            return [SimpleNamespace(boxes=Boxes())]

    utils = SimpleNamespace(ONLINE=True, AUTOINSTALL=True)
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=YOLO, settings=yolo_settings, utils=utils))
    model = tmp_path / 'yolo.pt'
    model.write_bytes(b'fixture')
    tracker = media._PersonTracker(model, tmp_path, media._settings(dict(device='cuda:0')))
    first = tracker.update(np.zeros((50, 100, 3), dtype=np.uint8))
    tracker.update(np.zeros((50, 100, 3), dtype=np.uint8))
    assert first['tracks'][0]['track_id'] == 4 and first['tracks'][0]['box'] == [.1, .1, .3, .9]
    assert len(models) == 1 and len(updates) == 2
    assert all(u['persist'] and u['classes'] == [0] and not u['save'] and not u['show'] for u in updates)
    assert 'tracker_type: bytetrack' in Path(updates[0]['tracker']).read_text()
    assert not utils.ONLINE and not utils.AUTOINSTALL and not any(yolo_settings.values())
    assert media.os.environ['YOLO_OFFLINE'] == 'true'
    with pytest.raises(ValueError, match='local'):
        media._PersonTracker(tmp_path / 'missing.pt', tmp_path, media._settings())


def test_disappearing_track_never_becomes_departure_event(tmp_path, monkeypatch):
    class Disappearing(Tracker):
        def update(self, frame):
            result = super().update(frame)
            if self.calls > 3:
                return dict(tracks=[], detected_people=0, untracked_people=0)
            return result

    video = install_capture(tmp_path, monkeypatch, Capture(total=60))
    detector = install_tracker(tmp_path, monkeypatch)
    monkeypatch.setattr(media, '_PersonTracker', Disappearing)
    result = media.prepare_video(video, tmp_path / 'media', detector_path=detector)
    assert len(result['global_tracks'][0]['samples']) == 3
    assert not result['missing_track_is_departure'] and not result['automatic_verdict_enabled']
    assert 'violations' not in result


def test_cashier_candidate_persists_when_same_track_moves_out_of_zone(tmp_path, monkeypatch):
    class Leaving(Tracker):
        def update(self, frame):
            result = super().update(frame)
            if self.calls > 3:
                result['tracks'][0]['box'] = [.7, .1, .9, .8]
            return result

    video = install_capture(tmp_path, monkeypatch, Capture(total=80, fps=10.))
    detector = install_tracker(tmp_path, monkeypatch)
    monkeypatch.setattr(media, '_PersonTracker', Leaving)
    result = media.prepare_video(video, tmp_path / 'media', detector_path=detector,
                                 camera={'cashier_zone': [0., 0., .5, 1.]},
                                 settings=dict(window_sec=2., overlap_sec=1., sample_fps=4., max_frames_per_window=8))
    roles = [window['tracks'][0] for window in result['windows']]
    assert all(row['role'] == 'cashier_candidate' and row['track_id'] == 7 for row in roles)
    assert all(row['role_established_sec'] == .5 for row in roles)
    assert roles[-1]['zone_observation_fraction'] == 0.
    assert result['global_tracks'][0]['role'] == 'cashier_candidate'
    assert not result['missing_track_is_departure'] and not result['actor_roles_verified']


def test_late_role_evidence_does_not_relabel_earlier_overlapping_windows_or_count_overlap_twice(tmp_path, monkeypatch):
    class LateEntry(Tracker):
        def update(self, frame):
            result = super().update(frame)
            if self.calls <= 4 or self.calls > 24:
                result['tracks'][0]['box'] = [.7, .1, .9, .8]
            return result

    video = install_capture(tmp_path, monkeypatch, Capture(total=120, fps=10.))
    detector = install_tracker(tmp_path, monkeypatch)
    monkeypatch.setattr(media, '_PersonTracker', LateEntry)
    result = media.prepare_video(video, tmp_path / 'media', detector_path=detector,
                                 camera={'cashier_zone': [0., 0., .5, 1.]},
                                 settings=dict(window_sec=2., overlap_sec=1., sample_fps=4., max_frames_per_window=8))
    established = result['global_tracks'][0]['role_established_sec']
    assert established == 4.8
    assert Tracker.instances[0].calls == result['sampled_frames'] == 48
    for window in result['windows']:
        track = window['tracks'][0]
        if window['times'][-1] < established:
            assert track['role'] == 'unknown' and track['role_established_sec'] is None
        else:
            assert track['role'] == 'cashier_candidate' and track['role_established_sec'] == established
    assert result['windows'][1]['tracks'][0]['zone_observation_fraction'] == 1.
    assert result['windows'][1]['tracks'][0]['role'] == 'unknown'
    assert result['windows'][-1]['tracks'][0]['zone_observation_fraction'] == 0.
    before = copy.deepcopy(result['windows'][0])
    assert media.prepare_video(video, tmp_path / 'media', detector_path=detector,
                               camera={'cashier_zone': [0., 0., .5, 1.]},
                               settings=dict(window_sec=2., overlap_sec=1., sample_fps=4., max_frames_per_window=8))['windows'][0] == before


def test_candidate_role_memory_resets_for_new_video_with_same_tracker_id_and_same_zone(tmp_path, monkeypatch):
    import cv2
    class SecondVideoOutside(Tracker):
        created = 0

        def __init__(self, *args):
            super().__init__(*args)
            type(self).created += 1
            self.outside = type(self).created == 2

        def update(self, frame):
            result = super().update(frame)
            if self.outside:
                result['tracks'][0]['box'] = [.7, .1, .9, .8]
            return result

    video = tmp_path / 'source.mov'
    video.write_bytes(b'synthetic fixture')
    detector = install_tracker(tmp_path, monkeypatch)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda path: Capture(total=60))
    monkeypatch.setattr(media, '_PersonTracker', SecondVideoOutside)
    camera = {'cashier_zone': [0., 0., .5, 1.]}
    first = media.prepare_video(video, tmp_path / 'first', camera=camera, detector_path=detector)
    second = media.prepare_video(video, tmp_path / 'second', camera=camera, detector_path=detector)
    assert first['global_tracks'][0]['role'] == 'cashier_candidate'
    assert second['global_tracks'][0]['role'] == 'unknown'
    assert second['global_tracks'][0]['role_established_sec'] is None
