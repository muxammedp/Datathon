import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prediction(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    monkeypatch.syspath_prepend(str(ROOT))
    predict = importlib.import_module('deploy.ensemble.predict')
    run = importlib.import_module('deploy.ensemble.run')
    fusion = importlib.import_module('deploy.ensemble.fusion')
    common = importlib.import_module('deploy.ensemble.common')
    source = tmp_path / 'synthetic.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64, 48))
    assert writer.isOpened()
    for position in range(30):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[10:20, position:position + 8] = 200
        writer.write(frame)
    writer.release()
    identity = common.file_hash(source)
    settings = dict(window_sec=16., overlap_sec=4., sample_fps=4., max_frames_per_window=64,
                    max_edge=848, device='cpu', qwen_frames=16, qwen_new_tokens=384)
    config = run.protocol(settings, None)

    def head(dimensions, bias=0.):
        return dict(dimensions=dimensions, mean=[0.] * dimensions, scale=[1.] * dimensions,
                    coef=[1.] + [0.] * (dimensions - 1), intercept=bias)

    bundle = dict(version=fusion.VERSION, feature_names=fusion.FEATURE_NAMES,
                  action_model=head(fusion.ACTION_DIMENSIONS), meta_model=head(6, -2.), threshold=.5,
                  extraction_protocol=config, fingerprint=config['fingerprint'],
                  train_sha256=[], calibration_sha256=[], test_sha256=[])
    bundle_path = tmp_path / 'predictor.json'
    common.save_json(bundle_path, bundle)
    args = SimpleNamespace(video=source, bundle=bundle_path, models=tmp_path / 'models',
                           cache=tmp_path / 'cache', output=tmp_path / 'result.json', timeout=300., same_camera=False)
    directory = args.cache / 'videos' / identity
    media_root = directory / 'media'
    media = dict(video_sha256=identity, coverage_complete=True, decoded_frames=30, source_frames=30,
                 sampled_frames=4, max_sample_gap_sec=1., windows=[])
    actions, language = [], []
    for number, times in enumerate(([0., 1.], [1., 2.])):
        paths, checksums = [], []
        for position, when in enumerate(times):
            path = media_root / f'frames/{number}-{position}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode('.png', np.full((48, 64, 3), number * 60 + position, np.uint8))
            assert ok
            path.write_bytes(encoded.tobytes())
            paths.append(path.relative_to(media_root).as_posix())
            checksums.append(common.file_hash(path))
        bounds = dict(index=number, start_sec=float(number), end_sec=float(number + 2))
        media['windows'].append(dict(bounds, frame_paths=paths, frame_sha256=checksums,
                                     frame_indices=[int(t * 10) for t in times], times=list(times)))
        actions.append(dict(bounds, features=[float(number)] + [0.] * (fusion.ACTION_DIMENSIONS - 1)))
        language.append(dict(bounds, status='done', context_score=-.2, cash_score=.3,
                             review_required=True, review_reasons=['roles_unverified'],
                             observations={'context': {'events': []}, 'cash': {'events': []}}))
    language[1]['observations']['cash']['events'] = [dict(kind='cash_handover_visible', time_sec=2., track_ids=[])]
    action = dict(video_sha256=identity, windows=actions)
    qwen = dict(video_sha256=identity, windows=language)
    state = SimpleNamespace(processes=[], mutation=None, timeout=False, lock_owner=None)

    def write_extraction():
        action_value, qwen_value, media_value = copy.deepcopy(action), copy.deepcopy(qwen), copy.deepcopy(media)
        if state.mutation:
            state.mutation(action_value, qwen_value, media_value)
        common.save_json(args.cache / 'protocol.json', config)
        common.save_json(args.cache / 'complete.json', dict(fingerprint=config['fingerprint'], videos=[identity], complete=True))
        common.save_payload(directory / 'action.json', action_value, config['fingerprint'])
        common.save_payload(directory / 'qwen.json', qwen_value, config['fingerprint'])
        common.save_json(media_root / 'index.json', media_value)
        common.save_json(media_root / 'complete.json', dict(index_sha256=common.file_hash(media_root / 'index.json')))

    class Process:
        pid = 12345

        def __init__(self, command):
            self.command, self.killed, self.waited = command, False, False
            state.processes.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self, timeout=None):
            if state.timeout and not self.killed:
                (args.cache / '.ensemble.lock').write_text(str(state.lock_owner or self.pid))
                raise subprocess.TimeoutExpired(self.command, timeout)
            if not self.killed and not self.waited:
                write_extraction()
            self.waited = True
            return 0

        def kill(self):
            self.killed = True

    monkeypatch.setattr(predict.subprocess, 'Popen', Process)
    return SimpleNamespace(module=predict, common=common, fusion=fusion, args=args, bundle=bundle,
                           media=media, qwen=qwen, action=action, media_root=media_root,
                           identity=identity, state=state)


def cli(fixture, monkeypatch, *extra):
    args = fixture.args
    monkeypatch.setattr(sys, 'argv', ['predict.py', '--video', str(args.video), '--bundle', str(args.bundle),
                                    '--cache', str(args.cache), '--models', str(args.models),
                                    '--output', str(args.output), *extra])
    fixture.module.main()
    return fixture.common.read_json(args.output)


def test_cli_completes_synthetic_video_with_real_fusion_and_exact_existing_evidence(prediction, monkeypatch):
    result = cli(prediction, monkeypatch)
    expected = prediction.fusion.predict_record(prediction.bundle, prediction.action, prediction.qwen)
    assert result['complete'] and result['within_time_budget']
    assert result['scores']['score'] == pytest.approx(expected['score'])
    assert result['prediction'] == 'no_violation'
    assert result['review_reasons'] == ['roles_unverified']
    assert result['scores']['specialist_review_required']
    assert not result['production_approved'] and not result['actor_roles_verified']
    assert result['video_sha256'] == prediction.identity == prediction.common.file_hash(prediction.args.video)
    evidence = result['candidate_evidence']
    assert evidence['window_index'] == 1 and evidence['frame_index'] == 20 and evidence['time_sec'] == 2.
    assert Path(evidence['frame_path']) == (prediction.media_root / 'frames/1-1.png').resolve()
    assert not evidence['verified_violation'] and not evidence['timestamps_verified_pts']
    assert prediction.state.processes[0].command[2:4] == ['--video', str(prediction.args.video.resolve())]


def test_evidence_rejects_changed_image_and_path_escape(prediction):
    target = prediction.media_root / 'frames/1-1.png'
    target.write_bytes(b'changed after extraction')
    with pytest.raises(ValueError, match='missing or modified'):
        prediction.module.candidate_evidence(prediction.media, [.1, .9], prediction.qwen, prediction.media_root)
    outside = prediction.args.cache / 'outside.png'
    outside.write_bytes(b'outside')
    prediction.media['windows'][1]['frame_paths'][1] = '../../../outside.png'
    prediction.media['windows'][1]['frame_sha256'][1] = prediction.common.file_hash(outside)
    with pytest.raises(ValueError, match='missing or modified'):
        prediction.module.candidate_evidence(prediction.media, [.1, .9], prediction.qwen, prediction.media_root)


@pytest.mark.parametrize('foreign_lock', [False, True])
def test_timeout_kills_extractor_removes_only_its_owned_lock_and_never_returns_normal(prediction, monkeypatch, foreign_lock):
    prediction.state.timeout = True
    prediction.state.lock_owner = 9876 if foreign_lock else None
    with pytest.raises(SystemExit) as error:
        cli(prediction, monkeypatch)
    assert error.value.code == 1
    result = prediction.common.read_json(prediction.args.output)
    assert not result['complete'] and result['prediction'] == 'insufficient_data'
    assert result['within_time_budget'] is False and result['error_type'] == 'TimeoutExpired'
    process = prediction.state.processes[0]
    assert process.killed and process.waited
    lock = prediction.args.cache / '.ensemble.lock'
    assert lock.exists() == foreign_lock
    if foreign_lock:
        assert lock.read_text() == '9876'


@pytest.mark.parametrize('mutation', ['failed_window', 'wrong_identity', 'missing_coverage', 'changed_media_interval'])
def test_validation_failure_is_insufficient_data_never_normal(prediction, monkeypatch, mutation):
    def change(action, qwen, media):
        if mutation == 'failed_window':
            qwen['windows'][1]['status'] = 'failed'
        elif mutation == 'wrong_identity':
            action['video_sha256'] = qwen['video_sha256'] = media['video_sha256'] = 'b' * 64
        elif mutation == 'missing_coverage':
            media['coverage_complete'] = False
        elif mutation == 'changed_media_interval':
            media['windows'][1]['end_sec'] += .25
    prediction.state.mutation = change
    with pytest.raises(SystemExit) as error:
        cli(prediction, monkeypatch)
    assert error.value.code == 1
    result = prediction.common.read_json(prediction.args.output)
    assert not result['complete'] and result['prediction'] == 'insufficient_data'
    assert result['error_type'] == 'ValueError'
    assert len(prediction.state.processes) == 1


def test_evidence_work_is_included_in_total_time_budget(prediction, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(prediction.module.time, 'monotonic', lambda: clock[0])
    original = prediction.module.candidate_evidence

    def slow_evidence(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 301.
        return result

    monkeypatch.setattr(prediction.module, 'candidate_evidence', slow_evidence)
    with pytest.raises(SystemExit):
        cli(prediction, monkeypatch)
    result = prediction.common.read_json(prediction.args.output)
    assert not result['complete'] and result['prediction'] == 'insufficient_data'
    assert not result['within_time_budget'] and result['error_type'] == 'TimeoutError'


def test_configured_geometry_requires_same_camera_before_any_extraction(prediction):
    run = importlib.import_module('deploy.ensemble.run')
    prediction.bundle['extraction_protocol'] = run.protocol(
        prediction.bundle['extraction_protocol']['settings'], {'cashier_zone': [0., 0., .5, 1.]})
    prediction.bundle['fingerprint'] = prediction.bundle['extraction_protocol']['fingerprint']
    prediction.common.save_json(prediction.args.bundle, prediction.bundle)
    with pytest.raises(ValueError, match='same-camera'):
        prediction.module.infer(prediction.args)
    assert prediction.state.processes == []


@pytest.mark.parametrize('timeout', ['0', '301', 'nan', 'inf'])
def test_cli_rejects_invalid_time_budget_before_extractor(prediction, monkeypatch, timeout):
    with pytest.raises(SystemExit) as error:
        cli(prediction, monkeypatch, '--timeout', timeout)
    assert error.value.code == 2 and not prediction.state.processes


@pytest.mark.parametrize('protected_input', ['video', 'bundle'])
def test_cli_cannot_overwrite_its_source_video_or_predictor(prediction, monkeypatch, protected_input):
    target = getattr(prediction.args, protected_input)
    before = prediction.common.file_hash(target)
    prediction.args.output = target
    with pytest.raises(SystemExit) as error:
        cli(prediction, monkeypatch)
    assert error.value.code == 2
    assert prediction.common.file_hash(target) == before
    assert not prediction.state.processes
