import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    monkeypatch.syspath_prepend(str(ROOT))
    run = importlib.import_module('deploy.ensemble.run')
    media = importlib.import_module('deploy.ensemble.media_tracking')
    action = importlib.import_module('deploy.ensemble.action_expert')
    qwen = importlib.import_module('deploy.ensemble.qwen_expert')
    dataset = tmp_path / 'input'
    dataset.mkdir()
    source = dataset / 'synthetic.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64, 48))
    assert writer.isOpened()
    for index in range(48):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[8:20, index % 40:index % 40 + 8] = 180
        writer.write(frame)
    writer.release()
    identity = run.file_hash(source)
    models = tmp_path / 'models'
    (models / 'detector').mkdir(parents=True)
    (models / 'detector/yolo11n.pt').write_bytes(b'test detector placeholder')
    settings = dict(window_sec=2., overlap_sec=.5, sample_fps=4., max_frames_per_window=8,
                    max_edge=96, device='cpu', qwen_frames=4, qwen_new_tokens=64)
    state = SimpleNamespace(tracker_calls=0, action_calls=0, action_released=0,
                            qwen_calls=0, qwen_previous=[], fail=False)

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, frame):
            state.tracker_calls += 1
            return dict(tracks=[], detected_people=0, untracked_people=0)

    class Action:
        def __init__(self, *args, **kwargs):
            pass

        def extract(self, index, root):
            state.action_calls += 1
            assert all((root / path).is_file() for window in index['windows'] for path in window['frame_paths'])
            return dict(video_sha256=index['video_sha256'], windows=[
                dict(index=w['index'], start_sec=w['start_sec'], end_sec=w['end_sec'], features=[.1, .2])
                for w in index['windows']])

        def release(self):
            state.action_released += 1

    class Qwen:
        def __init__(self, *args, **kwargs):
            pass

        def analyze(self, index, root, previous_results=None, on_window=None):
            state.qwen_calls += 1
            state.qwen_previous.append(copy.deepcopy(previous_results))
            payload = dict(video_sha256=index['video_sha256'], windows=[])
            for window in index['windows']:
                failed = state.fail and window['index'] == 1
                payload['windows'].append(dict(index=window['index'], start_sec=window['start_sec'],
                                               end_sec=window['end_sec'], status='failed' if failed else 'done',
                                               score=None if failed else .2,
                                               context_score=None if failed else .1,
                                               cash_score=None if failed else .3,
                                               review_required=True))
                if on_window:
                    on_window(payload)
            return payload

    monkeypatch.setattr(media, '_PersonTracker', Tracker)
    monkeypatch.setattr(action, 'ActionExpert', Action)
    monkeypatch.setattr(qwen, 'QwenExpert', Qwen)
    monkeypatch.setattr(run, 'verify_models', lambda path: {})
    monkeypatch.setattr(run, 'release_gpu', lambda: None)
    monkeypatch.setattr(run, 'event', lambda *args, **kwargs: None)
    return SimpleNamespace(module=run, dataset=dataset, source=source, identity=identity, models=models,
                           settings=settings, output=tmp_path / 'output', state=state,
                           rows=[dict(file=source.name, sha256=identity, label=1, event_times=[1.])])


def extract(fixture, **changes):
    values = dict(rows=fixture.rows, dataset=fixture.dataset, output=fixture.output,
                  models=fixture.models, settings=fixture.settings)
    values.update(changes)
    return fixture.module.extract(**values)


def test_actual_synthetic_media_runs_and_completed_expert_caches_resume_without_reprocessing(runner):
    protocol = extract(runner)
    first_counts = (runner.state.tracker_calls, runner.state.action_calls, runner.state.qwen_calls)
    assert first_counts[0] > 0 and first_counts[1:] == (1, 1)
    assert runner.state.action_released == 1
    complete = runner.module.read_json(runner.output / 'complete.json')
    assert complete['complete'] and complete['videos'] == [runner.identity]
    directory = runner.output / 'videos' / runner.identity
    media = runner.module.read_json(directory / 'media/index.json')
    assert media['decoded_frames'] == 48
    assert media['coverage_complete'] and len(media['windows']) == 3
    assert not media['global_tracks'] and media['cashier_role_status'] == 'unknown'
    assert runner.module.file_hash(runner.source) == runner.identity
    before = {name: runner.module.file_hash(directory / name) for name in ('action.json', 'qwen.json')}
    assert extract(runner) == protocol
    assert (runner.state.tracker_calls, runner.state.action_calls, runner.state.qwen_calls) == first_counts
    assert before == {name: runner.module.file_hash(directory / name) for name in before}
    assert runner.module.file_hash(runner.source) == runner.identity
    assert not (runner.output / '.ensemble.lock').exists()


def test_failed_specialist_persists_partial_and_retry_never_claims_normal_or_complete(runner):
    runner.state.fail = True
    with pytest.raises(ValueError, match='window failed'):
        extract(runner)
    directory = runner.output / 'videos' / runner.identity
    assert (directory / 'qwen.partial.json').is_file()
    assert not (directory / 'qwen.json').exists()
    assert not (runner.output / 'complete.json').exists()
    partial = runner.module.read_json(directory / 'qwen.partial.json')['payload']
    assert partial['windows'][1]['status'] == 'failed'
    assert partial['windows'][1]['score'] is None
    assert all('prediction' not in row for row in partial['windows'])
    assert not (runner.output / '.ensemble.lock').exists()
    runner.state.fail = False
    extract(runner)
    assert runner.state.action_calls == 1 and runner.state.qwen_calls == 2
    assert runner.state.qwen_previous[-1] == partial
    assert runner.module.read_json(runner.output / 'complete.json')['complete']
    assert runner.module.file_hash(runner.source) == runner.identity


def test_tampered_completed_expert_payload_is_rejected_and_completion_invalidated(runner):
    extract(runner)
    target = runner.output / 'videos' / runner.identity / 'action.json'
    value = runner.module.read_json(target)
    value['payload']['windows'][0]['features'][0] = 999.
    target.write_text(json.dumps(value), encoding='utf8')
    with pytest.raises(ValueError, match='checksum mismatch'):
        extract(runner)
    assert not (runner.output / 'complete.json').exists()
    assert runner.module.file_hash(runner.source) == runner.identity
    assert not (runner.output / '.ensemble.lock').exists()


def test_changed_settings_refuse_cache_and_preserve_existing_results(runner):
    extract(runner)
    original = runner.module.file_hash(runner.output / 'protocol.json')
    with pytest.raises(ValueError, match='Source/model/settings changed'):
        extract(runner, settings=dict(runner.settings, qwen_frames=8))
    assert runner.module.file_hash(runner.output / 'protocol.json') == original
    assert runner.module.file_hash(runner.source) == runner.identity


def test_payload_fingerprint_tampering_and_source_checksum_change_are_rejected(runner):
    extract(runner)
    target = runner.output / 'videos' / runner.identity / 'qwen.partial.json'
    value = runner.module.read_json(target)
    value['fingerprint'] = 'wrong-fingerprint'
    target.write_text(json.dumps(value), encoding='utf8')
    with pytest.raises(ValueError, match='identity/checksum mismatch'):
        runner.module.load_payload(target, runner.module.read_json(runner.output / 'protocol.json')['fingerprint'])
    runner.source.write_bytes(runner.source.read_bytes() + b'changed source')
    changed = runner.module.file_hash(runner.source)
    with pytest.raises(ValueError, match='Input path/checksum changed'):
        extract(runner)
    assert runner.module.file_hash(runner.source) == changed
    assert not (runner.output / 'complete.json').exists()


@pytest.mark.parametrize('args', [[], ['--dataset', 'data', '--video', 'video.mov', '--output', 'out'],
                                   ['--video', 'video.mov', '--output', 'out', '--qwen-frames', '0'],
                                   ['--video', 'video.mov', '--output', 'out', '--qwen-new-tokens', '99999']])
def test_cli_rejects_missing_conflicting_and_invalid_budget_arguments(monkeypatch, args):
    monkeypatch.syspath_prepend(str(ROOT))
    run = importlib.import_module('deploy.ensemble.run')
    monkeypatch.setattr('sys.argv', ['run.py'] + args)
    with pytest.raises(SystemExit) as caught:
        run.main()
    assert caught.value.code == 2


def test_cli_video_route_is_unlabelled_and_passes_bounded_settings(runner, monkeypatch):
    recorded = []
    monkeypatch.setattr(runner.module, 'extract', lambda *args: recorded.append(args))
    monkeypatch.setattr('sys.argv', ['run.py', '--video', str(runner.source), '--output', str(runner.output),
                                   '--models', str(runner.models), '--device', 'cpu', '--qwen-frames', '8',
                                   '--qwen-new-tokens', '128'])
    runner.module.main()
    rows, dataset, output, models, settings, camera = recorded[0]
    assert rows == [dict(file=runner.source.name, sha256=runner.identity)]
    assert 'label' not in rows[0]
    assert dataset == runner.source.parent
    assert settings['qwen_frames'] == 8 and settings['qwen_new_tokens'] == 128
    assert camera is None


def test_output_lock_refuses_parallel_mutation_and_releases_after_failure(tmp_path):
    from deploy.ensemble.common import exclusive_output
    with pytest.raises(RuntimeError, match='synthetic processing error'):
        with exclusive_output(tmp_path):
            with pytest.raises(RuntimeError, match='Output is locked'):
                with exclusive_output(tmp_path):
                    pytest.fail('Second writer acquired the same output lock')
            raise RuntimeError('synthetic processing error')
    assert not (tmp_path / '.ensemble.lock').exists()
