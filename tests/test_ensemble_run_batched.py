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
    run = importlib.import_module('deploy.ensemble.run_batched')
    old = importlib.import_module('deploy.ensemble.run')
    media = importlib.import_module('deploy.ensemble.media_tracking')
    action = importlib.import_module('deploy.ensemble.action_expert')
    qwen = importlib.import_module('deploy.ensemble.qwen_expert')
    batched = importlib.import_module('deploy.ensemble.qwen_batched')
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
    state = SimpleNamespace(tracker_calls=0, action_calls=0, batch_calls=0, old_calls=0,
                            loads=0, releases=0, previous=[], events=[], fail=False)

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
            result = []
            for window in index['windows']:
                count = len(window['frame_paths'])
                clips = action.clip_indices(count)
                result.append(dict(index=window['index'], start_sec=window['start_sec'], end_sec=window['end_sec'],
                                   features=[.1] * action.FEATURE_SIZE, crop_used=False, selected_times=window['times'],
                                   sampled_frames=count, clips_processed=len(clips), covered_sample_indices=list(range(count)),
                                   all_supplied_samples_covered=True, clips=[dict(sample_indices=c) for c in clips]))
            return dict(version=action.VERSION, video_sha256=index['video_sha256'], windows=result,
                        model_identifier=action.MODEL_ID, feature_dimensions=action.FEATURE_SIZE,
                        all_windows_processed=True, temporal_frames_per_clip=8, temporal_stride=4)

        def release(self):
            pass

    class OldQwen:
        def __init__(self, *args, **kwargs):
            self.max_frames = kwargs['max_frames']
            self.max_new_tokens = kwargs['max_new_tokens']

        def analyze(self, index, root, previous_results=None, on_window=None):
            state.old_calls += 1
            return dict(version=qwen.VERSION, video_sha256=index['video_sha256'], windows=[
                dict(index=w['index'], start_sec=w['start_sec'], end_sec=w['end_sec'], status='done',
                     score=-999., context_score=-999., cash_score=-999.) for w in index['windows']])

    class BatchQwen(OldQwen):
        def _load(self):
            state.loads += 1

        def release(self):
            state.releases += 1

        def analyze(self, index, root, previous_results=None, on_window=None):
            state.batch_calls += 1
            state.previous.append(copy.deepcopy(previous_results))
            payload = dict(version=batched.VERSION, video_sha256=index['video_sha256'], windows=[],
                           settings=dict(max_frames=self.max_frames, max_new_tokens=self.max_new_tokens,
                                         max_pixels=qwen.MAX_PIXELS, view_batch_size=2, execution='batched_views_v2',
                                         observations_version='observations_v2', max_observation_events=64))
            for window in index['windows']:
                failed = state.fail and window['index'] == 1
                payload['windows'].append(dict(index=window['index'], start_sec=window['start_sec'],
                                               end_sec=window['end_sec'], status='failed' if failed else 'done',
                                               score=None if failed else .2, context_score=None if failed else .1,
                                               cash_score=None if failed else .3, elapsed_sec=1.234,
                                               profile=dict(execution='batched_views_v2', view_batch_size=2)))
                if on_window:
                    on_window(payload)
            return payload

    monkeypatch.setattr(media, '_PersonTracker', Tracker)
    monkeypatch.setattr(action, 'ActionExpert', Action)
    monkeypatch.setattr(qwen, 'QwenExpert', OldQwen)
    monkeypatch.setattr(batched, 'BatchedQwenExpert', BatchQwen)
    for module in (run, old):
        monkeypatch.setattr(module, 'verify_models', lambda path: {})
        monkeypatch.setattr(module, 'release_gpu', lambda: None)
        monkeypatch.setattr(module, 'event', lambda stage, **kwargs: state.events.append(dict(stage=stage, **kwargs)))
    return SimpleNamespace(module=run, old=old, dataset=dataset, source=source, identity=identity, models=models,
                           settings=settings, output=tmp_path / 'output', old_output=tmp_path / 'old', state=state,
                           rows=[dict(file=source.name, sha256=identity, label=1, event_times=[1.])])


def extract(runner, **changes):
    values = dict(rows=runner.rows, dataset=runner.dataset, output=runner.output,
                  models=runner.models, settings=runner.settings)
    values.update(changes)
    return runner.module.extract(**values)


def original(runner):
    runner.old.extract(runner.rows, runner.dataset, runner.old_output, runner.models, runner.settings)
    return runner.old_output / 'videos' / runner.identity


def hashes(root):
    from deploy.ensemble.common import file_hash
    return {path.relative_to(root).as_posix(): file_hash(path) for path in root.rglob('*') if path.is_file()}


def test_real_media_action_reused_as_independent_copies_but_old_qwen_never_reused(runner):
    old_dir = original(runner)
    before = hashes(runner.old_output)
    code_before = {name: runner.module.file_hash(ROOT / 'deploy/ensemble' / name)
                   for name in ('common.py', 'run.py', 'media_tracking.py', 'action_expert.py', 'qwen_expert.py')}
    tracker, action = runner.state.tracker_calls, runner.state.action_calls
    fingerprint = extract(runner, reuse_from=runner.old_output)
    assert fingerprint['version'] == 'forte_specialists_batched_v2'
    assert set(fingerprint['source_hashes']) == {*code_before, 'run_batched.py', 'qwen_batched.py', 'observation_schema.py'}
    assert runner.state.tracker_calls == tracker and runner.state.action_calls == action
    assert runner.state.batch_calls == 1 and runner.state.previous == [None]
    assert before == hashes(runner.old_output)
    new_dir = runner.output / 'videos' / runner.identity
    payload = runner.module.load_payload(new_dir / 'qwen.json', fingerprint['fingerprint'])
    assert payload['version'] == 'qwen_batched_v2' and payload['windows'][0]['score'] == .2
    assert hashes(old_dir / 'media') == hashes(new_dir / 'media')
    copied = next((new_dir / 'media/frames').iterdir())
    copied.write_bytes(b'copy isolation verified')
    assert before == hashes(runner.old_output)
    assert code_before == {name: runner.module.file_hash(ROOT / 'deploy/ensemble' / name) for name in code_before}


def test_fresh_batch_cache_resumes_without_loading_or_recomputing_models(runner):
    fingerprint = extract(runner)
    counts = (runner.state.tracker_calls, runner.state.action_calls, runner.state.batch_calls, runner.state.loads)
    assert counts[0] > 0 and counts[1:] == (1, 1, 1)
    assert extract(runner) == fingerprint
    assert counts == (runner.state.tracker_calls, runner.state.action_calls, runner.state.batch_calls, runner.state.loads)
    assert runner.module.read_json(runner.output / 'complete.json')['complete']
    assert not (runner.output / '.ensemble.lock').exists()


def test_window_and_load_times_are_reported_separately(runner):
    extract(runner)
    stages = [row['stage'] for row in runner.state.events]
    assert stages.index('qwen_model_loaded') < stages.index('qwen_window')
    windows = [row for row in runner.state.events if row['stage'] == 'qwen_window']
    assert len(windows) == 3 and all(row['elapsed_sec'] == 1.234 for row in windows)
    assert all('video_elapsed_sec' in row and row['profile']['view_batch_size'] == 2 for row in windows)
    assert next(row for row in runner.state.events if row['stage'] == 'qwen_ready')['model_load_included'] is False


@pytest.mark.parametrize('field', ['version', 'settings', 'source_hashes'])
def test_original_protocol_mismatch_refused_before_output_created(runner, field):
    original(runner)
    path = runner.old_output / 'protocol.json'
    value = runner.module.read_json(path)
    value[field] = 'changed'
    runner.module.save_json(path, value)
    before = hashes(runner.old_output)
    with pytest.raises(ValueError, match='exactly match'):
        extract(runner, reuse_from=runner.old_output)
    assert not runner.output.exists() and before == hashes(runner.old_output)


@pytest.mark.parametrize('direction', ['same', 'child', 'parent'])
def test_source_output_alias_and_nesting_are_rejected(runner, direction):
    original(runner)
    destination = {'same': runner.old_output, 'child': runner.old_output / 'child',
                   'parent': runner.old_output.parent}[direction]
    before = hashes(runner.old_output)
    with pytest.raises(ValueError, match='separate non-nested'):
        extract(runner, output=destination, reuse_from=runner.old_output)
    assert before == hashes(runner.old_output)


def test_damaged_imported_frame_rejected_by_actual_media_resume(runner):
    old_dir = original(runner)
    frame = next((old_dir / 'media/frames').iterdir())
    frame.write_bytes(b'changed pixels')
    before = hashes(runner.old_output)
    with pytest.raises(ValueError, match='frame checksum mismatch'):
        extract(runner, reuse_from=runner.old_output)
    assert before == hashes(runner.old_output) and runner.state.batch_calls == 0
    assert not (runner.output / 'complete.json').exists()


@pytest.mark.parametrize('rewrap', [False, True])
def test_damaged_action_checksum_or_feature_schema_refused(runner, rewrap):
    old_dir = original(runner)
    target = old_dir / 'action.json'
    value = runner.module.read_json(target)
    value['payload']['windows'][0]['features'] = [42.]
    if rewrap:
        runner.module.save_payload(target, value['payload'], value['fingerprint'])
    else:
        runner.module.save_json(target, value)
    before = hashes(runner.old_output)
    with pytest.raises(ValueError, match='invalid feature|checksum mismatch'):
        extract(runner, reuse_from=runner.old_output)
    assert before == hashes(runner.old_output) and runner.state.batch_calls == 0


def test_source_video_checksum_change_refused(runner):
    original(runner)
    runner.source.write_bytes(runner.source.read_bytes() + b'changed')
    before = hashes(runner.old_output)
    with pytest.raises(ValueError, match='Input path/checksum'):
        extract(runner, reuse_from=runner.old_output)
    assert before == hashes(runner.old_output) and runner.state.batch_calls == 0


def test_incomplete_original_media_is_not_imported(runner):
    old_dir = original(runner)
    (old_dir / 'media/complete.json').unlink()
    tracker = runner.state.tracker_calls
    before = hashes(runner.old_output)
    extract(runner, reuse_from=runner.old_output)
    assert runner.state.tracker_calls > tracker and before == hashes(runner.old_output)


def test_failed_batch_persists_partial_and_retry_receives_only_new_backend_partial(runner):
    original(runner)
    runner.state.fail = True
    with pytest.raises(ValueError, match='window failed'):
        extract(runner, reuse_from=runner.old_output)
    partial_path = runner.output / 'videos' / runner.identity / 'qwen.partial.json'
    partial = runner.module.read_json(partial_path)['payload']
    assert partial['version'] == 'qwen_batched_v2' and partial['windows'][1]['score'] is None
    assert not (runner.output / 'complete.json').exists()
    assert not (runner.output / '.ensemble.lock').exists()
    runner.state.fail = False
    runner.state.events.clear()
    extract(runner, reuse_from=runner.old_output)
    assert runner.state.previous == [None, partial]
    windows = [row for row in runner.state.events if row['stage'] == 'qwen_window']
    assert [row['reused'] for row in windows] == [True, False, False]
    assert windows[0]['elapsed_sec'] == 0. and windows[0]['cached_elapsed_sec'] == 1.234
    assert windows[1]['elapsed_sec'] == 1.234 and windows[1]['cached_elapsed_sec'] is None


@pytest.mark.parametrize('field', ['max_pixels', 'execution'])
def test_batched_cache_requires_exact_pixel_and_execution_settings(runner, field):
    fingerprint = extract(runner)
    target = runner.output / 'videos' / runner.identity / 'qwen.json'
    value = runner.module.read_json(target)['payload']
    value['settings'][field] = 'changed'
    runner.module.save_payload(target, value, fingerprint['fingerprint'])
    with pytest.raises(ValueError, match='batch settings'):
        extract(runner)


def test_rewrapped_old_qwen_is_rejected_even_with_new_cache_fingerprint(runner):
    old_dir = original(runner)
    fingerprint = extract(runner, reuse_from=runner.old_output)
    old_payload = runner.module.read_json(old_dir / 'qwen.json')['payload']
    target = runner.output / 'videos' / runner.identity / 'qwen.json'
    runner.module.save_payload(target, old_payload, fingerprint['fingerprint'])
    with pytest.raises(ValueError, match='batched backend'):
        extract(runner)
    assert not (runner.output / 'complete.json').exists()


def test_cli_passes_reuse_source_and_same_budgets(runner, monkeypatch):
    recorded = []
    monkeypatch.setattr(runner.module, 'extract', lambda *args, **kwargs: recorded.append((args, kwargs)))
    monkeypatch.setattr('sys.argv', ['run_batched.py', '--video', str(runner.source), '--output', str(runner.output),
                                   '--reuse-from', str(runner.old_output), '--device', 'cpu'])
    runner.module.main()
    args, kwargs = recorded[0]
    assert args[0] == [dict(file=runner.source.name, sha256=runner.identity)]
    assert args[4]['qwen_frames'] == 16 and args[4]['qwen_new_tokens'] == 384
    assert kwargs['reuse_from'] == runner.old_output


@pytest.mark.parametrize('args', [[], ['--video', 'x.mov', '--output', 'out', '--qwen-frames', '0'],
                                   ['--video', 'x.mov', '--output', 'out', '--qwen-new-tokens', '900']])
def test_invalid_cli_budgets_refused(runner, monkeypatch, args):
    monkeypatch.setattr('sys.argv', ['run_batched.py', *args])
    with pytest.raises(SystemExit) as caught:
        runner.module.main()
    assert caught.value.code == 2
