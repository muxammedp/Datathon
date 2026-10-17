import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from deploy.simple_video import analyzer


class Capture:
    def __init__(self, declared=320, actual=320, fps=2.0):
        self.declared = declared
        self.actual = actual
        self.fps = fps
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, field):
        return {cv2.CAP_PROP_FRAME_COUNT: self.declared, cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_FRAME_HEIGHT: 64, cv2.CAP_PROP_FRAME_WIDTH: 32}.get(field, 0)

    def read(self):
        if self.position >= self.actual:
            return False, None
        frame = np.full((64, 32, 3), self.position % 255, dtype=np.uint8)
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


def prepare(monkeypatch, tmp_path, capture, **settings):
    path = tmp_path / 'input.mov'
    path.touch()
    monkeypatch.setattr(analyzer.cv2, 'VideoCapture', lambda _: capture)
    return analyzer.prepare_video(path, dict(max_edge=64, max_pixels=4096, **settings))


def test_long_video_keeps_target_sampling_and_covers_final_frame(monkeypatch, tmp_path):
    capture = Capture()
    clips, metadata = prepare(monkeypatch, tmp_path, capture)
    assert capture.released
    assert metadata['decoded_frames'] == metadata['source_frames'] == 320
    assert len(clips) == 2
    assert len(clips[0]['source_indices']) > 120
    assert clips[0]['source_indices'][0] == 0
    assert clips[-1]['source_indices'][-1] == 319
    assert all(len(clip['source_indices']) % 2 == 0 for clip in clips)
    assert clips[1]['start_sec'] < clips[0]['end_sec']
    assert metadata['max_sample_gap_sec'] <= 1.0
    assert not metadata['presentation_timestamps_verified']
    assert not metadata['every_source_frame_seen_by_model']
    for clip in clips:
        assert all(t == i / 2 for t, i in zip(clip['times'], clip['source_indices']))
        assert len(clip['display_times']) * 2 == len(clip['times'])


@pytest.mark.parametrize('declared,actual', [(320, 319), (320, 321)])
def test_decode_mismatch_never_silently_drops_frames(monkeypatch, tmp_path, declared, actual):
    capture = Capture(declared=declared, actual=actual)
    with pytest.raises(ValueError, match='Sequential decoding'):
        prepare(monkeypatch, tmp_path, capture)
    assert capture.released


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf')])
def test_bad_fps_is_rejected_without_assuming_default(monkeypatch, tmp_path, fps):
    capture = Capture(fps=fps)
    with pytest.raises(ValueError, match='FPS'):
        prepare(monkeypatch, tmp_path, capture)
    assert capture.released


def test_rgb_letterbox_preserves_shape_and_actual_frame_pixels(monkeypatch, tmp_path):
    clips, metadata = prepare(monkeypatch, tmp_path, Capture(declared=8, actual=8), sample_fps=2)
    assert metadata['canvas_height'] == 64
    assert metadata['canvas_width'] == 32
    assert clips[0]['rgb'].shape == (8, 64, 32, 3)
    assert clips[0]['rgb'][-1, 0, 0].tolist() == [7, 7, 7]


@pytest.fixture
def clip():
    return dict(start_sec=0.0, end_sec=3.0, source_indices=[0, 10, 20, 29],
                times=[0.0, 1.0, 2.0, 2.9], display_times=[0.5, 2.5])


def answer(**changes):
    data = dict(decision='violation', rule='cashier_left', time_sec=.5)
    return json.dumps(dict(data, **changes))


def test_sample_timestamp_maps_to_actual_frame_with_uncertainty(clip):
    parsed = analyzer.parse_response(answer(), clip)
    assert parsed['prediction'] == parsed['decision'] == 'violation'
    assert parsed['source_frame_index'] == 0
    assert parsed['source_time_sec'] == 0
    assert parsed['timestamp_mapping_error_sec'] == .5
    assert parsed['timestamp_pair_span_sec'] == 1
    assert parsed['event_frame_verified'] is False


def test_unknown_rule_needs_review_not_automatic_violation(clip):
    parsed = analyzer.parse_response(answer(decision='insufficient_data', rule='other'), clip)
    assert parsed['prediction'] == 'insufficient_data'
    assert parsed['review_required']
    with pytest.raises(analyzer.ResponseError):
        analyzer.parse_response(answer(rule='other'), clip)


def test_valid_negative_and_unlocated_uncertainty(clip):
    negative = analyzer.parse_response(answer(decision='no_violation', rule=None, time_sec=None), clip)
    assert negative['prediction'] == 'no_violation'
    assert negative['source_frame_index'] is None
    uncertain = analyzer.parse_response(answer(decision='insufficient_data', rule=None, time_sec=None), clip)
    assert uncertain['prediction'] == 'insufficient_data'


@pytest.mark.parametrize('changes', [
    dict(time_sec=1.5), dict(time_sec=99), dict(time_sec=-.01), dict(time_sec=True),
    dict(time_sec=float('nan')), dict(rule='invented_rule'), dict(rule=['cashier_left']),
    dict(decision=['violation']), dict(rule=None), dict(time_sec=None),
    dict(evidence=''),
    dict(decision='no_violation'), dict(decision='normal'),
    dict(decision='insufficient_data', rule=None, time_sec=.5),
])
def test_invalid_event_is_not_accepted(clip, changes):
    with pytest.raises(analyzer.ResponseError):
        analyzer.parse_response(answer(**changes), clip)


@pytest.mark.parametrize('text', [
    '{}', '[]', 'null', '{"decision":"violation","decision":"no_violation"}',
    '```json\n{}\n```', 'Here is JSON: {}', '{"decision":"violation"',
    '{"decision":"no_violation","rule":null,"time_sec":null,"extra":1}',
])
def test_parser_rejects_prose_duplicate_keys_extra_keys_and_truncation(clip, text):
    with pytest.raises(analyzer.ResponseError):
        analyzer.parse_response(text, clip)


@pytest.fixture(scope='module')
def native_tokenizer():
    from transformers import AutoTokenizer
    directory = Path(__file__).resolve().parents[1] / 'models' / 'Qwen3-VL-2B-Instruct'
    if not directory.is_dir():
        pytest.skip('Existing local Qwen tokenizer is not installed')
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)


def test_native_tokenizer_every_constrained_completion_is_valid_and_complete(native_tokenizer, clip):
    prompt = [12, 42, 74]
    constraint = analyzer.JsonDecisionConstraint(native_tokenizer, clip, len(prompt), native_tokenizer.eos_token_id, 96)
    supported_positive_rules = sum(not rule.get('full_sequence') for rule in analyzer.RULES.values())
    assert constraint.completion_count == 2 + len(clip['display_times']) * (supported_positive_rules + len(analyzer.RULES) + 1)
    assert constraint.max_completion_tokens <= 96
    seen = set()
    for tokens in constraint.token_sequences:
        assert tokens[-1] == native_tokenizer.eos_token_id
        prefix = list(prompt)
        for token in tokens:
            assert token in constraint(0, prefix)
            prefix.append(token)
        assert constraint(0, prefix) == [native_tokenizer.eos_token_id]
        text = native_tokenizer.decode(tokens[:-1], skip_special_tokens=True)
        parsed = analyzer.parse_response(text, clip)
        seen.add(parsed['prediction'])
        if parsed['prediction'] == 'no_violation':
            assert parsed['rule'] is None and parsed['time_sec'] is None
        if parsed['rule'] == 'other':
            assert parsed['prediction'] == 'insufficient_data'
        assert constraint(0, prefix[:-1]) == [native_tokenizer.eos_token_id]
    assert seen == {'violation', 'no_violation', 'insufficient_data'}


def test_native_tokenizer_constraints_reject_impossible_prefix_and_short_budget(native_tokenizer, clip):
    constraint = analyzer.JsonDecisionConstraint(native_tokenizer, clip, 1, native_tokenizer.eos_token_id, 96)
    with pytest.raises(analyzer.ResponseError, match='invalid_constrained_prefix'):
        constraint(0, [10, native_tokenizer.eos_token_id])
    with pytest.raises(analyzer.ResponseError, match='invalid_constrained_prefix'):
        constraint(1, [10])
    with pytest.raises(ValueError, match='Token budget'):
        analyzer.JsonDecisionConstraint(native_tokenizer, clip, 1, native_tokenizer.eos_token_id,
                                       constraint.max_completion_tokens - 1)
    exact = analyzer.JsonDecisionConstraint(native_tokenizer, clip, 1, native_tokenizer.eos_token_id,
                                           constraint.max_completion_tokens)
    assert exact.completion_count == constraint.completion_count


def test_native_tokenizer_negative_completion_cannot_add_a_rule_or_timestamp(native_tokenizer, clip):
    constraint = analyzer.JsonDecisionConstraint(native_tokenizer, clip, 0, native_tokenizer.eos_token_id, 96)
    negative = next(tokens for tokens in constraint.token_sequences
                    if json.loads(native_tokenizer.decode(tokens[:-1]))['decision'] == 'no_violation')
    assert constraint(0, negative[:-1]) == [native_tokenizer.eos_token_id]
    with pytest.raises(analyzer.ResponseError):
        constraint(0, list(negative[:-1]) + native_tokenizer.encode(',"rule":"cashier_left"', add_special_tokens=False))
