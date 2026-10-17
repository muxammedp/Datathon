import json

import pytest

from deploy.ensemble import observation_diagnostics as diagnostics
from deploy.ensemble import qwen_expert as base


def window():
    return dict(start_sec=0., end_sec=2., times=[0., 1.], tracks=[])


def valid():
    return dict(actors=[], service_state='unknown', cash_action='unknown', events=[])


def test_success_preserves_output_and_view_order_without_recording_scene_values():
    original = base.validate_observations
    with diagnostics.trace_validation() as records:
        assert base.validate_observations(json.dumps(valid()), window()) == valid()
        assert base.validate_observations(valid(), window()) == valid()
    assert base.validate_observations is original
    assert [row['view'] for row in records] == ['context', 'cash']
    assert all(row['code'] == 'valid' and row['strict_valid'] for row in records)
    assert records[0]['char_count'] == len(json.dumps(valid()))
    assert records[1]['char_count'] is None
    assert 'service_state' not in json.dumps(records)


def test_whole_fenced_json_is_diagnosed_but_never_accepted_or_mutated():
    raw = '```json\n' + json.dumps(valid()) + '\n```'
    with diagnostics.trace_validation() as records:
        with pytest.raises(json.JSONDecodeError):
            base.validate_observations(raw, window())
    row = records[0]
    assert not row['strict_valid'] and row['code'] == 'invalid_json'
    assert row['line'] == row['column'] == 1
    assert row['starts_with_json_fence'] and row['ends_with_fence']
    assert row['fence_probe'] == dict(attempted=True, valid_after_unwrap=True, code='valid')
    assert raw == '```json\n' + json.dumps(valid()) + '\n```'


def test_fenced_semantic_error_reports_fixed_code_and_still_raises_original_json_error():
    value = valid()
    value['service_state'] = 'private_scene_description'
    raw = '```json\n' + json.dumps(value) + '\n```'
    with diagnostics.trace_validation() as records:
        with pytest.raises(json.JSONDecodeError):
            base.validate_observations(raw, window())
    assert records[0]['fence_probe'] == dict(attempted=True, valid_after_unwrap=False, code='unknown_observation_enum')
    assert 'private_scene_description' not in json.dumps(records)


@pytest.mark.parametrize('raw,code', [
    ('{"private_scene_description":', 'invalid_json'),
    (json.dumps(dict(valid(), service_state='private_scene_description')), 'unknown_observation_enum'),
    (json.dumps(dict(valid(), cash_action=['private_scene_description'])), 'invalid_value_type'),
])
def test_malformed_or_wrong_enum_outputs_only_sanitized_metadata(raw, code):
    with diagnostics.trace_validation() as records:
        with pytest.raises((ValueError, TypeError)):
            base.validate_observations(raw, window())
    assert records[0]['code'] == code and not records[0]['strict_valid']
    assert 'private_scene_description' not in json.dumps(records)


@pytest.mark.parametrize('raw', [
    'Explanation containing private_scene_description\n```json\n{}\n```',
    '```json\n{}\n```\nprivate_scene_description',
    '```python\n{}\n```',
    '```json\n{}\n```\n```json\n{}\n```',
])
def test_only_an_entire_single_json_fence_is_probed(raw):
    with diagnostics.trace_validation() as records:
        with pytest.raises(ValueError):
            base.validate_observations(raw, window())
    assert not records[0]['fence_probe']['attempted']


def test_untrusted_exception_text_is_never_exported_and_same_exception_is_reraised(monkeypatch):
    class UntrustedError(Exception):
        def __str__(self):
            raise AssertionError('Exception text must not be inspected')

    failure = UntrustedError('private_scene_description')

    def explode(*args):
        raise failure

    monkeypatch.setattr(base, 'validate_observations', explode)
    with pytest.raises(UntrustedError) as error:
        with diagnostics.trace_validation() as records:
            base.validate_observations('private_scene_description', window())
    assert error.value is failure
    assert base.validate_observations is explode
    assert records[0]['code'] == 'unclassified_validation_error'
    assert 'private_scene_description' not in json.dumps(records)


def test_validator_restored_after_context_failure_and_nested_tracing_is_rejected():
    original = base.validate_observations
    with pytest.raises(RuntimeError, match='caller failure'):
        with diagnostics.trace_validation():
            with pytest.raises(RuntimeError, match='already active'):
                with diagnostics.trace_validation():
                    pass
            raise RuntimeError('caller failure')
    assert base.validate_observations is original
    with diagnostics.trace_validation() as records:
        base.validate_observations(valid(), window())
    assert records[0]['code'] == 'valid'
