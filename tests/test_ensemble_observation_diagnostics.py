import json
from types import SimpleNamespace

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


@pytest.mark.parametrize('events,kind,count', [
    ([], 'list', 0),
    ([{'kind': 'private_scene_description'}] * 9, 'list', 9),
    ({'private_scene_description': 'private_scene_description'}, 'object', None),
    ('private_scene_description', 'string', None),
    (None, 'null', None),
    (False, 'boolean', None),
    (123., 'number', None),
])
def test_event_structural_metadata_never_exports_values(events, kind, count):
    value = dict(valid(), events=events)
    with diagnostics.trace_validation() as records:
        try:
            base.validate_observations(json.dumps(value), window())
        except (ValueError, TypeError, KeyError):
            pass
    assert records[0]['events_type'] == kind and records[0]['event_list_count'] == count
    assert 'private_scene_description' not in json.dumps(records)


def test_schema_v2_observes_all_events_and_restores_only_selected_validator():
    from deploy.ensemble import observation_schema
    event = dict(kind='cash_counting_visible', time_sec=0., track_ids=[])
    value = dict(valid(), events=[dict(event) for _ in range(12)])
    original, original_base = observation_schema.validate_observations, base.validate_observations
    with diagnostics.trace_validation(validator_module=observation_schema) as records:
        result = observation_schema.validate_observations(json.dumps(value), window())
        assert base.validate_observations is original_base
        with pytest.raises(RuntimeError, match='already active'):
            with diagnostics.trace_validation():
                pass
    assert len(records) == 1 and result == value
    assert records[0]['strict_valid'] and records[0]['accepted_event_count'] == 12
    assert records[0]['events_type'] == 'list' and records[0]['event_list_count'] == 12
    assert observation_schema.validate_observations is original
    assert base.validate_observations is original_base


def test_schema_v2_single_event_object_is_reported_without_recording_event_content():
    from deploy.ensemble import observation_schema
    event = dict(kind='cash_counting_visible', time_sec=0., track_ids=[])
    value = dict(valid(), events=event)
    with diagnostics.trace_validation(validator_module=observation_schema) as records:
        result = observation_schema.validate_observations(value, window())
    assert result['events'] == [event] and value['events'] == event
    assert records[0]['events_type'] == 'object' and records[0]['event_list_count'] is None
    assert records[0]['accepted_event_count'] == 1 and records[0]['code'] == 'valid'
    assert 'cash_counting_visible' not in json.dumps(records)


@pytest.mark.parametrize('events,code', [
    ('private_scene_description', 'invalid_event_list_type'),
    ([dict(kind='cash_counting_visible', time_sec=0., track_ids=[])] * 65, 'event_list_over_budget'),
])
def test_schema_v2_distinguishes_wrong_event_type_from_over_budget(events, code):
    from deploy.ensemble import observation_schema
    original = observation_schema.validate_observations
    with diagnostics.trace_validation(validator_module=observation_schema) as records:
        with pytest.raises(ValueError):
            observation_schema.validate_observations(dict(valid(), events=events), window())
    assert records[0]['code'] == code and not records[0]['strict_valid']
    assert observation_schema.validate_observations is original
    assert 'private_scene_description' not in json.dumps(records)


def test_invalid_validator_does_not_leave_global_trace_locked():
    with pytest.raises(TypeError, match='callable validate_observations'):
        with diagnostics.trace_validation(validator_module=SimpleNamespace()):
            pass
    with diagnostics.trace_validation() as records:
        base.validate_observations(valid(), window())
    assert records[0]['strict_valid']


@pytest.mark.parametrize('require_valid,statuses,exit_code', [
    (True, {'context': 'valid', 'cash': 'valid'}, None),
    (True, {'context': 'valid', 'cash': 'invalid_json_or_schema'}, 2),
    (True, {'context': 'valid'}, 2),
    (False, {'context': 'invalid_json_or_schema', 'cash': 'invalid_json_or_schema'}, None),
])
def test_diagnostic_cli_require_valid_only_gates_after_report_is_generated(monkeypatch, require_valid, statuses, exit_code):
    from deploy.ensemble import diagnose_qwen
    calls = []

    def report(args):
        calls.append(args)
        return dict(parse_status=statuses, valid_observation_views=sum(value == 'valid' for value in statuses.values()))

    monkeypatch.setattr(diagnose_qwen, 'diagnose', report)
    monkeypatch.setattr(diagnose_qwen, 'event', lambda *args, **kwargs: None)
    argv = ['diagnose_qwen.py', '--cache', 'cache', '--output', 'report.json']
    if require_valid:
        argv.append('--require-valid')
    monkeypatch.setattr('sys.argv', argv)
    if exit_code is None:
        diagnose_qwen.main()
    else:
        with pytest.raises(SystemExit) as error:
            diagnose_qwen.main()
        assert error.value.code == exit_code
    assert len(calls) == 1 and calls[0].require_valid is require_valid
