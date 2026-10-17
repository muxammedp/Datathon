import json
import re
from contextlib import contextmanager

from deploy.ensemble import qwen_expert as base


ERROR_CODES = {
    'Expected a finite number': 'nonfinite_or_invalid_number',
    'Invalid track ID': 'invalid_track_id',
    'Duplicate JSON key': 'duplicate_json_key',
    'Observation output exceeds the bounded schema': 'output_too_long',
    'Unexpected observation fields': 'unexpected_observation_fields',
    'Unknown observation enum': 'unknown_observation_enum',
    'Invalid actor observations': 'invalid_actor_observations',
    'Unexpected actor fields': 'unexpected_actor_fields',
    'Hallucinated actor identity or activity': 'unknown_actor_or_activity',
    'Too many observation events': 'invalid_event_list',
    'Observation events must be a list or a complete event object': 'invalid_event_list_type',
    'Observation event count exceeds the limit of 64': 'event_list_over_budget',
    'Unexpected event fields': 'unexpected_event_fields',
    'Event is outside the sampled evidence or closed vocabulary': 'invalid_event_time_kind_or_tracks',
    'Hallucinated event track ID': 'unknown_event_track_id',
    'A departure must refer to a tracked person': 'departure_without_track',
}
WHOLE_JSON_FENCE = re.compile(r'\A```json[ \t]*\r?\n([\s\S]*?)\r?\n```\Z')
_active = False


def _structure(raw):
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            return dict(events_type='unavailable', event_list_count=None)
    if not isinstance(value, dict):
        return dict(events_type='unavailable', event_list_count=None)
    if 'events' not in value:
        return dict(events_type='missing', event_list_count=None)
    events = value['events']
    if isinstance(events, list):
        return dict(events_type='list', event_list_count=len(events))
    kind = ('object' if isinstance(events, dict) else 'string' if isinstance(events, str)
            else 'null' if events is None else 'boolean' if isinstance(events, bool)
            else 'number' if isinstance(events, (int, float)) else 'other')
    return dict(events_type=kind, event_list_count=None)


def _error(exc):
    if isinstance(exc, json.JSONDecodeError):
        result = dict(code='invalid_json')
        for source, target in (('lineno', 'line'), ('colno', 'column')):
            number = getattr(exc, source, None)
            if type(number) is int and number > 0:
                result[target] = number
        return result
    if type(exc) is ValueError:
        message = exc.args[0] if len(exc.args) == 1 and type(exc.args[0]) is str else None
        return dict(code=ERROR_CODES.get(message, 'unclassified_validation_error'))
    if isinstance(exc, TypeError):
        return dict(code='invalid_value_type')
    if isinstance(exc, KeyError):
        return dict(code='missing_required_field')
    return dict(code='unclassified_validation_error')


@contextmanager
def trace_validation(validator_module=base):
    global _active
    if _active:
        raise RuntimeError('Validation tracing is already active in this process')
    original = getattr(validator_module, 'validate_observations', None)
    if not callable(original):
        raise TypeError('Diagnostic validator module must provide a callable validate_observations')
    _active = True
    records = []

    def traced(raw, window):
        text = raw.strip() if isinstance(raw, str) else None
        record = dict(call_index=len(records) + 1, view=('context', 'cash')[len(records) % 2],
                      input_kind='text' if text is not None else 'object' if isinstance(raw, dict) else 'other',
                      char_count=len(raw) if text is not None else None,
                      starts_with_json_fence=bool(text is not None and text.startswith('```json')),
                      ends_with_fence=bool(text is not None and text.endswith('```')),
                      fence_probe=dict(attempted=False), **_structure(raw))
        try:
            value = original(raw, window)
        except Exception as exc:
            record.update(strict_valid=False, **_error(exc))
            match = WHOLE_JSON_FENCE.fullmatch(text) if text is not None else None
            if match is not None and '```' not in match.group(1):
                probe = dict(attempted=True)
                try:
                    original(match.group(1), window)
                except Exception as unwrap_error:
                    probe.update(valid_after_unwrap=False, **_error(unwrap_error))
                else:
                    probe.update(valid_after_unwrap=True, code='valid')
                record['fence_probe'] = probe
            records.append(record)
            raise
        else:
            events = value.get('events') if isinstance(value, dict) else None
            record.update(strict_valid=True, code='valid',
                          accepted_event_count=len(events) if isinstance(events, list) else None)
            records.append(record)
            return value

    validator_module.validate_observations = traced
    try:
        yield records
    finally:
        validator_module.validate_observations = original
        _active = False
