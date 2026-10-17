import json

from deploy.ensemble import qwen_expert as base


VERSION = 'observations_v2'
MAX_EVENTS = 64
EVENT_KEYS = {'kind', 'time_sec', 'track_ids'}


def validate_observations(raw, window):
    if isinstance(raw, str):
        if len(raw) > 16000:
            raise ValueError('Observation output exceeds the bounded schema')
        raw = json.loads(raw, object_pairs_hook=base._unique_object)
    if not isinstance(raw, dict) or set(raw) != base.FACT_KEYS:
        raise ValueError('Unexpected observation fields')
    validated = base.validate_observations(dict(raw, events=[]), window)
    events = raw['events']
    if isinstance(events, dict):
        if set(events) != EVENT_KEYS:
            raise ValueError('Unexpected event fields')
        events = [events]
    elif not isinstance(events, list):
        raise ValueError('Observation events must be a list or a complete event object')
    if len(events) > MAX_EVENTS:
        raise ValueError('Observation event count exceeds the limit of 64')
    for start in range(0, len(events), base.MAX_EVENTS):
        checked = base.validate_observations(dict(raw, events=events[start:start + base.MAX_EVENTS]), window)
        validated['events'].extend(checked['events'])
    return validated
