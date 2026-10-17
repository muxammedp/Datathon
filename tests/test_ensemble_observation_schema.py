import copy
import json

import pytest

from deploy.ensemble import observation_schema as schema
from deploy.ensemble import qwen_expert as base


def window():
    return dict(start_sec=0., end_sec=64., times=list(map(float, range(64))),
                tracks=[dict(track_id=7)])


def event(index=0):
    return dict(kind='cash_counting_visible', time_sec=float(index), track_ids=[7])


def observations(events=None):
    return dict(actors=[dict(track_id=7, visibility='visible', activity='handling_cash')],
                service_state='active', cash_action='counting_visible',
                events=[event()] if events is None else events)


@pytest.mark.parametrize('count', [0, 1, 8, 9, 64])
def test_every_event_is_preserved_in_order_through_64_without_changing_old_schema(count):
    value = observations([event(number) for number in range(count)])
    untouched = copy.deepcopy(value)
    assert schema.validate_observations(json.dumps(value), window()) == value
    assert schema.validate_observations(value, window()) == value
    assert value == untouched
    assert base.MAX_EVENTS == 8 and schema.MAX_EVENTS == 64
    if count > 8:
        with pytest.raises(ValueError, match='Too many observation events'):
            base.validate_observations(value, window())


def test_repeated_events_are_not_silently_deduplicated():
    value = observations([event()] * 9)
    result = schema.validate_observations(value, window())
    assert len(result['events']) == 9 and result == value


@pytest.mark.parametrize('change', [dict(track_ids=[999]), dict(time_sec=8.5), dict(time_sec=64.),
                                   dict(kind='invented_violation'), dict(track_ids=[7, 7])])
def test_invalid_ninth_event_is_rejected_instead_of_being_truncated(change):
    events = [event(number) for number in range(9)]
    events[-1].update(change)
    with pytest.raises(ValueError):
        schema.validate_observations(observations(events), window())


def test_more_than_64_events_fails_even_when_each_event_is_valid():
    with pytest.raises(ValueError, match='count exceeds the limit of 64'):
        schema.validate_observations(observations([event()] * 65), window())


def test_single_complete_event_object_becomes_one_validated_event_without_mutating_input():
    value = observations(event(12))
    untouched = copy.deepcopy(value)
    result = schema.validate_observations(json.dumps(value), window())
    assert result == dict(value, events=[event(12)])
    assert value == untouched


@pytest.mark.parametrize('events', [None, 'cash_counting_visible', 3, True, (),
                                    {'0': event()}, {'events': [event()]}, {},
                                    dict(event(), category='invented_violation'),
                                    {'kind': 'cash_counting_visible', 'time_sec': 1.}])
def test_event_list_normalization_never_accepts_arbitrary_containers_or_partial_objects(events):
    value = observations()
    value['events'] = events
    with pytest.raises((ValueError, TypeError)):
        schema.validate_observations(value, window())


@pytest.mark.parametrize('change', [dict(track_ids=[999]), dict(time_sec=1.5),
                                   dict(kind='invented_violation'), dict(track_ids=None)])
def test_singleton_event_is_subject_to_all_event_checks(change):
    value = event()
    value.update(change)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_observations(observations(value), window())


@pytest.mark.parametrize('change', [dict(extra='not_allowed'), dict(service_state='invented_state'),
                                   dict(cash_action='invented_action'),
                                   dict(actors=[dict(track_id=99, visibility='visible', activity='seated')]),
                                   dict(actors=[dict(track_id=7, visibility='visible', activity='seated', role='cashier')])])
def test_root_actor_and_enum_validation_remains_strict(change):
    value = observations([event(i) for i in range(9)])
    value.update(change)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_observations(value, window())


def test_json_duplicates_fences_and_size_limit_are_still_rejected():
    value = json.dumps(observations())
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        schema.validate_observations(value[:-1] + ',"events":[]}', window())
    with pytest.raises(json.JSONDecodeError):
        schema.validate_observations('```json\n' + value + '\n```', window())
    with pytest.raises(ValueError, match='exceeds the bounded schema'):
        schema.validate_observations(value + ' ' * 16000, window())
