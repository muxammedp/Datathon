import copy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from deploy.ensemble import qwen_batched as batched
from deploy.ensemble import qwen_expert as base


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def media(tmp_path):
    paths = []
    for number in range(3):
        path = tmp_path / f'{number}.png'
        Image.new('RGB', (96, 64), (number * 60, 20, 40)).save(path)
        paths.append(path.name)
    tracks = [dict(track_id=7, role='cashier_candidate', role_source='configured_zone',
                   samples=[dict(time_sec=1., bbox=[.3, .3, .6, .6])]),
              dict(track_id=8, role='unknown', role_source='unknown',
                   samples=[dict(time_sec=1., bbox=[.8, .1, .9, .9])])]
    window = dict(index=0, start_sec=0., end_sec=3., frame_paths=paths, times=[0., 1., 2.],
                  tracks=tracks, counter_crop=[.25, .25, .75, .75])
    return tmp_path, dict(video_sha256='a' * 64, windows=[window])


def facts(identities=(7,), kind='cash_counting_visible', timestamp=1.):
    return dict(actors=[dict(track_id=i, visibility='visible', activity='seated') for i in identities],
                service_state='active', cash_action='counting_visible',
                events=[dict(kind=kind, time_sec=timestamp, track_ids=list(identities))])


def stub_batch(instance, monkeypatch):
    calls = []
    monkeypatch.setattr(instance, '_load', lambda: None)

    def scores(images, windows, prompts):
        calls.append(('risk', [pictures[0].size for pictures in images], copy.deepcopy(windows), prompts))
        return [.5, -.25], dict(input_tokens=[20, 25], padded_input_tokens=25, model_calls=1)

    def generated(images, windows, prompts):
        calls.append(('facts', [pictures[0].size for pictures in images], copy.deepcopy(windows), prompts))
        return [json.dumps(facts([t['track_id'] for t in w['tracks']], timestamp=w['times'][1])) for w in windows], dict(
            model_calls=1, generated_tokens=[30, 20])

    monkeypatch.setattr(instance, '_risk_scores', scores)
    monkeypatch.setattr(instance, '_generate_facts', generated)
    return calls


def test_batches_same_frames_and_distinct_specialists_without_losing_crop_or_tracking(media, monkeypatch):
    root, index = media
    instance = batched.BatchedQwenExpert(root)
    calls = stub_batch(instance, monkeypatch)
    result = instance.analyze(index, root)
    row = result['windows'][0]
    assert result['version'] == batched.VERSION
    assert result['settings']['view_batch_size'] == 2
    assert [call[0] for call in calls] == ['risk', 'facts']
    assert all(call[1] == [(96, 64), (48, 32)] for call in calls)
    assert [t['track_id'] for t in calls[0][2][0]['tracks']] == [7, 8]
    assert [t['track_id'] for t in calls[0][2][1]['tracks']] == [7]
    assert calls[0][2][1]['tracks'][0]['samples'][0]['bbox'] == pytest.approx([.1, .1, .7, .7])
    window = base._window(index['windows'][0], root, 16)
    for kind, _, windows, prompts in calls:
        for view, actual, prompt in zip(batched.VIEWS, windows, prompts):
            assert prompt == base.build_prompt(base._view_window(window, view), view, [], kind)
            assert actual['times'] == [0., 1., 2.]
    assert row['model_passes'] == 2 and row['score'] == .125
    assert row['parse_status'] == dict(context='valid', cash='valid')
    assert row['view_disagreement'] and row['review_required']
    assert row['evidence_frames'][0]['frame_path'] == '1.png'
    assert row['evidence_frames'][1]['crop'] == [.25, .25, .75, .75]
    assert row['profile']['risk']['model_calls'] == row['profile']['facts']['model_calls'] == 1


def test_invalid_cash_schema_does_not_invalidate_valid_context_or_invent_normal_verdict(media, monkeypatch):
    root, index = media
    instance = batched.BatchedQwenExpert(root)
    stub_batch(instance, monkeypatch)
    monkeypatch.setattr(instance, '_generate_facts', lambda *args: (
        [json.dumps(facts(kind='unclassified_activity')), json.dumps(facts([999]))], dict(model_calls=1)))
    row = instance.analyze(index, root)['windows'][0]
    assert row['status'] == 'done'
    assert row['parse_status'] == dict(context='valid', cash='invalid_json_or_schema')
    assert row['observations']['cash'] == base.unknown_observations()
    assert 'invalid_or_incomplete_observations' in row['review_reasons']
    assert 'unclassified_activity' in row['review_reasons']
    assert row['context_score'] == .5 and row['cash_score'] == -.25
    assert 'prediction' not in row


def test_no_crop_and_no_tracks_stays_unknown(media, monkeypatch):
    root, index = media
    index['windows'][0].update(counter_crop=None, tracks=[])
    instance = batched.BatchedQwenExpert(root)
    calls = stub_batch(instance, monkeypatch)
    row = instance.analyze(index, root)['windows'][0]
    assert all(call[1] == [(96, 64), (96, 64)] for call in calls)
    assert row['cash_view'] == 'full_scene' and 'cashier_role_unknown' in row['review_reasons']


def test_resume_versions_callbacks_and_memory_preserved_without_global_mutation(media, monkeypatch):
    root, index = media
    second = copy.deepcopy(index['windows'][0])
    second.update(index=1, start_sec=3., end_sec=6., times=[3., 4., 5.])
    for track in second['tracks']:
        track['samples'][0]['time_sec'] = 4.
    index['windows'].append(second)
    instance = batched.BatchedQwenExpert(root)
    calls = stub_batch(instance, monkeypatch)
    callbacks = []
    result = instance.analyze(index, root, on_window=lambda p: callbacks.append(copy.deepcopy(p)))
    assert len(callbacks) == 2 and all(p['version'] == batched.VERSION for p in callbacks)
    assert len(result['windows'][1]['memory']['recent_events']) == 2
    assert base.VERSION == 'qwen_specialists_v1'
    calls.clear()
    resumed = instance.analyze(index, root, previous_results=callbacks[0])
    assert len(calls) == 2
    assert resumed['windows'][1]['memory'] == result['windows'][1]['memory']
    legacy = dict(callbacks[0], version=base.VERSION)
    with pytest.raises(ValueError, match='different execution'):
        instance.analyze(index, root, previous_results=legacy)
    assert callbacks[0]['version'] == batched.VERSION


def test_oom_reports_incomplete_window_then_raises_without_serial_fallback(media, monkeypatch):
    root, index = media
    instance = batched.BatchedQwenExpert(root)
    stub_batch(instance, monkeypatch)
    saved = []

    def oom(*args):
        raise torch.cuda.OutOfMemoryError('synthetic out of memory')

    monkeypatch.setattr(instance, '_generate_facts', oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        instance.analyze(index, root, on_window=lambda p: saved.append(copy.deepcopy(p)))
    assert saved[0]['version'] == batched.VERSION
    assert saved[0]['windows'][0]['score'] is None
    assert saved[0]['windows'][0]['status'] == 'failed'


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_tiny_qwen_batch_matches_serial_with_left_padding_and_image_association(tmp_path, device):
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    processor_path = ROOT / 'models/Qwen3-VL-2B-Instruct'
    if not (processor_path / 'tokenizer.json').is_file():
        pytest.skip('Local Qwen processor is not installed; tests do not download models')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('No local CUDA device')
    processor = AutoProcessor.from_pretrained(str(processor_path), local_files_only=True, trust_remote_code=False)
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096,
                         rope_scaling=dict(rope_type='default', mrope_section=[2, 1, 1])),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           num_position_embeddings=16, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, out_hidden_size=32, deepstack_visual_indexes=[0, 1]),
        tie_word_embeddings=False)
    torch.manual_seed(231)
    model = Qwen3VLForConditionalGeneration(config).to(
        device=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32).eval()
    model.requires_grad_(False)
    instance = batched.BatchedQwenExpert(tmp_path, device=device, max_frames=2)
    instance.model, instance.processor = model, processor
    instance.answer_ids = [processor.tokenizer.encode(value, add_special_tokens=False)[0] for value in ('0', '1')]
    images = [[Image.new('RGB', (96, 64), (180, 0, 0)), Image.new('RGB', (64, 64), (0, 180, 0))],
              [Image.new('RGB', (32, 64), (0, 0, 180))]]
    windows = [dict(times=[0., 1.]), dict(times=[3.])]
    prompts = ['Answer exactly 0 or 1. Does the second frame have green pixels?', 'Answer 0 or 1.']
    previous_padding = processor.tokenizer.padding_side
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        queries, inputs = instance._batch_inputs(images, windows, prompts)
        assert processor.tokenizer.padding_side == previous_padding
        assert inputs['attention_mask'][1, 0] == 0 and inputs['attention_mask'][:, -1].all()
        assert inputs['image_grid_thw'].shape[0] == 3
        pixel_offset, image_offset = 0, 0
        for number, (pictures, window, prompt) in enumerate(zip(images, windows, prompts)):
            query, single = instance._inputs(pictures, window['times'], prompt)
            assert queries[number] == query
            assert torch.equal(inputs['input_ids'][number][inputs['attention_mask'][number].bool()], single['input_ids'][0])
            count = single['pixel_values'].shape[0]
            assert torch.equal(inputs['pixel_values'][pixel_offset:pixel_offset+count], single['pixel_values'])
            assert torch.equal(inputs['image_grid_thw'][image_offset:image_offset+len(pictures)], single['image_grid_thw'])
            pixel_offset += count
            image_offset += len(pictures)
        scores, profile = instance._risk_scores(images, windows, prompts)
        serial = [instance._risk_score(pictures, window, prompt) for pictures, window, prompt in zip(images, windows, prompts)]
        assert scores == pytest.approx(serial, abs=4e-3 if device == 'cuda' else 2e-6)
        assert profile['model_calls'] == 1 and profile['input_tokens'][0] > profile['input_tokens'][1]
        instance.max_new_tokens = 3
        texts, generation = instance._generate_facts(images, windows, prompts)
        expected = []
        for pictures, window, prompt in zip(images, windows, prompts):
            _, single = instance._inputs(pictures, window['times'], prompt)
            with torch.inference_mode():
                output = model.generate(**single, do_sample=False, max_new_tokens=3, use_cache=True)
            expected.append(processor.tokenizer.decode(output[0, single['input_ids'].shape[1]:], skip_special_tokens=True))
        assert texts == expected
        assert generation['generated_tokens'] == [3, 3]
    finally:
        torch.set_num_threads(old_threads)
        for pictures in images:
            for picture in pictures:
                picture.close()
