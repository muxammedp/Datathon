import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / 'deploy/ensemble/action_expert.py'
SPEC = importlib.util.spec_from_file_location('action_expert', SOURCE)
action = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(action)


@pytest.mark.parametrize('count', [1, 2, 7, 8, 9, 12, 15, 16, 17, 24, 25, 30, 32, 63, 64, 65, 129])
def test_overlapping_clips_cover_every_sample_with_only_tail_duplicate_padding(count):
    clips = action.clip_indices(count)
    assert sorted({value for clip in clips for value in clip}) == list(range(count))
    assert all(len(clip) == 8 and clip == sorted(clip) for clip in clips)
    assert all(len(set(clip)) == 8 for clip in clips[:-1])
    assert all(b[0] - a[0] == 4 for a, b in zip(clips, clips[1:]))
    last = clips[-1]
    for a, b in zip(last, last[1:]):
        assert b == a + 1 or a == b == count - 1


def test_letterbox_preserves_scene_aspect_and_crops_only_when_explicit():
    image = Image.new('RGB', (400, 200), (255, 0, 0))
    value = np.asarray(action.letterbox(image, fill=(0, 0, 0)))
    assert value.shape == (224, 224, 3)
    assert np.all(value[:56] == 0) and np.all(value[168:] == 0)
    assert np.all(value[56:168, :, 0] == 255)
    assert np.all(value[56:168, :, 1:] == 0)
    cropped = np.asarray(action.letterbox(image, crop=[.25, 0, .75, 1], fill=(0, 0, 0)))
    assert np.all(cropped[:, :, 0] == 255)
    portrait = np.asarray(action.letterbox(Image.new('RGB', (200, 400), (0, 255, 0)), fill=(0, 0, 0)))
    assert np.all(portrait[:, :56] == 0) and np.all(portrait[:, 168:] == 0)


@pytest.mark.parametrize('crop', [[0, 0, 0, 1], [-.1, 0, 1, 1], [0, 0, 1.1, 1],
                                  [0, .5, 1, .4], [0, 0, float('nan'), 1],
                                  [False, 0, 1, 1], '0,0,1,1', [0, 0, 1]])
def test_crop_validation_rejects_ambiguous_or_invalid_regions(crop):
    with pytest.raises(ValueError):
        action.validate_crop(crop)


def fake_expert():
    expert = object.__new__(action.ActionExpert)
    expert.device = torch.device('cpu')
    expert.model = object()
    expert.mean = np.zeros(3, dtype=np.float32)
    expert.std = np.ones(3, dtype=np.float32)
    expert.rescale = 1 / 255
    expert.fill = (0, 0, 0)
    return expert


def sample_index(tmp_path, count=25, crop=None):
    names = []
    for number in range(count):
        path = tmp_path / f'frame-{number}.png'
        Image.new('RGB', (20, 10), (number, number, number)).save(path)
        names.append(path.name)
    return dict(video_sha256='a' * 64, windows=[dict(index=0, start_sec=0., end_sec=count / 4,
                frame_paths=names, times=[number / 4 for number in range(count)], counter_crop=crop)])


def test_extract_is_chronological_all_windows_full_coverage_and_no_crop_reuses_scene(tmp_path):
    expert = fake_expert()
    calls = []

    def encode(tensors):
        values = [round(float(tensor[0, 112, 112]) * 255) for tensor in tensors]
        calls.append(values)
        return np.full(512, np.mean(values), dtype=np.float32)

    expert._encode = encode
    index = sample_index(tmp_path)
    second = dict(index['windows'][0], index=1)
    index['windows'].append(second)
    result = expert.extract(index, tmp_path)
    assert result['all_windows_processed'] and len(result['windows']) == 2
    expected = action.clip_indices(25)
    assert calls == expected + expected
    for window in result['windows']:
        features = np.asarray(window['features'])
        assert features.shape == (2048,) and np.isfinite(features).all()
        assert np.array_equal(features[:1024], features[1024:])
        assert window['covered_sample_indices'] == list(range(25))
        assert window['selected_times'] == index['windows'][0]['times']
        assert window['clips_processed'] == window['encoder_passes'] == 6
        assert window['clips'][-1]['padding_frames'] == 3
        assert not window['crop_used']


def test_explicit_counter_crop_is_separate_view_and_keeps_fixed_feature_size(tmp_path):
    expert = fake_expert()
    calls = []

    def encode(tensors):
        value = float(tensors[0][0].mean())
        calls.append(value)
        return np.full(512, value, dtype=np.float32)

    expert._encode = encode
    index = sample_index(tmp_path, 2, crop=[.5, 0, 1, 1])
    image = Image.new('RGB', (20, 10), (0, 0, 0))
    image.paste((255, 0, 0), (10, 0, 20, 10))
    for name in index['windows'][0]['frame_paths']:
        image.save(tmp_path / name)
    result = expert.extract(index, tmp_path)['windows'][0]
    assert result['crop_used'] and result['encoder_passes'] == 2
    assert result['clips_processed'] == 1 and len(result['features']) == 2048
    assert calls[0] < calls[1]
    assert result['clips'][0]['padding_frames'] == 6


@pytest.mark.parametrize('change', ['times_length', 'order', 'nan', 'outside_time', 'crop', 'escape',
                                   'absolute', 'missing', 'identity', 'window_index', 'empty'])
def test_extract_rejects_invalid_media_index_before_inference(tmp_path, change):
    expert = fake_expert()
    expert._encode = lambda tensors: pytest.fail('Inference should not run for an invalid index')
    index = sample_index(tmp_path, 2)
    window = index['windows'][0]
    if change == 'times_length':
        window['times'].pop()
    elif change == 'order':
        window['times'].reverse()
    elif change == 'nan':
        window['times'][0] = float('nan')
    elif change == 'outside_time':
        window['times'][0] = -1.
    elif change == 'crop':
        window['counter_crop'] = [0, 0, 0, 0]
    elif change == 'escape':
        window['frame_paths'][0] = '../outside.png'
    elif change == 'absolute':
        window['frame_paths'][0] = str(tmp_path / 'frame-0.png')
    elif change == 'missing':
        window['frame_paths'][0] = 'missing.png'
    elif change == 'identity':
        index['video_sha256'] = 'invalid'
    elif change == 'window_index':
        window['index'] = 3
    elif change == 'empty':
        window['frame_paths'], window['times'] = [], []
    with pytest.raises(ValueError):
        expert.extract(index, tmp_path)


def test_native_local_loader_is_frozen_eval_and_matches_real_xclip_cpu(tmp_path, monkeypatch):
    from transformers import VideoMAEImageProcessor, XCLIPConfig, XCLIPModel
    config = XCLIPConfig(
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=16),
        vision_config=dict(hidden_size=768, num_hidden_layers=1, num_attention_heads=12,
                           intermediate_size=64, image_size=224, num_frames=8, patch_size=32,
                           mit_hidden_size=512, mit_intermediate_size=64, mit_num_hidden_layers=1,
                           mit_num_attention_heads=8), projection_dim=512)
    model = XCLIPModel(config)
    processor = VideoMAEImageProcessor(image_mean=[.48145466, .4578275, .40821073],
                                      image_std=[.26862954, .26130258, .27577711])
    model_calls, processor_calls = [], []

    def load_model(path, **kwargs):
        model_calls.append((path, kwargs))
        return model

    def load_processor(path, **kwargs):
        processor_calls.append((path, kwargs))
        return processor

    monkeypatch.setattr(XCLIPModel, 'from_pretrained', load_model)
    monkeypatch.setattr(VideoMAEImageProcessor, 'from_pretrained', load_processor)
    expert = action.ActionExpert(tmp_path, device='cpu')
    assert model_calls[0][1] == dict(local_files_only=True, trust_remote_code=False)
    assert processor_calls[0][1] == dict(local_files_only=True)
    np.testing.assert_allclose(expert.mean, processor.image_mean, rtol=1e-6)
    np.testing.assert_allclose(expert.std, processor.image_std, rtol=1e-6)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    tensor = torch.linspace(-1., 1., 3 * 224 * 224).reshape(3, 224, 224)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        actual = expert._encode([tensor] * 8)
        with torch.inference_mode():
            reference = model.get_video_features(pixel_values=torch.stack([tensor] * 8)[None]).float()[0].numpy()
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
    finally:
        torch.set_num_threads(old_threads)
    expert.release()
    assert expert.model is None
    with pytest.raises(RuntimeError, match='released'):
        expert.extract({}, tmp_path)


def test_encoder_rejects_nonfinite_features():
    expert = fake_expert()

    class Nonfinite:
        def get_video_features(self, **kwargs):
            return torch.full((1, 512), float('nan'))

    expert.model = Nonfinite()
    with pytest.raises(ValueError, match='nonfinite'):
        expert._encode([torch.zeros(3, 224, 224)] * 8)
