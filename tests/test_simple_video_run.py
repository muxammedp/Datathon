import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.simple_video import run


def test_uncertain_or_failed_clip_is_never_a_negative_verdict():
    normal = dict(status='done', prediction='no_violation')
    uncertain = dict(status='done', prediction='insufficient_data')
    failed = dict(status='failed', prediction='no_violation')
    positive = dict(status='done', prediction='violation')
    assert run.combine([normal, uncertain]) == ('insufficient_data', None)
    assert run.combine([normal, failed]) == ('insufficient_data', None)
    assert run.combine([normal, positive, uncertain]) == ('violation', 1)
    assert run.combine([normal, normal]) == ('no_violation', None)
    assert run.combine([]) == ('insufficient_data', None)


def test_correct_video_label_does_not_prove_exact_event_localization():
    result = run.comparison('violation', 1, 8., [22.])
    assert result['label_matches']
    assert result['nearest_annotation_error_sec'] == 14.
    assert result['exact_event_verified'] is False
    assert run.comparison('insufficient_data', 1, None, [22.])['label_matches'] is None


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., True])
def test_invalid_annotation_time_is_rejected(bad):
    with pytest.raises(ValueError):
        run.comparison('no_violation', 0, None, [bad])


def test_one_video_run_keeps_labels_out_of_model_and_scene_text_out_of_console(tmp_path, monkeypatch, capsys):
    video = tmp_path / 'private.mov'
    video.write_bytes(b'placeholder')
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    clip = dict(rgb=np.zeros((2, 32, 32, 3), dtype=np.uint8), source_indices=[0, 10])
    metadata = dict(source_frames=12)
    calls = []

    class Analyzer:
        def __init__(self, path, **kwargs):
            assert path == model

        def load(self):
            calls.append('load')

        def analyze_clip(self, actual_clip, actual_metadata):
            assert actual_clip is clip and actual_metadata is metadata
            calls.append('infer')
            return dict(status='done', prediction='violation', evidence='PRIVATE SCENE DESCRIPTION',
                        source_frame_index=10, source_time_sec=1., profile={'model_passes': 1})

    monkeypatch.setattr(run, 'SimpleVideoAnalyzer', Analyzer)
    monkeypatch.setattr(run, 'prepare_video', lambda path, settings: ([clip], metadata))
    args = SimpleNamespace(video=video, dataset=None, expected_label=0, expected_time=[],
                           output=tmp_path / 'results', model=model, sample_fps=1.5,
                           max_new_tokens=96, device='cpu', limit=1)
    before = video.read_bytes()
    summary = run.run(args)
    assert calls == ['load', 'infer']
    assert summary['wrong'] == 1 and summary['correct'] == 0
    assert 'PRIVATE SCENE DESCRIPTION' not in capsys.readouterr().out
    assert video.read_bytes() == before
    result = json.loads(next(args.output.glob('*/result.json')).read_text())
    assert result['comparison']['annotation_used_for_inference'] is False
    assert next(args.output.glob('*/candidate-frame.jpg')).is_file()
    with pytest.raises(ValueError, match='new output directory'):
        run.run(args)


def test_dataset_checksum_failure_prevents_inference(tmp_path):
    video = tmp_path / 'clip.mov'
    video.write_bytes(b'modified')
    manifest = {'videos': [{'file': video.name, 'label': 1, 'sha256': 'a' * 64}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    args = SimpleNamespace(video=None, dataset=tmp_path, limit=1)
    with pytest.raises(ValueError, match='differs'):
        run.select_inputs(args)
