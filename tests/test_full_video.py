from app.full_video import quality_gate
from app.full_video import frame_shape
import pytest


@pytest.mark.parametrize('score', [-100., 100.])
def test_unvalidated_full_video_model_abstains_for_both_predictions(monkeypatch, score):
    import contextlib
    import sys
    import types
    import app.full_video as full
    import app.engine as engine
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(inference_mode=contextlib.nullcontext))
    analyzer = full.FullVideoAnalyzer()
    analyzer.metadata = {'threshold': 0}
    monkeypatch.setattr(analyzer, 'load', lambda: None)
    monkeypatch.setattr(engine, 'video_info', lambda path: {'duration_sec': 20})
    monkeypatch.setattr(full, 'prepare_video', lambda *args: ({}, {
        'sampled_frames': 40, 'context_tokens': 1000, 'max_gap_sec': .51}))
    monkeypatch.setattr(full, 'binary_logit', lambda *args: score)
    result = analyzer.analyze('synthetic.mov')
    assert result['verdict'] == 'inconclusive'
    assert result['violations'] == []
    assert result['review_required'] is True
    assert result['analysis']['automatic_verdict_enabled'] is False
    assert result['analysis']['whole_video_context'] is True


def test_full_video_validation_excludes_duplicates_and_calibration_from_test():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'data/full-video-lora-4b/metrics.json'
    if not path.exists():
        pytest.skip('Completed research results are not present')
    report = json.loads(path.read_text(encoding='utf8'))
    held_out = []
    for fold in report['folds']:
        train, calibration, test = (set(fold[k]) for k in ('train_sha256', 'calibration_sha256', 'test_sha256'))
        assert not train & calibration
        assert not train & test
        assert not calibration & test
        assert len(train) + len(calibration) + len(test) == report['unique_videos']
        held_out.extend(test)
    assert len(held_out) == len(set(held_out)) == report['unique_videos']


def test_unseen_aspect_ratios_obey_visual_memory_budget():
    for height, width in [(320, 320), (320, 175), (180, 320), (32, 320)]:
        h, w = frame_shape(height, width)
        assert h * w <= 320 * 192
        assert h % 32 == w % 32 == 0
        assert min(h, w) >= 32


def test_high_accuracy_from_silent_detector_cannot_pass_quality_gate():
    assert not quality_gate({'unique_videos': 40, 'folds': [None] * 5,
                             'metrics': {'accuracy': .9, 'precision': .99, 'recall': .01, 'false_positive_rate': 0}})


def test_high_recall_with_false_alarms_cannot_pass_quality_gate():
    assert not quality_gate({'unique_videos': 40, 'folds': [None] * 5,
                             'metrics': {'accuracy': .9, 'precision': .9, 'recall': .99, 'false_positive_rate': .3}})


def test_missing_partial_or_small_evaluation_cannot_enable_verdicts():
    assert not quality_gate({})
    assert not quality_gate({'unique_videos': 8, 'folds': [None],
                             'metrics': {'accuracy': 1, 'precision': 1, 'recall': 1, 'false_positive_rate': 0}})
