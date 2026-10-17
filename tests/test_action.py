import json

import numpy as np
import pytest

from app.action_model import ActionAnalyzer, clip_indices, window_centers


@pytest.mark.parametrize('score', [0., 1.])
def test_failed_model_cannot_make_an_accusation_or_clear_verdict(monkeypatch, score):
    import app.action_model as action
    import app.engine as engine
    analyzer = ActionAnalyzer()
    analyzer.model = object()
    analyzer.metadata = {'threshold': .5}
    monkeypatch.setattr(analyzer, 'load', lambda: None)
    monkeypatch.setattr(action, 'decode_video', lambda p: (
        np.zeros((32, 112, 112, 3), dtype=np.uint8), np.arange(32) / 4, {'duration_sec': 8.}))
    monkeypatch.setattr(action, 'predict_windows', lambda *args: (np.array([2., 4., 6.]), np.full(3, score)))
    monkeypatch.setattr(engine, 'evidence', lambda *args: [{'timestamp_sec': 2., 'image_base64': 'synthetic'}])
    result = analyzer.analyze('synthetic.mov')
    assert result['verdict'] == 'inconclusive'
    assert result['violations'] == []
    assert result['review_required'] is True
    assert result['analysis']['rules_checked'] == 0
    assert result['analysis']['automatic_verdict_enabled'] is False
    for segment in result['review_segments']:
        assert segment['rule'] is None
        assert segment['rule_id'] is None


def test_clip_boundaries_do_not_wrap_to_unrelated_end():
    assert np.all(clip_indices(10, -2) == 0)
    assert np.all(clip_indices(10, 30) == 9)
    assert len(clip_indices(10, 1)) == 16
    centers = window_centers(np.arange(2) / 4)
    assert len(centers) == 1 and 0 <= centers[0] <= .25


def test_training_folds_do_not_share_videos_or_duplicates():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'data' / 'action-training' / 'metrics.json'
    if not path.exists():
        pytest.skip('Training results are not included in source-only installations')
    report = json.loads(path.read_text(encoding='utf8'))
    test_hashes = []
    for fold in report['folds']:
        train, calibration, test = (set(fold[k]) for k in ('train_sha256', 'calibration_sha256', 'test_sha256'))
        assert not train & calibration
        assert not train & test
        assert not calibration & test
        test_hashes.extend(test)
    assert len(test_hashes) == len(set(test_hashes)) == report['unique_videos']
