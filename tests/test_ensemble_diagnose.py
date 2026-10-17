from types import SimpleNamespace

import pytest

from deploy.ensemble import diagnose_qwen as diagnostic
from deploy.ensemble import qwen_expert as base
from deploy.ensemble import observation_schema as observations


def test_diagnostic_preserves_validator_and_cache_and_exports_no_observation_text(tmp_path, monkeypatch):
    cache, models = tmp_path / 'cache', tmp_path / 'models'
    cache.mkdir()
    sentinel = cache / 'untouched.json'
    sentinel.write_text('existing extraction')
    config = dict(fingerprint='a' * 64, settings=dict(device='cpu', qwen_frames=16, qwen_new_tokens=384))
    monkeypatch.setattr(diagnostic, 'load_window', lambda *args: (config, 'b' * 64, {}))
    monkeypatch.setattr(diagnostic, 'verify_models', lambda *args, **kwargs: None)
    released = []

    class Expert:
        def __init__(self, *args, **kwargs):
            pass

        def _load(self):
            pass

        def _analyze_window(self, *args):
            for _ in range(2):
                with pytest.raises(ValueError):
                    observations.validate_observations('private scene output', {})
            return dict(parse_status=dict(context='invalid_json_or_schema', cash='invalid_json_or_schema'),
                        elapsed_sec=1., profile=dict(generated_tokens=[20, 20]),
                        observations=dict(private='another secret'))

        def release(self):
            released.append(True)

    monkeypatch.setattr(diagnostic, 'BatchedQwenExpert', Expert)
    original = observations.validate_observations
    args = SimpleNamespace(cache=cache, models=models, output=tmp_path / 'report.json', video_sha=None)
    report = diagnostic.diagnose(args)
    assert observations.validate_observations is original and released == [True]
    assert report['validation_calls'] == 2
    assert [row['view'] for row in report['diagnostics']] == ['context', 'cash']
    assert sentinel.read_text() == 'existing extraction'
    assert not report['raw_responses_exported'] and not report['extraction_cache_modified']
    saved = args.output.read_text()
    assert 'private scene output' not in saved and 'another secret' not in saved
    with pytest.raises(ValueError, match='new report file'):
        diagnostic.diagnose(args)
