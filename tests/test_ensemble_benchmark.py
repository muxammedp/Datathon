from types import SimpleNamespace

import pytest
from PIL import Image

from deploy.ensemble import benchmark_qwen as bench
from deploy.ensemble.common import file_hash, save_json
from deploy.ensemble.run import protocol


def fixture_cache(tmp_path):
    cache = tmp_path / 'cache'
    settings = dict(window_sec=16., overlap_sec=4., sample_fps=4., max_frames_per_window=64,
                    max_edge=848, device='cpu', qwen_frames=16, qwen_new_tokens=384)
    save_json(cache / 'protocol.json', protocol(settings, None))
    media = cache / 'videos' / ('a' * 64) / 'media'
    media.mkdir(parents=True)
    Image.new('RGB', (64, 64), 'blue').save(media / 'frame.png')
    window = dict(index=0, start_sec=0., end_sec=1., frame_paths=['frame.png'], times=[0.],
                  frame_sha256=[file_hash(media / 'frame.png')], tracks=[], counter_crop=None)
    index = dict(video_sha256='a' * 64, coverage_complete=True, windows=[window])
    save_json(media / 'index.json', index)
    save_json(media / 'complete.json', dict(index_sha256=file_hash(media / 'index.json')))
    return cache, media


def test_benchmark_rejects_modified_inputs_without_modifying_existing_cache(tmp_path):
    cache, media = fixture_cache(tmp_path)
    _, identity, window = bench.load_window(cache)
    assert identity == 'a' * 64 and window['times'] == [0.]
    with pytest.raises(ValueError, match='new report file'):
        bench.safe_output(media / 'report.json', cache, tmp_path / 'models')
    with pytest.raises(ValueError, match='new report file'):
        bench.safe_output(tmp_path / 'models' / 'report.json', cache, tmp_path / 'models')
    (media / 'frame.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='frame checksum'):
        bench.load_window(cache)


@pytest.mark.parametrize('schema_valid', [True, False])
def test_benchmark_shares_one_model_alternates_order_and_reports_no_observations(tmp_path, monkeypatch, schema_valid):
    cache, _ = fixture_cache(tmp_path)
    calls, loads, released, models = [], [], [], []

    class Serial:
        def __init__(self, *args, **kwargs):
            self.model = self.processor = self.answer_ids = None

        def _load(self):
            loads.append(True)
            self.model = SimpleNamespace(generate=lambda **kwargs: None)
            self.processor = object()
            self.answer_ids = [0, 1]

        def _inputs(self, *args):
            return '', {}

        def _analyze_window(self, window, memory):
            calls.append(type(self).__name__)
            models.append(self.model)
            assert memory == [] and len(window['paths']) == 1
            status = 'valid' if schema_valid else 'invalid_json_or_schema'
            return dict(sampled_frames=1, model_passes=4 if type(self).__name__ == 'Serial' else 2,
                        context_score=.3, cash_score=.1, parse_status=dict(context=status, cash=status),
                        observations=dict(private_test_text='not exported'))

        def release(self):
            released.append(True)

    class Batch(Serial):
        pass

    monkeypatch.setattr(bench, 'QwenExpert', Serial)
    monkeypatch.setattr(bench, 'BatchedQwenExpert', Batch)
    monkeypatch.setattr(bench, 'verify_models', lambda *args, **kwargs: None)
    args = SimpleNamespace(cache=cache, models=tmp_path / 'models', output=tmp_path / 'report.json',
                           video_sha=None, repeats=2)
    report = bench.benchmark(args)
    assert calls == ['Serial', 'Batch', 'Batch', 'Serial']
    assert len(loads) == len(released) == 1 and all(model is models[0] for model in models)
    assert all(row['observations_equal'] is (True if schema_valid else None) for row in report['comparisons'])
    assert all(row['both_schemas_valid'] is schema_valid for row in report['comparisons'])
    assert not report['accuracy_measured'] and not report['full_video_latency_measured']
    assert 'private_test_text' not in args.output.read_text()
    with pytest.raises(ValueError, match='new report file'):
        bench.benchmark(args)
