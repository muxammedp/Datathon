import base64
import importlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ['FORTE_SKIP_WARMUP'] = '1'
os.environ['FORTE_API_KEY'] = 'test-key-not-for-deployment-123456789'


@pytest.fixture(scope='module')
def service(tmp_path_factory):
    os.environ['FORTE_DATA_DIR'] = str(tmp_path_factory.mktemp('service'))
    module = importlib.import_module('app.main')
    with TestClient(module.app) as client:
        yield module, client


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'fixture.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (128, 96))
    assert writer.isOpened()
    for i in range(20):
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i + 10, 30), (i + 35, 55), (90, 170, 210), -1)
        writer.write(frame)
    writer.release()
    return path


def headers():
    return {'X-API-Key': os.environ['FORTE_API_KEY']}


def test_key_required_on_upload_and_results(service):
    _, client = service
    assert client.post('/api/v1/analyze', files={'file': ('test.mov', b'video')}).status_code == 401
    assert client.get('/api/v1/jobs/unknown').status_code == 401
    assert client.get('/api/v1/auth', headers={'X-API-Key': 'wrong'}).status_code == 401
    assert client.get('/api/v1/auth', headers=headers()).status_code == 200


def test_static_and_rules(service):
    _, client = service
    response = client.get('/')
    assert response.status_code == 200
    assert 'ForteVision' in response.text
    assert "connect-src 'self'" in response.headers['content-security-policy']
    rules = client.get('/api/rules').json()
    assert len(rules) == 8
    assert all(r['source']['url'].startswith('https://adilet.zan.kz/') for r in rules)
    assert 'criterion' not in rules[0]


def test_legacy_accusations_are_withdrawn_without_overwriting_original(service):
    import uuid
    module, client = service
    job_id = str(uuid.uuid4())
    (module.JOBS / job_id).mkdir()
    original = {'job_id': job_id, 'created_at': time.time(), 'status': 'done',
                'verdict': 'violation', 'violations': [{'type': 'synthetic unsupported claim'}],
                'analysis': {'model': 'Qwen3-VL-2B-Instruct'}}
    module.save_job(original)
    result = client.get(f'/api/v1/jobs/{job_id}', headers=headers()).json()
    assert result['legacy_result_unreliable'] is True
    assert result['verdict'] == 'inconclusive'
    assert result['violations'] == []
    assert result['review_required'] is True
    assert module.load_job(job_id) == original


def test_unavailable_model_is_not_a_clean_verdict(service):
    module, client = service
    module.state['engine'] = 'unavailable'
    response = client.post('/api/v1/analyze', headers=headers(), files={'file': ('test.mov', b'video')})
    assert response.status_code == 503
    assert 'verdict' not in response.json()


def test_reject_bad_uploads_and_ids(service):
    module, client = service
    module.state['engine'] = 'ready'
    assert client.post('/api/v1/analyze', headers=headers(), files={'file': ('file.txt', b'test')}).status_code == 415
    assert client.post('/api/v1/analyze', headers=headers(), files={'file': ('file.mov', b'')}).status_code == 422
    assert client.post('/api/v1/analyze', headers=headers(), files={'file': ('file.mov', b'broken')}).status_code == 422
    assert client.get('/api/v1/jobs/not-a-uuid', headers=headers()).status_code == 404


def test_job_lifecycle_and_evidence(service, video, monkeypatch):
    from app.engine import evidence
    module, client = service
    module.state['engine'] = 'ready'
    def fixture_analysis(path, progress):
        progress(50, 'Synthetic test only')
        time.sleep(.1)
        return {'verdict': 'violation', 'violations': [
            {'type': 'Synthetic test event', 'timecode_sec': 1.0, 'description': 'Synthetic fixture, not a detection result',
             'evidence_frames': evidence(path, 1.0, 2.0)}]}
    monkeypatch.setattr(module.analyzer, 'analyze', fixture_analysis)
    with video.open('rb') as handle:
        response = client.post('/api/v1/analyze', headers=headers(), files={'file': ('fixture.mov', handle, 'video/quicktime')})
    assert response.status_code == 202
    job_id = response.json()['job_id']
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = client.get(f'/api/v1/jobs/{job_id}', headers=headers()).json()
        if result['status'] != 'processing':
            break
        time.sleep(.05)
    assert result['status'] == 'done'
    for frame in result['violations'][0]['evidence_frames']:
        raw = base64.b64decode(frame['image_base64'])
        assert raw[:2] == b'\xff\xd8'
        assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None
        assert 0 <= frame['timestamp_sec'] < 2
    assert not (module.JOBS / job_id / 'input.mov').exists()
    assert client.delete(f'/api/v1/jobs/{job_id}', headers=headers()).status_code == 200
    assert client.get(f'/api/v1/jobs/{job_id}', headers=headers()).status_code == 404


def test_failure_status_and_release(service, video, monkeypatch):
    module, client = service
    def fail(path, progress):
        raise RuntimeError('Synthetic inference failure')
    monkeypatch.setattr(module.analyzer, 'analyze', fail)
    with video.open('rb') as handle:
        response = client.post('/api/v1/analyze', headers=headers(), files={'file': ('test.mov', handle)})
    job_id = response.json()['job_id']
    for _ in range(100):
        result = client.get(f'/api/v1/jobs/{job_id}', headers=headers()).json()
        if result['status'] != 'processing':
            break
        time.sleep(.02)
    assert result['status'] == 'failed'
    assert 'verdict' not in result
    assert not (module.JOBS / job_id / 'input.mov').exists()


def test_capacity_and_size(service, monkeypatch):
    module, client = service
    module.capacity.acquire()
    try:
        assert client.post('/api/v1/analyze', headers=headers(), files={'file': ('test.mov', b'123')}).status_code == 429
    finally:
        module.capacity.release()
    monkeypatch.setattr(module, 'MAX_BYTES', 4)
    assert client.post('/api/v1/analyze', headers=headers(), files={'file': ('test.mov', b'12345')}).status_code == 413


def test_malformed_model_json_does_not_become_clean():
    from app.engine import parse_json
    with pytest.raises(ValueError):
        parse_json('no violations')
    with pytest.raises(ValueError):
        parse_json('{"unexpected": []}')
    assert parse_json('```json\n{"events":[]}\n```') == {'events': []}


def test_video_limits(video):
    from app.engine import video_info
    info = video_info(video)
    assert info['duration_sec'] == 2
    assert info['frame_count'] == 20


def test_truncated_tail_uses_actual_frame_time(monkeypatch):
    from app.engine import read_frames
    class TruncatedVideo:
        index = 0
        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_POS_MSEC: self.index * 100,
                    cv2.CAP_PROP_POS_FRAMES: self.index + 1}.get(prop, 0)
        def set(self, prop, value):
            self.index = value
        def read(self):
            if self.index >= 17:
                return False, None
            return True, np.zeros((64, 64, 3), dtype=np.uint8)
        def release(self):
            pass
    monkeypatch.setattr(cv2, 'VideoCapture', lambda path: TruncatedVideo())
    frames = read_frames('synthetic-truncated.mov', [1.9])
    assert len(frames) == 1
    assert frames[0][0] == 1.5
    assert frames[0][0] != 1.9


@pytest.mark.parametrize('responses,success', [
    (['invalid', '{"events":[]}'], True),
    (['invalid', 'still invalid'], False),
])
def test_model_format_retry_is_bounded(monkeypatch, responses, success):
    import contextlib
    import sys
    import types
    from app.engine import LocalAnalyzer
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(inference_mode=contextlib.nullcontext))
    class Inputs(dict):
        input_ids = np.zeros((1, 1))
        def to(self, device):
            return self
    class Processor:
        calls = 0
        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            return Inputs()
        def batch_decode(self, output, **kwargs):
            return [responses[self.calls - 1]]
    analyzer = LocalAnalyzer()
    analyzer.processor = Processor()
    analyzer.model = types.SimpleNamespace(generate=lambda **kwargs: np.zeros((1, 2)))
    if success:
        assert analyzer.infer([], 'Synthetic schema test') == {'events': []}
    else:
        with pytest.raises(ValueError):
            analyzer.infer([], 'Synthetic schema test')
    assert analyzer.processor.calls == 2
