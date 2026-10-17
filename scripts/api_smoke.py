import argparse
import base64
import json
import time
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--key-file', type=Path, default=ROOT / 'data' / 'api-key.txt')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'validation')
    args = parser.parse_args()
    key = args.key_file.read_text(encoding='utf8').strip()
    with httpx.Client(base_url=args.base_url, timeout=90, trust_env=False) as client:
        assert client.get('/api/v1/auth').status_code == 401
        health = client.get('/api/health').json()
        assert health['engine'] == 'ready', health['engine']
        started = time.monotonic()
        with args.video.open('rb') as file:
            response = client.post('/api/v1/analyze', headers={'X-API-Key': key}, files={'file': (args.video.name, file, 'video/quicktime')})
        assert response.status_code == 202, response.status_code
        job_id = response.json()['job_id']
        while time.monotonic() - started < 300:
            response = client.get(f'/api/v1/jobs/{job_id}', headers={'X-API-Key': key})
            response.raise_for_status()
            result = response.json()
            if result['status'] in ('done', 'failed'):
                break
            time.sleep(2)
        assert result['status'] == 'done', result.get('error', 'timeout')
        assert result['verdict'] in ('violation', 'no_violation', 'inconclusive')
        assert (result['verdict'] == 'violation') == bool(result['violations'])
        if result['verdict'] == 'inconclusive':
            assert result['review_required'] is True
            assert not result['violations']
        evidence_count = 0
        for event in result['violations'] + result.get('review_segments', []):
            assert isinstance(event['type'], str) and isinstance(event['description'], str)
            assert 0 <= event['timecode_sec'] < result['video']['duration_sec']
            assert 1 <= len(event['evidence_frames']) <= 3
            for frame in event['evidence_frames']:
                raw = base64.b64decode(frame['image_base64'], validate=True)
                assert raw[:2] == b'\xff\xd8'
                with Image.open(BytesIO(raw)) as jpeg:
                    assert jpeg.format == 'JPEG'
                    jpeg.load()
                assert 0 <= frame['timestamp_sec'] < result['video']['duration_sec']
                evidence_count += 1
        folder = args.output_dir
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'api_real_result.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf8')
        report = {'http_upload_status': 202, 'unauthenticated_status': 401, 'status': result['status'],
                  'elapsed_sec': round(time.monotonic() - started, 2), 'events': len(result['violations']),
                  'verdict': result['verdict'], 'review_segments': len(result.get('review_segments', [])),
                  'evidence_jpegs_validated': evidence_count, 'schema_passed': True,
                  'input_deleted': not any((ROOT / 'data' / 'jobs' / job_id).glob('input.*'))}
        assert report['input_deleted']
        assert report['elapsed_sec'] < 300
        (folder / 'api_smoke.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
