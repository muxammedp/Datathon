import argparse
import hashlib
import json
import os
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.engine import LocalAnalyzer


def labels_from_xlsx(path):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(node.itertext()) for node in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        root = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        labels = []
        for row in root.findall('.//s:row', ns)[1:]:
            cells = {}
            for cell in row.findall('s:c', ns):
                val = cell.find('s:v', ns)
                inline = cell.find('s:is', ns)
                value = val.text if val is not None else ''.join(inline.itertext()) if inline is not None else ''
                cells[re.sub(r'\d', '', cell.attrib['r'])] = strings[int(value)] if cell.get('t') == 's' else value
            name = cells.get('A', '').strip()
            if not name:
                continue
            values = list(cells.values())
            clean = any('без нарушени' in v.lower() for v in values)
            timestamp_text = cells.get('B', '')
            timestamp = None
            if not clean:
                numbers = list(map(int, re.findall(r'\d+', timestamp_text)))
                timestamp = numbers[0] * 60 + numbers[1] if len(numbers) > 1 and 'мин' in timestamp_text else numbers[0] if numbers else None
            labels.append({'name': name, 'violation': not clean, 'timecode_sec': timestamp})
        return labels


def normalize(name):
    return name.upper().replace('Х', 'X')


def summarize(records):
    successful = [r for r in records if r['status'] == 'done']
    decided = [r for r in successful if isinstance(r.get('predicted_violation'), bool)]
    tp = sum(r['expected_violation'] and r['predicted_violation'] for r in decided)
    tn = sum(not r['expected_violation'] and not r['predicted_violation'] for r in decided)
    fp = sum(not r['expected_violation'] and r['predicted_violation'] for r in decided)
    fn = sum(r['expected_violation'] and not r['predicted_violation'] for r in decided)
    errors = [r['timestamp_error_sec'] for r in successful if r.get('timestamp_error_sec') is not None]
    unique = {}
    for record in records:
        unique.setdefault(record['video_sha256'], record)
    unique_correct = sum(r['status'] == 'done' and r['expected_violation'] == r.get('predicted_violation') for r in unique.values())
    return {'dataset': 'provided_open_dataset', 'method': 'Runtime audit, not independent holdout. Training quality is in data/full-video-lora-4b/metrics.json. Inconclusive results are abstentions, never correct negatives.',
            'attempted': len(records), 'completed': len(successful), 'failed': len(records) - len(successful),
            'inconclusive': len(successful) - len(decided), 'decision_coverage': len(decided) / max(1, len(records)),
            'true_positive': tp, 'true_negative': tn, 'false_positive': fp, 'false_negative': fn,
            'accuracy_completed': round((tp + tn) / len(decided), 4) if decided else None,
            'accuracy_all_attempted': round((tp + tn) / len(records), 4) if records else None,
            'precision': round(tp / (tp + fp), 4) if tp + fp else None,
            'recall': round(tp / (tp + fn), 4) if tp + fn else None,
            'recall_all_positives': round(tp / sum(r['expected_violation'] for r in records), 4) if any(r['expected_violation'] for r in records) else None,
            'mean_timestamp_error_on_true_positives_sec': round(sum(errors) / len(errors), 2) if errors else None,
            'max_processing_sec': max((r['elapsed_sec'] for r in records), default=0),
            'within_5_minutes': sum(r['elapsed_sec'] < 300 and r['status'] == 'done' for r in records),
            'classification_accuracy': None, 'classification_note': 'No type labels provided; not measurable',
            'unseen_camera_evaluation': False, 'unique_videos': len(unique),
            'unique_accuracy_all_attempted': round(unique_correct / len(unique), 4) if unique else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'evaluation')
    parser.add_argument('--legacy', action='store_true', help='Reproduce the retired, unreliable Qwen detector')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels = labels_from_xlsx(args.dataset / 'open_labels.xlsx')
    if args.ids:
        labels = [r for r in labels if normalize(r['name']) in [normalize(x) for x in args.ids]]
    if args.limit:
        labels = labels[:args.limit]
    videos = {normalize(p.stem): p for p in (args.dataset / 'videos').glob('*.mov')}
    from app.full_video import FullVideoAnalyzer
    analyzer = LocalAnalyzer() if args.legacy else FullVideoAnalyzer()
    analyzer.load()
    records = []
    cache = {}
    for index, label in enumerate(labels):
        output = args.output / (normalize(label['name']) + '.json')
        began = time.monotonic()
        video = videos[normalize(label['name'])]
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        reused = digest in cache
        record = {'name': label['name'], 'video_sha256': digest, 'inference_reused_for_identical_file': reused,
                  'expected_violation': label['violation'], 'expected_timecode_sec': label['timecode_sec']}
        try:
            result = cache[digest] if reused else analyzer.analyze(video)
            cache[digest] = result
            output.write_text(json.dumps(result, ensure_ascii=False), encoding='utf8')
            times = [e['timecode_sec'] for e in result['violations']]
            record.update(status='done', predicted_violation=None if result['verdict'] == 'inconclusive' else result['verdict'] == 'violation',
                          events=len(times), elapsed_sec=result['analysis']['elapsed_sec'],
                          format_retries=result['analysis'].get('format_retries', 0),
                          timestamp_error_sec=min(abs(t - label['timecode_sec']) for t in times) if times and label['violation'] and label['timecode_sec'] is not None else None)
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error_message=str(exc)[:300], elapsed_sec=round(time.monotonic() - began, 2))
        records.append(record)
        (args.output / 'metrics.json').write_text(json.dumps(summarize(records), indent=2), encoding='utf8')
        (args.output / 'records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({'completed': index + 1, 'total': len(labels), **record}, ensure_ascii=True), flush=True)
    print(json.dumps(summarize(records)), flush=True)


if __name__ == '__main__':
    main()

