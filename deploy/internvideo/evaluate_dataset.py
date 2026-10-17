import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_dataset(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf8'))
    seen = set()
    for row in manifest['videos']:
        path = (directory / row['file']).resolve()
        if not path.is_relative_to(directory) or row['label'] not in (0, 1):
            raise ValueError('Invalid dataset entry')
        if row['sha256'] in seen or file_hash(path) != row['sha256']:
            raise ValueError('Duplicate or mismatched video checksum')
        seen.add(row['sha256'])
    return manifest


def summarize(records, total):
    counts = dict(tn=0, fp=0, fn=0, tp=0, inconclusive=0, failed=0)
    for row in records:
        if row['status'] != 'done':
            counts['failed'] += 1
            continue
        prediction = row['result']['prediction']
        if prediction not in ('violation', 'no_violation'):
            counts['inconclusive'] += 1
            continue
        positive = prediction == 'violation'
        key = ('tp' if positive else 'fn') if row['label'] else ('fp' if positive else 'tn')
        counts[key] += 1
    decided = sum(counts[k] for k in ('tn', 'fp', 'fn', 'tp'))
    positives = sum(row['label'] == 1 for row in records)
    negatives = sum(row['label'] == 0 for row in records)
    return dict(counts, attempted=len(records), dataset_total=total,
                complete=len(records) == total,
                decision_coverage=decided / len(records) if records else 0,
                accuracy_decided=(counts['tn'] + counts['tp']) / decided if decided else None,
                correct_fraction_all_attempted=(counts['tn'] + counts['tp']) / len(records) if records else None,
                recall_all_positive_videos=counts['tp'] / positives if positives else None,
                false_alarms_per_normal_video=counts['fp'] / negatives if negatives else None,
                categories_evaluated=False, actor_roles_evaluated=False,
                trained_on_this_dataset=False, independent_new_dataset=False,
                method='Frozen pretrained model, no label-driven fitting. Abstentions and failures are not correct negatives.')


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('open-dataset'))
    parser.add_argument('--output', type=Path, default=Path('results/baseline'))
    parser.add_argument('--fps', type=float, default=4.)
    parser.add_argument('--max-frames', type=int, default=256)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry failed videos, preserving completed predictions and settings')
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames < 4 or args.max_frames > 512 or args.max_frames % 4:
        parser.error('Use positive fps and a frame limit divisible by 4, between 4 and 512')
    manifest = check_dataset(args.dataset)
    rows = manifest['videos']
    print(json.dumps({'dataset_verified': True, 'unique_videos': len(rows),
                      'positive': sum(row['label'] for row in rows),
                      'negative': sum(not row['label'] for row in rows)}), flush=True)
    if args.check_only:
        return
    source = Path(__file__).resolve().parent
    model_manifest = json.loads((source / 'model-manifest.json').read_text(encoding='utf8'))
    protocol = {'dataset_manifest_sha256': file_hash(args.dataset / 'manifest.json'),
                'inference_code_sha256': file_hash(source / 'analyze.py'),
                'model_revision': model_manifest['revision'], 'fps': args.fps,
                'max_frames': args.max_frames, 'kind': 'pretrained_baseline_v1'}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf8')) != protocol:
        raise ValueError('Settings changed; use a new --output directory')
    save_json(protocol_path, protocol)
    records_path = args.output / 'records.json'
    records = json.loads(records_path.read_text(encoding='utf8')) if records_path.exists() else []
    expected = {row['sha256']: row for row in rows}
    if len({row['sha256'] for row in records}) != len(records):
        raise ValueError('Duplicate saved records')
    for row in records:
        if row['sha256'] not in expected or row['label'] != expected[row['sha256']]['label']:
            raise ValueError('Saved results do not match dataset')
    completed = {row['sha256'] for row in records
                 if row['status'] == 'done' or not args.retry_failed}
    pending = [row for row in rows if row['sha256'] not in completed]
    if pending:
        from analyze import load_model, analyze
        model, tokenizer = load_model()
    for row in pending:
        began = time.monotonic()
        fatal_error = None
        record = {'sha256': row['sha256'], 'file': row['file'],
                  'aliases': row['aliases'], 'label': row['label']}
        try:
            result = analyze(model, tokenizer, args.dataset / row['file'], args.fps, args.max_frames)
            record.update(status='done', result=result)
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__)
            error_path = args.output / ('error-' + row['sha256'] + '.log')
            with error_path.open('a', encoding='utf8') as error_file:
                error_file.write('\n' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
                error_file.write(traceback.format_exc())
            record['error_log'] = error_path.name
            if isinstance(exc, RuntimeError):
                fatal_error = exc
        record['elapsed_sec'] = round(time.monotonic() - began, 2)
        existing = next((i for i, saved in enumerate(records)
                         if saved['sha256'] == row['sha256']), None)
        if existing is None:
            records.append(record)
        else:
            records[existing] = record
        save_json(records_path, records)
        summary = summarize(records, len(rows))
        save_json(args.output / 'metrics.json', summary)
        print(json.dumps({'completed': len(records), 'total': len(rows),
                          'status': record['status'], 'prediction': record.get('result', {}).get('prediction'),
                          'error_type': record.get('error_type'), 'error_log': record.get('error_log'),
                          'expected_label': row['label'], 'elapsed_sec': record['elapsed_sec']}), flush=True)
        if fatal_error is not None:
            raise fatal_error
    summary = summarize(records, len(rows))
    save_json(args.output / 'metrics.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
