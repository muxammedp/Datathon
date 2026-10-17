import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.ensemble.benchmark_qwen import load_window, safe_output
from deploy.ensemble.common import file_hash, offline, save_json, verify_models
from deploy.ensemble.qwen_batched import BatchedQwenExpert, VIEWS
from deploy.ensemble.run import event


def diagnose(args):
    from deploy.ensemble.observation_diagnostics import trace_validation
    from deploy.ensemble import observation_schema

    output = safe_output(args.output, args.cache, args.models)
    config, identity, window = load_window(args.cache, args.video_sha)
    settings = config['settings']
    offline()
    started = time.perf_counter()
    verify_models(args.models, names=['qwen'])
    event('qwen_weights_verified', elapsed_sec=round(time.perf_counter() - started, 2))
    expert = BatchedQwenExpert(args.models / 'qwen', device=settings['device'],
                               max_frames=settings['qwen_frames'], max_new_tokens=settings['qwen_new_tokens'])
    try:
        started = time.perf_counter()
        expert._load()
        event('qwen_model_loaded', elapsed_sec=round(time.perf_counter() - started, 2))
        with trace_validation(validator_module=observation_schema) as records:
            result = expert._analyze_window(window, [])
        report = dict(version='qwen_observation_diagnostic_v2', video_sha256=identity, window_index=0,
                      extraction_fingerprint=config['fingerprint'], diagnostic_source_sha256=file_hash(__file__),
                      observation_schema_version=observation_schema.VERSION,
                      parse_status=result['parse_status'], validation_calls=len(records),
                      valid_observation_views=sum(status == 'valid' for status in result['parse_status'].values()),
                      accepted_event_counts={view: len(result.get('observations', {}).get(view, {}).get('events', []))
                                             if result['parse_status'].get(view) == 'valid' else None for view in VIEWS},
                      diagnostics=[dict(record, view=VIEWS[i] if len(records) == 2 else 'unmapped')
                                   for i, record in enumerate(records)],
                      profile=result.get('profile'), elapsed_sec=result['elapsed_sec'],
                      raw_responses_exported=False, extraction_cache_modified=False,
                      note='First-window diagnostic only. The selected observation schema validates every event; '
                           'a formatting probe neither accepts nor changes observations. '
                           'Only fixed reason codes and structural metadata are exported.')
        save_json(output, report)
        event('qwen_diagnostic', **report)
        return report
    finally:
        expert.release()


def main():
    parser = argparse.ArgumentParser(description='Explain Qwen observation parsing failures without exporting scene text.')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video-sha')
    parser.add_argument('--require-valid', action='store_true',
                        help='Exit nonzero after saving the report unless both observation views validate.')
    args = parser.parse_args()
    try:
        report = diagnose(args)
        if getattr(args, 'require_valid', False) and any(report.get('parse_status', {}).get(view) != 'valid' for view in VIEWS):
            event('qwen_diagnostic_not_valid', valid_observation_views=report.get('valid_observation_views', 0))
            raise SystemExit(2)
    except Exception as exc:
        event('qwen_diagnostic_failed', error_type=type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
