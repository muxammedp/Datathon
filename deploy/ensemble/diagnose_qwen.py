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
        with trace_validation() as records:
            result = expert._analyze_window(window, [])
        report = dict(version='qwen_observation_diagnostic_v1', video_sha256=identity, window_index=0,
                      extraction_fingerprint=config['fingerprint'], diagnostic_source_sha256=file_hash(__file__),
                      parse_status=result['parse_status'], validation_calls=len(records),
                      diagnostics=[dict(record, view=VIEWS[i] if len(records) == 2 else 'unmapped')
                                   for i, record in enumerate(records)],
                      profile=result.get('profile'), elapsed_sec=result['elapsed_sec'],
                      raw_responses_exported=False, extraction_cache_modified=False,
                      note='First-window diagnostic only. Original strict validation is preserved; '
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
    args = parser.parse_args()
    try:
        diagnose(args)
    except Exception as exc:
        event('qwen_diagnostic_failed', error_type=type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
