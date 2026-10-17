import argparse
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.ensemble.common import digest, file_hash, offline, read_json, save_json, verify_models
from deploy.ensemble.qwen_expert import QwenExpert, _window
from deploy.ensemble.qwen_batched import BatchedQwenExpert
from deploy.ensemble.run import event


def load_window(cache, video_sha=None):
    cache = Path(cache).resolve()
    config = read_json(cache / 'protocol.json')
    if config['version'] == 'forte_specialists_v1':
        from deploy.ensemble.run import protocol
    elif config['version'] in ('forte_specialists_batched_v1', 'forte_specialists_batched_v2'):
        from deploy.ensemble.run_batched import protocol
    else:
        raise ValueError('Unsupported extraction protocol')
    if config != protocol(config['settings'], config['camera']):
        raise ValueError('Cache differs from the installed extraction code or settings')
    directories = sorted((cache / 'videos').iterdir())
    if video_sha is not None:
        directories = [path for path in directories if path.name == video_sha]
    for directory in directories:
        root = directory / 'media'
        if not (root / 'complete.json').exists():
            continue
        index = read_json(root / 'index.json')
        complete = read_json(root / 'complete.json')
        if (complete.get('index_sha256') != file_hash(root / 'index.json')
                or index.get('video_sha256') != directory.name or not index.get('coverage_complete')):
            raise ValueError('Media identity or completion checksum mismatch')
        raw = index['windows'][0]
        if raw['index'] != 0:
            raise ValueError('Benchmark needs the first chronological window with empty prior memory')
        window = _window(raw, root, config['settings']['qwen_frames'])
        checksums = dict(zip(raw['frame_paths'], raw['frame_sha256']))
        for relative, path in zip(window['frame_paths'], window['paths']):
            if file_hash(path) != checksums[relative]:
                raise ValueError('Benchmark frame checksum mismatch')
        return config, index['video_sha256'], window
    raise ValueError('No completed media cache found')


def safe_output(output, cache, models):
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(Path(root).resolve()) for root in (cache, models)):
        raise ValueError('Choose a new report file outside the extraction cache and model directory')
    return output


def benchmark(args):
    import torch
    from PIL import Image

    output = safe_output(args.output, args.cache, args.models)
    config, identity, window = load_window(args.cache, args.video_sha)
    settings = config['settings']
    device = settings['device']
    cuda = str(device).startswith('cuda')
    offline()
    started = time.perf_counter()
    verify_models(args.models, names=['qwen'])
    verified_sec = time.perf_counter() - started
    event('qwen_weights_verified', elapsed_sec=round(verified_sec, 2))
    serial = QwenExpert(args.models / 'qwen', device=device, max_frames=settings['qwen_frames'],
                        max_new_tokens=settings['qwen_new_tokens'])
    batch = BatchedQwenExpert(args.models / 'qwen', device=device, max_frames=settings['qwen_frames'],
                             max_new_tokens=settings['qwen_new_tokens'])
    runs, results = [], []
    try:
        started = time.perf_counter()
        serial._load()
        batch.model, batch.processor, batch.answer_ids = serial.model, serial.processor, serial.answer_ids
        load_sec = time.perf_counter() - started
        event('qwen_model_loaded', elapsed_sec=round(load_sec, 2))
        with Image.open(window['paths'][0]) as source:
            picture = source.convert('RGB')
        try:
            _, inputs = serial._inputs([picture], [window['times'][0]], 'Answer exactly one digit: 0 or 1.')
            with torch.inference_mode():
                serial.model.generate(**inputs, do_sample=False, max_new_tokens=1, use_cache=True)
            del inputs
        finally:
            picture.close()
        if cuda:
            torch.cuda.synchronize(device)
        for repeat in range(args.repeats):
            order = [('serial', serial), ('batched', batch)]
            if repeat % 2:
                order.reverse()
            pair = {}
            for name, expert in order:
                if cuda:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = expert._analyze_window(window, [])
                if cuda:
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                record = dict(execution=name, repeat=repeat + 1, elapsed_sec=elapsed,
                              sampled_frames=result['sampled_frames'], model_passes=result['model_passes'],
                              scores={view: result[view + '_score'] for view in ('context', 'cash')},
                              parse_status=result['parse_status'], profile=result.get('profile'),
                              peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3 if cuda else None)
                runs.append(record)
                pair[name] = result
                event('benchmark_window', **record)
            schemas_valid = all(status == 'valid' for row in pair.values()
                                for status in row['parse_status'].values())
            results.append(dict(repeat=repeat + 1,
                                max_score_difference=max(abs(pair['serial'][v + '_score'] - pair['batched'][v + '_score'])
                                                         for v in ('context', 'cash')),
                                observations_equal=(pair['serial']['observations'] == pair['batched']['observations'])
                                                   if schemas_valid else None,
                                both_schemas_valid=schemas_valid))
        medians = {name: statistics.median(r['elapsed_sec'] for r in runs if r['execution'] == name)
                   for name in ('serial', 'batched')}
        report = dict(version='qwen_batch_benchmark_v1', video_sha256=identity, window_index=0,
                      extraction_fingerprint=config['fingerprint'], settings=settings,
                      benchmark_source_sha256=file_hash(__file__), prior_memory=[],
                      window_input_digest=digest({key: value for key, value in window.items() if key != 'paths'}),
                      verification_sec=verified_sec, model_load_sec=load_sec, runs=runs, comparisons=results,
                      median_window_sec=medians, speedup=medians['serial'] / medians['batched'],
                      device_name=torch.cuda.get_device_name(device) if cuda else 'CPU',
                      torch_version=torch.__version__, accuracy_measured=False, full_video_latency_measured=False,
                      note='Local first-window comparison with one-token warmup; run without competing GPU jobs. '
                           'Repeated pairs alternate order. Small BF16 differences can change generated observations; '
                           'use a separate cache and retrain fusion for batched extraction.')
        save_json(output, report)
        event('benchmark_complete', median_window_sec=medians, speedup=report['speedup'], output=str(output))
        return report
    finally:
        batch.model, batch.processor, batch.answer_ids = None, None, None
        serial.release()


def main():
    parser = argparse.ArgumentParser(description='Compare serial and batched Qwen on one cached window, entirely locally.')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video-sha')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error('--repeats must be 1..5')
    benchmark(args)


if __name__ == '__main__':
    main()
