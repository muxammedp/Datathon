import argparse
import json
import re
import time
import traceback
from pathlib import Path

from evaluate_dataset import file_hash, save_json
from video_windows import SequentialFrames, add_visual_context, decode_window, plan_windows, scan_motion, summarize_windows, window_prompt


def main():
    parser = argparse.ArgumentParser(description='Experimental chronological window analysis, without automatic accusations')
    parser.add_argument('video', type=Path)
    parser.add_argument('--window-sec', type=float, default=16.)
    parser.add_argument('--overlap-sec', type=float, default=4.)
    parser.add_argument('--fps', type=float, default=4.)
    parser.add_argument('--detail-frames', type=int, default=96)
    parser.add_argument('--history-frames', type=int, default=32)
    parser.add_argument('--output', type=Path, default=Path('results/window-analysis'))
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--adapter-run', type=Path)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    if args.detail_frames + args.history_frames > 256:
        parser.error('Combined detail and history budget must not exceed 256 frames')
    import cv2
    capture = cv2.VideoCapture(str(args.video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        windows = plan_windows(total, fps, args.window_sec, args.overlap_sec, args.fps, args.detail_frames)
        print(json.dumps(dict(stage='motion_scan', source_frames=total)), flush=True)
        motion = scan_motion(capture, total)
        windows = add_visual_context(windows, motion, fps, args.detail_frames, args.history_frames)
        summary = dict(duration_sec=total / fps, source_frames=total, windows=len(windows),
                       frame_slots=sum(len(w['frame_indices']) for w in windows),
                       unique_sampled_frames=len({i for w in windows for i in w['frame_indices']}),
                       maximum_frames_per_window=max(len(w['frame_indices']) for w in windows),
                       all_source_frames_motion_scanned=True, inference_started=False)
        print(json.dumps(summary), flush=True)
        if args.plan_only:
            return
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Output directory is not empty; choose a new --output')
        args.output.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parent
        protocol = dict(video_sha256=file_hash(args.video), window_sec=args.window_sec, overlap_sec=args.overlap_sec,
                        sample_fps=args.fps, detail_frames=args.detail_frames, history_frames=args.history_frames,
                        model_revision=json.loads((root / 'model-manifest.json').read_text())['revision'],
                        code_sha256={name: file_hash(root / name) for name in
                                     ('analyze.py', 'analyze_windows.py', 'video_windows.py')})
        save_json(args.output / 'protocol.json', protocol)
        save_json(args.output / 'plan.json', dict(summary, intervals=windows))
        from analyze import load_model, PROMPT
        import torch
        model, tokenizer = load_model()
        adapter_threshold = None
        if args.adapter_run is not None:
            if args.fold not in range(5):
                raise ValueError('Adapter fold must be between 0 and 4')
            trained_protocol = json.loads((args.adapter_run / 'protocol.json').read_text(encoding='utf8'))
            if trained_protocol.get('version') != 'internvideo_events_v1':
                raise ValueError('This analysis requires an events-v1 adapter, not a whole-video adapter')
            if trained_protocol['model_revision'] != protocol['model_revision']:
                raise ValueError('Adapter base model revision mismatch')
            for key in ('window_sec', 'overlap_sec', 'fps', 'detail_frames', 'history_frames'):
                if getattr(args, key) != trained_protocol['settings'][key]:
                    raise ValueError('Sampling settings differ from adapter training: ' + key)
            for name in ('lora_binary.py', 'event_cache.py', 'video_windows.py', 'analyze.py'):
                if file_hash(root / name) != trained_protocol['code_sha256'][name]:
                    raise ValueError('Inference code differs from adapter training: ' + name)
            folder = args.adapter_run / f'fold-{args.fold}'
            metadata = json.loads((folder / 'result.json').read_text(encoding='utf8'))
            if file_hash(folder / 'adapter.safetensors') != metadata['adapter_sha256']:
                raise ValueError('Adapter checksum mismatch')
            from safetensors.torch import load_file
            from lora_binary import attach_adapters, binary_logits, frozen_inputs, restore_adapter
            from event_cache import binary_prompt
            model.requires_grad_(False)
            attach_adapters(model.language_model, trained_protocol['settings']['last_layers'],
                            trained_protocol['settings']['rank'])
            restore_adapter(model.language_model, load_file(str(folder / 'adapter.safetensors')))
            model.eval()
            adapter_threshold = metadata['best']['threshold']
            protocol.update(adapter_sha256=metadata['adapter_sha256'], adapter_fold=args.fold,
                            adapter_threshold=adapter_threshold, selected_epoch=metadata['best']['epoch'])
            save_json(args.output / 'protocol.json', protocol)
        records = []
        save_json(args.output / 'summary.json', summarize_windows(records, len(windows)))
        with SequentialFrames(args.video, windows, args.output) as reader:
            for window in windows:
                began = time.monotonic()
                record = {key: window[key] for key in ('index', 'start_sec', 'end_sec')}
                fatal = None
                try:
                    pixels, times = decode_window(reader, window, fps)
                    pixels = pixels.to(device='cuda', dtype=torch.bfloat16)
                    prefix = ''.join(f'Frame {i + 1}, absolute time {t:.3f} seconds: <image>\n'
                                     for i, t in enumerate(times))
                    prompt = window_prompt(window, PROMPT)
                    if adapter_threshold is None:
                        with torch.inference_mode():
                            response = model.chat(tokenizer, pixels, prefix + prompt,
                                                  {'do_sample': False, 'max_new_tokens': 12},
                                                  num_patches_list=[1] * len(times), history=None, verbose=False)
                        match = re.fullmatch(r'\s*([012])[.!]?\s*', response)
                        decision = int(match.group(1)) if match else 2
                        structured = bool(match)
                        del response
                    else:
                        with torch.no_grad():
                            embeds, ids = frozen_inputs(model, tokenizer, pixels, times,
                                                       window_prompt(window, binary_prompt(PROMPT)))
                            logits = binary_logits(model.language_model, embeds.to('cuda'), ids)
                            score = float(logits[0, 1] - logits[0, 0])
                            if not torch.isfinite(logits).all():
                                raise ValueError('Nonfinite adapted window score')
                        decision, structured = int(score >= adapter_threshold), True
                        record.update(score=score, threshold=adapter_threshold,
                                      trained_adapter_used=metadata['best']['epoch'] > 0,
                                      adapter_selected_epoch=metadata['best']['epoch'])
                        del embeds, logits
                    record.update(status='done', sampled_frames=len(times),
                                  history_frames=len(window['history_indices']),
                                  detail_frames=len(window['detail_indices']),
                                  first_sample_sec=times[0], last_sample_sec=times[-1],
                                  max_context_sample_gap_sec=max(b - a for a, b in zip(times, times[1:])),
                                  max_detail_sample_gap_sec=max((b - a) / fps for a, b in
                                                                zip(window['detail_indices'], window['detail_indices'][1:])),
                                  candidate_prediction={0: 'no_violation', 1: 'violation', 2: 'inconclusive'}[decision],
                                  structured_output=structured)
                    del pixels
                except Exception as exc:
                    record.update(status='failed', error_type=type(exc).__name__)
                    log = args.output / f'window-{window["index"]}-error.log'
                    log.write_text(traceback.format_exc(), encoding='utf8')
                    if isinstance(exc, RuntimeError):
                        fatal = exc
                record['elapsed_sec'] = round(time.monotonic() - began, 2)
                records.append(record)
                save_json(args.output / 'windows.json', records)
                save_json(args.output / 'summary.json', dict(summarize_windows(records, len(windows)),
                          cross_window_memory='sparse_visual_history', actor_tracking=False,
                          all_source_frames_motion_scanned=True))
                print(json.dumps(dict(stage='window_processed', total=len(windows), **record)), flush=True)
                if fatal is not None:
                    raise fatal
        print(json.dumps(dict(stage='finished', **summarize_windows(records, len(windows)))), flush=True)
    finally:
        capture.release()


if __name__ == '__main__':
    main()
