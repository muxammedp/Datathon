import json
import time

from evaluate_dataset import file_hash, save_json
from video_windows import add_visual_context, decode_window, plan_windows, scan_motion, window_prompt


def binary_prompt(base_prompt):
    return base_prompt.split('Answer with exactly one digit:')[0] + (
        'Classify only the current interval. Answer with exactly one digit: '
        '0 for no clearly established violation, 1 for a clearly established violation. '
        'Do not provide names or descriptions.')


def prepare(model, tokenizer, rows, args, prompt):
    import cv2
    import torch
    from safetensors.torch import save_file
    from lora_binary import frozen_inputs
    directory = args.output / 'window-cache'
    directory.mkdir(exist_ok=True)
    all_windows, answer_ids = [], None
    for number, row in enumerate(rows):
        began = time.monotonic()
        capture = cv2.VideoCapture(str(args.dataset / row['file']))
        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            plan = plan_windows(total, source_fps, args.window_sec, args.overlap_sec, args.fps, args.detail_frames)
            motion = scan_motion(capture, total)
            windows = add_visual_context(plan, motion, source_fps, args.detail_frames, args.history_frames)
            for window in windows:
                key = row['sha256'] + '-' + str(window['index'])
                path = directory / (key + '.safetensors')
                metadata_path = path.with_suffix('.json')
                if path.exists() and metadata_path.exists():
                    meta = json.loads(metadata_path.read_text(encoding='utf8'))
                    if file_hash(path) != meta['cache_sha256'] or meta['frame_indices'] != window['frame_indices']:
                        raise ValueError('Window cache does not match the locked input plan')
                    ids = meta['answer_ids']
                else:
                    pixels, times = decode_window(capture, window, source_fps)
                    pixels = pixels.to('cuda', dtype=torch.bfloat16)
                    embeds, ids = frozen_inputs(model, tokenizer, pixels, times, window_prompt(window, prompt))
                    if not torch.isfinite(embeds).all():
                        raise ValueError('Nonfinite window embeddings')
                    temporary = path.with_suffix('.tmp')
                    save_file({'embeds': embeds}, str(temporary))
                    temporary.replace(path)
                    save_json(metadata_path, dict(answer_ids=ids, cache_sha256=file_hash(path),
                                                  frame_indices=window['frame_indices']))
                    del pixels, embeds
                if answer_ids is not None and answer_ids != ids:
                    raise ValueError('Answer tokenization changed between windows')
                answer_ids = ids
                window['cache_key'] = key
            all_windows.append(windows)
        finally:
            capture.release()
        print(json.dumps(dict(stage='event_inputs', video=number + 1, total_videos=len(rows),
                              windows=len(windows), motion_scanned_frames=total,
                              elapsed_sec=round(time.monotonic() - began, 2))), flush=True)
    save_json(directory / 'index.json', dict(windows=all_windows, answer_ids=answer_ids))
    torch.cuda.empty_cache()
    return directory, all_windows, answer_ids


def score_windows(language_model, rows, indices, windows, directory, answer_ids, device):
    import torch
    from safetensors.torch import load_file
    from lora_binary import binary_logits
    language_model.eval()
    result = {}
    with torch.no_grad():
        for index in indices:
            scores = []
            for window in windows[index]:
                embeds = load_file(str(directory / (window['cache_key'] + '.safetensors')), device=str(device))['embeds']
                logits = binary_logits(language_model, embeds, answer_ids)
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite window score')
                scores.append(float(logits[0, 1] - logits[0, 0]))
            result[index] = scores
    return result
