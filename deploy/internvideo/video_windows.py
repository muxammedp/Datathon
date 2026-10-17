import math


def scan_motion(capture, expected_frames):
    import cv2
    import numpy as np
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    scores, previous = [], None
    for index in range(expected_frames):
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f'Full-frame motion scan stopped at frame {index} of {expected_frames}')
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 96)).astype(np.float32)
        if previous is None:
            score = 0.
        else:
            difference = np.abs(gray - previous)
            tiles = difference.reshape(6, 16, 10, 16).mean(axis=(1, 3))
            score = float(tiles.max())
        scores.append(score)
        previous = gray
    return scores


def add_visual_context(windows, motion_scores, source_fps, detail_frames=96, history_frames=32):
    if (detail_frames < 4 or detail_frames % 4 or history_frames < 0 or history_frames % 4
            or any(not math.isfinite(score) or score < 0 for score in motion_scores)):
        raise ValueError('Invalid adaptive sampling settings')
    result = []
    for window in windows:
        start, end = window['start_frame'], window['end_frame_exclusive']
        if end > len(motion_scores) or len(window['frame_indices']) > detail_frames:
            raise ValueError('Insufficient motion coverage or detail frame budget')
        detail = set(window['frame_indices'])
        peaks = []
        candidates = sorted(range(start, end), key=lambda i: (-motion_scores[i], i))
        for index in candidates:
            if motion_scores[index] <= 0 or len(detail) >= detail_frames:
                break
            if any(abs(index - peak) < max(1, round(source_fps * .25)) for peak in peaks):
                continue
            additions = {i for i in (index - 1, index, index + 1) if start <= i < end}
            if len(detail | additions) <= detail_frames:
                detail.update(additions)
                peaks.append(index)
        detail = sorted(detail)
        while len(detail) % 4:
            detail.append(detail[-1])
        history = []
        if start > 0 and history_frames:
            count = history_frames // 2
            anchors = {round(i * (start - 1) / max(1, count - 1)) for i in range(count)}
            prior_peaks = sorted((i for i in range(start) if motion_scores[i] > 0),
                                 key=lambda i: (-motion_scores[i], i))
            for index in prior_peaks:
                if len(anchors) >= history_frames:
                    break
                if all(abs(index - other) >= max(1, round(source_fps)) for other in anchors):
                    anchors.add(index)
            history = sorted(anchors)
            while len(history) % 4:
                history.append(history[-1])
        result.append(dict(window, frame_indices=history + detail,
                           history_indices=history, detail_indices=detail, motion_peaks=peaks))
    return result


def plan_windows(total_frames, source_fps, seconds=16., overlap=4., sample_fps=4., max_frames=128):
    values = (source_fps, seconds, overlap, sample_fps)
    if (total_frames <= 0 or any(not math.isfinite(v) for v in values)
            or source_fps <= 0 or seconds <= 0 or sample_fps <= 0
            or not 0 <= overlap < seconds or max_frames < 4 or max_frames % 4):
        raise ValueError('Invalid window sampling settings')
    width = max(1, round(seconds * source_fps))
    stride = max(1, round((seconds - overlap) * source_fps))
    windows = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + width)
        required = math.ceil((end - start - 1) / source_fps * sample_fps) + 1
        count = max(4, 4 * math.ceil(required / 4))
        if count > max_frames:
            raise ValueError('Window requires too many frames; shorten the window or increase --max-frames. '
                             'Sampling density is never silently reduced.')
        indices = [start + round(i * (end - start - 1) / (count - 1)) for i in range(count)]
        windows.append(dict(index=len(windows), start_frame=start, end_frame_exclusive=end,
                            start_sec=start / source_fps, end_sec=end / source_fps,
                            frame_indices=indices))
        if end == total_frames:
            break
        start += stride
    return windows


def decode_window(capture, window, source_fps):
    import cv2
    from PIL import Image
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    tensors, times = [], []
    for index in window['frame_indices']:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        if not ok or actual != index:
            raise ValueError('Selected frame could not be decoded accurately; window requires review')
        tensors.append(transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        times.append(actual / source_fps)
    return torch.stack(tensors), times


def summarize_windows(records, expected):
    done = [r for r in records if r['status'] == 'done']
    candidates = [dict(start_sec=r['start_sec'], end_sec=r['end_sec'])
                  for r in done if r['candidate_prediction'] == 'violation']
    return dict(processed_windows=len(records), expected_windows=expected,
                all_windows_attempted=len(records) == expected,
                coverage_complete=len(done) == expected,
                failed_windows=sum(r['status'] != 'done' for r in records),
                inconclusive_windows=sum(r['candidate_prediction'] == 'inconclusive' for r in done),
                candidate_window_count=len(candidates), candidate_intervals=candidates,
                prediction='inconclusive', review_required=True, violations=[],
                production_approved=False, actor_roles_verified=False,
                cross_window_memory='sparse_visual_history' if any(r.get('history_frames', 0) for r in done) else False,
                trained_adapter_used=any(r.get('trained_adapter_used', False) for r in done),
                note='Candidate intervals are unvalidated window-level model outputs, not confirmed events. '
                     'All-negative windows do not establish that the video is normal. '
                     'Timestamps use frame index / reported FPS; variable-frame-rate PTS are not verified.')


def window_prompt(window, base_prompt):
    return (f'The current interval is {window["start_sec"]:.3f} to {window["end_sec"]:.3f} seconds. '
            f'The first {len(window["history_indices"])} frames are sparse visual history from earlier '
            'in the SAME recording. Use them for context, but classify only the current interval. '
            'History is incomplete: missing actions are not proof they never occurred. '
            'Events may start before or end after the current interval. Do not infer a violation '
            'from a truncated action or from the absence of a sampled action. ' + base_prompt)


def annotated_windows(row, windows):
    if row['label'] == 0:
        return list(range(len(windows)))
    times = row.get('event_times', [])
    if not times or any(not math.isfinite(t) or t < 0 or t >= windows[-1]['end_sec'] for t in times):
        raise ValueError('Positive video has missing or out-of-range event timestamps')
    selected = [i for i, w in enumerate(windows) if any(w['start_sec'] <= t < w['end_sec'] for t in times)]
    if not selected:
        raise ValueError('No window covers the annotated event')
    return selected
