import importlib.util
from pathlib import Path

import pytest

source = Path(__file__).resolve().parents[1] / 'deploy/internvideo/video_windows.py'
spec = importlib.util.spec_from_file_location('video_windows', source)
windows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(windows)


@pytest.mark.parametrize('total', [1, 2, 29, 480, 481, 2150, 18000, 108000])
def test_whole_timeline_is_covered_without_a_global_frame_cap(total):
    plan = windows.plan_windows(total, 30.)
    assert plan[0]['start_frame'] == 0
    assert plan[-1]['end_frame_exclusive'] == total
    for left, right in zip(plan, plan[1:]):
        assert right['start_frame'] < left['end_frame_exclusive']
    for row in plan:
        indices = row['frame_indices']
        assert len(indices) % 4 == 0 and len(indices) <= 68
        assert indices[0] == row['start_frame']
        assert indices[-1] == row['end_frame_exclusive'] - 1
        assert all(0 <= b - a <= 8 for a, b in zip(indices, indices[1:]))
    if total >= 18000:
        assert sum(len(row['frame_indices']) for row in plan) > 2000


def test_dense_sampling_is_not_silently_reduced_to_fit_frame_limit():
    with pytest.raises(ValueError, match='too many frames'):
        windows.plan_windows(18000, 30., sample_fps=30., max_frames=128)
    for overlap in (-1, 16, 17):
        with pytest.raises(ValueError, match='Invalid'):
            windows.plan_windows(18000, 30., overlap=overlap)


@pytest.mark.parametrize('candidate', ['violation', 'no_violation', 'inconclusive'])
def test_window_predictions_are_not_automatic_video_verdicts(candidate):
    records = [dict(status='done', start_sec=0, end_sec=16, candidate_prediction=candidate)]
    result = windows.summarize_windows(records, 1)
    assert result['coverage_complete']
    assert result['prediction'] == 'inconclusive'
    assert result['violations'] == [] and result['review_required']
    assert result['candidate_window_count'] == int(candidate == 'violation')
    assert not windows.summarize_windows(records, 2)['coverage_complete']
    assert not windows.summarize_windows([{'status': 'failed'}], 1)['coverage_complete']


def test_adaptive_details_keep_motion_peak_and_history_never_uses_future_frames():
    plan = windows.plan_windows(1800, 30.)
    scores = [0.] * 1800
    for index in (101, 700, 1201):
        scores[index] = 100.
    enhanced = windows.add_visual_context(plan, scores, 30.)
    for original, row in zip(plan, enhanced):
        assert set(original['frame_indices']) <= set(row['detail_indices'])
        assert len(row['detail_indices']) <= 96 and len(row['history_indices']) <= 32
        assert len(row['frame_indices']) % 4 == 0
        assert all(i < row['start_frame'] for i in row['history_indices'])
        assert row['frame_indices'] == sorted(row['frame_indices'])
    assert 101 in enhanced[0]['detail_indices']
    assert enhanced[0]['history_indices'] == []
    assert enhanced[1]['history_indices']
    assert 0 in enhanced[-1]['history_indices']


def test_timestamps_select_positive_bags_without_labeling_other_windows_normal():
    plan = windows.plan_windows(1800, 30.)
    selected = windows.annotated_windows({'label': 1, 'event_times': [22.]}, plan)
    assert selected and len(selected) < len(plan)
    assert all(plan[i]['start_sec'] <= 22 < plan[i]['end_sec'] for i in selected)
    assert windows.annotated_windows({'label': 0}, plan) == list(range(len(plan)))
    with pytest.raises(ValueError, match='out-of-range'):
        windows.annotated_windows({'label': 1, 'event_times': [100.]}, plan)


def test_full_frame_motion_scan_and_selected_frame_decoding(tmp_path):
    import cv2
    import numpy as np
    path = tmp_path / 'synthetic.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64, 64))
    assert writer.isOpened()
    for index in range(45):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[20:35, index:index + 8] = 255
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    try:
        scores = windows.scan_motion(capture, 45)
        assert len(scores) == 45 and max(scores) > 0
        plan = windows.plan_windows(45, 10., seconds=2, overlap=.5, sample_fps=4., max_frames=24)
        plan = windows.add_visual_context(plan, scores, 10., detail_frames=24, history_frames=8)
        for row in plan:
            pixels, times = windows.decode_window(capture, row, 10.)
            assert pixels.shape == (len(row['frame_indices']), 3, 448, 448)
            assert times == [i / 10. for i in row['frame_indices']]
        with pytest.raises(ValueError, match='stopped at frame'):
            windows.scan_motion(capture, 46)
    finally:
        capture.release()
