import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models' / 'forte-action-v1'
VERSION = 'forte-r3d18-v1'
FPS = 4
CLIP_FRAMES = 16
MEAN = (.43216, .394666, .37645)
STD = (.22803, .22145, .216989)


def decode_video(path):
    from app.engine import video_info
    info = video_info(path)
    cap = cv2.VideoCapture(str(path))
    frames, times = [], []
    next_time = 0.
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / info['fps']
            index += 1
            if t + .5 / info['fps'] < next_time:
                continue
            frames.append(cv2.cvtColor(cv2.resize(frame, (112, 112)), cv2.COLOR_BGR2RGB))
            times.append(t)
            next_time += 1 / FPS
    finally:
        cap.release()
    if len(frames) < 2:
        raise ValueError('Недостаточно читаемых кадров.')
    if times[-1] < info['duration_sec'] - 1.2:
        raise ValueError('Видео повреждено: декодирование остановилось до конца записи.')
    return np.stack(frames), np.asarray(times, dtype=np.float32), info


def clip_indices(length, center):
    return np.clip(np.arange(CLIP_FRAMES) + int(round(center * FPS)) - CLIP_FRAMES // 2, 0, length - 1)


def prepare_clips(frames, centers, device='cuda'):
    import torch
    arrays = np.stack([frames[clip_indices(len(frames), t)] for t in centers])
    tensor = torch.from_numpy(arrays).to(device).permute(0, 4, 1, 2, 3).float() / 255
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1, 1)
    return (tensor - mean) / std


def make_model(pretrained=False):
    import torch
    from torchvision.models.video import r3d_18
    model = r3d_18(weights=None)
    if pretrained:
        model.load_state_dict(torch.load(ROOT / 'models' / 'r3d_18-b3b3357e.pth', map_location='cpu', weights_only=True))
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def window_centers(times):
    end = float(times[-1])
    if end <= 4:
        return np.array([end / 2], dtype=np.float32)
    return np.unique(np.r_[np.arange(2., end - 1.99, 2.), end - 2.]).astype(np.float32)


def predict_windows(model, frames, times, batch_size=4):
    import torch
    centers = window_centers(times)
    output = []
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for start in range(0, len(centers), batch_size):
            x = prepare_clips(frames, centers[start:start + batch_size])
            output.extend(torch.sigmoid(model(x).flatten()).float().cpu().numpy())
    return centers, np.asarray(output, dtype=np.float32)


class ActionAnalyzer:
    def __init__(self):
        self.model = None
        self.metadata = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from safetensors.torch import load_file
        metadata = json.loads((MODEL_DIR / 'metadata.json').read_text(encoding='utf8'))
        if metadata['version'] != VERSION:
            raise ValueError('Несовместимая версия обученной модели.')
        if not torch.cuda.is_available():
            raise RuntimeError('Не найдена CUDA.')
        torch.set_num_threads(4)
        model = make_model()
        model.load_state_dict(load_file(str(MODEL_DIR / 'model.safetensors')))
        self.model = model.cuda().eval()
        self.metadata = metadata

    def analyze(self, path, progress=lambda *args: None):
        from app.engine import evidence
        began = time.monotonic()
        self.load()
        progress(10, 'Чтение последовательности кадров')
        frames, times, info = decode_video(path)
        progress(35, 'Проверка движения обученной видеомоделью')
        centers, scores = predict_windows(self.model, frames, times)
        threshold = float(self.metadata['threshold'])
        selected = []
        for i in np.argsort(-scores):
            if scores[i] < threshold:
                continue
            if any(abs(float(centers[i]) - t) < 6 for t in selected):
                continue
            selected.append(float(centers[i]))
            if len(selected) == 3:
                break
        progress(85, 'Подготовка фрагментов для проверки')
        events = []
        for timestamp in sorted(selected):
            actual = float(times[np.argmin(np.abs(times - timestamp))])
            events.append({
                'type': 'Фрагмент для просмотра', 'rule_id': None, 'timecode_sec': round(actual, 3),
                'description': 'Экспериментальная модель выделила этот фрагмент. '
                               'Это не обнаруженное нарушение: действие и роль человека не подтверждены.',
                'evidence_frames': evidence(path, actual, info['duration_sec']),
                'rule': None, 'review_status': 'required',
                'limitation': 'В исходной разметке нет типов нарушений и ролей людей. '
                              'Нельзя утверждать уход кассира или отсутствие пересчёта по этой оценке.',
            })
        return {
            'verdict': 'inconclusive', 'violations': [], 'review_segments': events, 'video': info,
            'review_required': True, 'category_status': 'not_trained',
            'analysis': {'model': VERSION, 'mode': 'local', 'elapsed_sec': round(time.monotonic() - began, 2),
                         'sampled_frames': len(frames), 'rules_checked': 0,
                         'windows_checked': len(centers), 'format_retries': 0,
                         'automatic_verdict_enabled': False,
                         'validation_status': 'failed',
                         'notice': 'Автоматический вердикт отключён: модель не прошла проверку качества. '
                                   'Выделенные фрагменты не являются доказательством нарушения. '
                                   'Просмотрите запись вручную; отсутствие отметок также ничего не подтверждает.'},
        }
