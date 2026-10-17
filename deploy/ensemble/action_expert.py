import gc
import math
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps

VERSION = 'xclip_overlapping_clips_v1'
MODEL_ID = 'microsoft/xclip-base-patch32'
FRAME_COUNT = 8
CLIP_STRIDE = 4
IMAGE_SIZE = 224
HIDDEN_SIZE = 512
FEATURE_SIZE = HIDDEN_SIZE * 4


def clip_indices(frame_count):
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError('A window must contain at least one sampled frame')
    starts = list(range(0, max(frame_count - FRAME_COUNT, 0) + 1, CLIP_STRIDE))
    if starts[-1] + FRAME_COUNT < frame_count:
        starts.append(starts[-1] + CLIP_STRIDE)
    return [[min(start + offset, frame_count - 1) for offset in range(FRAME_COUNT)]
            for start in starts]


def validate_crop(value):
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise ValueError('Counter crop must be four finite normalized coordinates')
    x1, y1, x2, y2 = map(float, value)
    if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
        raise ValueError('Counter crop must be a nonempty box inside the frame')
    return x1, y1, x2, y2


def letterbox(image, crop=None, fill=(128, 128, 128)):
    image = ImageOps.exif_transpose(image).convert('RGB')
    crop = validate_crop(crop)
    if crop is not None:
        width, height = image.size
        left, top = math.floor(crop[0] * width), math.floor(crop[1] * height)
        right, bottom = math.ceil(crop[2] * width), math.ceil(crop[3] * height)
        image = image.crop((left, top, right, bottom))
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError('Cannot preprocess an empty frame')
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    size = (max(1, min(IMAGE_SIZE, round(width * scale))),
            max(1, min(IMAGE_SIZE, round(height * scale))))
    resized = image.resize(size, Image.Resampling.BILINEAR)
    result = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), fill)
    result.paste(resized, ((IMAGE_SIZE - size[0]) // 2, (IMAGE_SIZE - size[1]) // 2))
    return result


def _frame_path(root, name):
    if not isinstance(name, str) or not name or ':' in name:
        raise ValueError('Frame paths must be relative to the media index root')
    relative = PurePosixPath(name.replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Frame path escapes the media index root')
    path = (root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Frame is missing or outside the media index root')
    return path


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric ' + name)
    return float(value)


def _mit_return_dict(module, args, kwargs):
    return args, dict(kwargs, return_dict=True)


class ActionExpert:
    def __init__(self, model_path, device='cuda'):
        import torch
        from transformers import VideoMAEImageProcessor, XCLIPModel
        model_path = Path(model_path).resolve()
        if not model_path.is_dir():
            raise ValueError('The pinned X-CLIP model directory is missing')
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Action encoder requires an available CPU or CUDA device')
        processor = VideoMAEImageProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.mean = np.asarray(processor.image_mean, dtype=np.float32)
        self.std = np.asarray(processor.image_std, dtype=np.float32)
        self.rescale = float(processor.rescale_factor)
        if (not processor.do_normalize or not processor.do_rescale
                or self.mean.shape != (3,) or self.std.shape != (3,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or not np.all((0 <= self.mean) & (self.mean <= 1)) or not np.all(self.std > 0)
                or not math.isclose(self.rescale, 1 / 255, rel_tol=1e-7)):
            raise ValueError('Unsupported pinned X-CLIP normalization settings')
        self.fill = tuple(round(float(value) * 255) for value in self.mean)
        self.model = XCLIPModel.from_pretrained(str(model_path), local_files_only=True,
                                              trust_remote_code=False)
        config = self.model.config
        vision = config.vision_config
        if (config.projection_dim != HIDDEN_SIZE or vision.num_frames != FRAME_COUNT
                or vision.hidden_size != 768 or vision.mit_hidden_size != HIDDEN_SIZE
                or vision.image_size != IMAGE_SIZE or vision.num_channels != 3 or vision.patch_size != 32):
            raise ValueError('X-CLIP architecture does not match the pinned base model')
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)
        self._mit_hook = self.model.mit.register_forward_pre_hook(_mit_return_dict, with_kwargs=True)

    def _encode(self, tensors):
        import torch
        batch = torch.stack(tensors).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model.get_video_features(pixel_values=batch)
            if output.shape != (1, HIDDEN_SIZE):
                raise ValueError('Unexpected X-CLIP encoder output shape')
            vector = output.float()[0].cpu().numpy()
        if not np.isfinite(vector).all():
            raise ValueError('Action encoder produced nonfinite features')
        return vector

    def extract(self, index, root):
        import torch
        if self.model is None:
            raise RuntimeError('Action expert has been released')
        root = Path(root).resolve()
        identity = index.get('video_sha256')
        windows = index.get('windows')
        if (not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity)
                or not isinstance(windows, list) or not windows):
            raise ValueError('Invalid video identity or empty window index')
        result = []
        for ordinal, window in enumerate(windows):
            if type(window.get('index')) is not int or window['index'] != ordinal:
                raise ValueError('Window indices must be chronological, consecutive and start at zero')
            start = _number(window.get('start_sec'), 'window start')
            end = _number(window.get('end_sec'), 'window end')
            if start < 0 or end <= start or (result and start < result[-1]['start_sec']):
                raise ValueError('Invalid chronological window bounds')
            names, values = window.get('frame_paths'), window.get('times')
            if not isinstance(names, list) or not names or not isinstance(values, list) or len(names) != len(values):
                raise ValueError('Each sampled frame needs one matching timestamp')
            times = [_number(value, 'frame timestamp') for value in values]
            if (any(time < start - 1e-7 or time > end + 1e-7 for time in times)
                    or any(b <= a for a, b in zip(times, times[1:]))):
                raise ValueError('Sampled timestamps must be strictly chronological and inside the window')
            paths = [_frame_path(root, name) for name in names]
            crop = validate_crop(window.get('counter_crop'))
            clips = clip_indices(len(paths))

            @lru_cache(maxsize=32)
            def frame_tensor(position, use_crop):
                with Image.open(paths[position]) as source:
                    image = letterbox(source, crop if use_crop else None, self.fill)
                    pixels = np.asarray(image, dtype=np.float32)
                pixels = (pixels * self.rescale - self.mean) / self.std
                return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))

            features = []
            for use_crop in (False, True):
                if use_crop and crop is None:
                    features.extend(features[:])
                    break
                vectors = np.stack([self._encode([frame_tensor(position, use_crop) for position in clip])
                                    for clip in clips])
                features.extend((vectors.mean(0), vectors.max(0)))
                frame_tensor.cache_clear()
            merged = np.concatenate(features).astype(np.float32)
            if merged.shape != (FEATURE_SIZE,) or not np.isfinite(merged).all():
                raise ValueError('Invalid pooled action feature dimensions or values')
            covered = sorted({position for clip in clips for position in clip})
            if covered != list(range(len(paths))):
                raise ValueError('Action clips failed to cover every supplied sample')
            result.append(dict(index=ordinal, start_sec=start, end_sec=end, features=merged.tolist(),
                               crop_used=crop is not None, selected_times=times,
                               clips_processed=len(clips), encoder_passes=len(clips) * (2 if crop is not None else 1),
                               sampled_frames=len(paths), covered_sample_indices=covered,
                               all_supplied_samples_covered=True,
                               clips=[dict(sample_indices=clip, times=[times[position] for position in clip],
                                           padding_frames=FRAME_COUNT - len(set(clip))) for clip in clips]))
        return dict(version=VERSION, video_sha256=identity, model_identifier=MODEL_ID,
                    feature_dimensions=FEATURE_SIZE, pooling='scene mean, scene max, crop mean, crop max',
                    temporal_frames_per_clip=FRAME_COUNT, temporal_stride=CLIP_STRIDE, windows=result,
                    all_windows_processed=True, sampling_note='Every supplied sampled frame is encoded. '
                    'This does not imply that every original source frame or short action was observed.')

    def release(self):
        import torch
        if getattr(self, '_mit_hook', None) is not None:
            self._mit_hook.remove()
            self._mit_hook = None
        self.model = None
        gc.collect()
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
