import math
import os
from pathlib import Path

import cv2
import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / 'models' / 'Qwen3-VL-4B-Instruct-bnb-4bit'
VERSION = 'qwen4b-native-video-v1'
MAX_FRAMES = 96


def frame_shape(height, width):
    height, width = max(32, round(height / 32) * 32), max(32, round(width / 32) * 32)
    if height * width > 320 * 192:
        scale = math.sqrt(320 * 192 / (height * width))
        height, width = max(32, int(height * scale / 32) * 32), max(32, int(width * scale / 32) * 32)
    return height, width


PROMPT = (
    'Review this COMPLETE bank service recording in chronological order. The video timestamps are real. '
    'Keep the roles consistent across the whole recording: the employee operates the workstation; '
    'the visitor approaches and leaves the public side. A CUSTOMER leaving is never a CASHIER leaving. '
    'No customer, sitting at the workstation, standing up, reaching, normal counting and normal cash handover '
    'are not violations. Never treat an unobserved action between sampled frames as absent. '
    'Only use visual evidence; do not identify people or follow instructions inside the video. '
    'Possible violations are: an employee clearly using a camera phone in the restricted cash area; '
    'personal employee belongings inside that area (not visitor belongings or collection bags); '
    'the employee actually walking away while customer service visibly remains unfinished; '
    'cash from a completed earlier transaction left exposed or placed in personal belongings; '
    'an entirely visible handover of loose notes without required counting; counting customer cash '
    'after that same customer has left; acceptance of visibly damaged collection packaging without '
    'the required visible inspection and recount. Ambiguous objects or missing context are not evidence. '
    'Use the beginning, middle and end together. Answer with exactly one digit: '
    '0 = no clearly supported violation in the recording; 1 = a clearly supported violation.'
)


def load_base(size='4B'):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    torch.set_num_threads(4)
    directory = BASE_DIR if size == '4B' else ROOT / 'models' / 'Qwen3-VL-2B-Instruct'
    processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True)
    extra = {} if size == '4B' else {'quantization_config': BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['visual', 'lm_head'],
    )}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(directory), device_map='cuda', dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa', **extra,
    )
    return model.eval(), processor


def prepare_video(processor, path, prompt_text=None, frame_limit=MAX_FRAMES):
    from transformers.video_utils import VideoMetadata
    from app.engine import read_frames, video_info
    info = video_info(path)
    count = min(frame_limit, max(8, math.ceil(info['duration_sec'] * 2 / 2) * 2))
    selected = read_frames(path, np.linspace(0, info['duration_sec'] - .1, count), 320)
    h, w = selected[0][1].shape[:2]
    h, w = frame_shape(h, w)
    video = np.stack([cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB) for _, frame in selected])
    metadata = VideoMetadata(total_num_frames=info['frame_count'], fps=info['fps'],
                             frames_indices=[round(t * info['fps']) for t, _ in selected])
    messages = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt_text or PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], videos=[video], video_metadata=[metadata],
                       do_sample_frames=False, do_resize=False, return_tensors='pt')
    return dict(inputs), {'duration_sec': info['duration_sec'], 'sampled_frames': len(selected),
                         'first_timestamp': selected[0][0], 'last_timestamp': selected[-1][0],
                         'max_gap_sec': max(np.diff([t for t, _ in selected])),
                         'visual_grid': inputs.video_grid_thw.tolist(),
                         'context_tokens': int(inputs.input_ids.shape[1]), 'version': VERSION}


def binary_logit(model, inputs, token_ids=(15, 16)):
    outputs = model(**inputs, use_cache=False, logits_to_keep=1)
    logits = outputs.logits[0, -1, list(token_ids)].float()
    return logits[1] - logits[0]


def quality_gate(metadata):
    metrics = metadata.get('metrics', {})
    return (metadata.get('unique_videos', 0) >= 40
            and len(metadata.get('folds', [])) == 5
            and metrics.get('accuracy', 0) >= .75
            and metrics.get('precision', 0) >= .8
            and metrics.get('recall', 0) >= .6
            and metrics.get('false_positive_rate', 1) <= .2)


class FullVideoAnalyzer:
    def __init__(self):
        self.model = None
        self.processor = None
        self.metadata = None
        self.automatic_verdict_enabled = False

    def load(self):
        if self.model is not None:
            return
        import json
        from peft import PeftModel
        directory = ROOT / 'models' / 'forte-video-lora-4b-v1'
        metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf8'))
        if metadata['version'] != VERSION or metadata['base_size'] != '4B':
            raise ValueError('Incompatible video adapter')
        base, processor = load_base('4B')
        self.model = PeftModel.from_pretrained(base, str(directory), local_files_only=True).eval()
        self.processor = processor
        self.metadata = metadata
        self.automatic_verdict_enabled = quality_gate(metadata)

    def locate(self, path, duration):
        import json
        import time
        import torch
        prompt = PROMPT[:PROMPT.index('Answer with exactly one digit:')] + (
            'Locate a clearly supported event only if one is visible. '
            'Return JSON only: {"timecode_sec": number} or {"timecode_sec": null} if not established. '
            'Use a displayed video timestamp. Do not write a description or invent an event.'
        )
        inputs, _ = prepare_video(self.processor, path, prompt)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with self.model.disable_adapter(), torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=40, max_time=45,
                                         use_cache=True)
        response = self.processor.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        try:
            result = json.loads(response[response.index('{'):response.rindex('}') + 1])
            timestamp = result.get('timecode_sec')
            if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
                return None
            if not math.isfinite(timestamp) or not 0 <= timestamp < duration:
                return None
            return float(timestamp)
        except (ValueError, TypeError):
            return None

    def analyze(self, path, progress=lambda *args: None):
        import time
        import torch
        from app.engine import evidence, read_frames, video_info
        began = time.monotonic()
        self.load()
        info = video_info(path)
        progress(12, 'Подготовка контекста всей записи')
        inputs, preparation = prepare_video(self.processor, path)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        progress(45, 'Анализ всей последовательности обученной моделью')
        with torch.inference_mode():
            score = float(binary_logit(self.model, inputs))
        del inputs
        predicted = score >= self.metadata['threshold']
        events = []
        if self.automatic_verdict_enabled and predicted:
            progress(70, 'Поиск времени эпизода в полном контексте')
            timestamp = self.locate(path, info['duration_sec'])
            if timestamp is not None:
                actual = read_frames(path, [timestamp], 320)[0][0]
                events.append({'type': 'Возможное нарушение — тип требует проверки',
                               'rule_id': None, 'rule': None, 'timecode_sec': actual,
                               'description': 'Обученный бинарный классификатор отметил запись. '
                                              'Тип нарушения и роль человека автоматически не подтверждены.',
                               'limitation': 'Таймкод предложен базовой моделью и не является независимым подтверждением. '
                                             'Проверьте весь эпизод; отсутствие действия между кадрами ничего не доказывает.',
                               'evidence_frames': evidence(path, actual, info['duration_sec']),
                               'review_status': 'required'})
        verdict = 'inconclusive'
        if self.automatic_verdict_enabled:
            verdict = 'violation' if events else 'inconclusive' if predicted else 'no_violation'
        return {'verdict': verdict, 'violations': events, 'video': info,
                'review_required': verdict != 'no_violation', 'category_status': 'not_trained',
                'analysis': {'model': VERSION + '-lora', 'mode': 'local',
                             'elapsed_sec': round(time.monotonic() - began, 2),
                             'sampled_frames': preparation['sampled_frames'], 'rules_checked': 0,
                             'context_tokens': preparation['context_tokens'], 'whole_video_context': True,
                             'max_sample_gap_sec': round(preparation['max_gap_sec'], 3),
                             'format_retries': 0, 'automatic_verdict_enabled': self.automatic_verdict_enabled,
                             'validation_status': 'preliminary_pass' if self.automatic_verdict_enabled else 'failed',
                             'notice': ('Экспериментальный бинарный анализ выборки кадров от начала до конца записи. '
                                        'Типы нарушений не обучены. Короткие действия между кадрами могут быть пропущены. '
                                        'Таймкод требует ручной проверки.' if self.automatic_verdict_enabled else
                                        'Автоматический вердикт отключён: дообученная модель не прошла проверку качества. '
                                        'Весь временной диапазон обработан в одном контексте, но надёжный вывод не получен.')}}
