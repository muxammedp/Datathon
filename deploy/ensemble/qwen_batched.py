import math
import time

from deploy.ensemble import qwen_expert as base


VERSION = 'qwen_batched_v1'
VIEWS = ('context', 'cash')


class BatchedQwenExpert(base.QwenExpert):
    def _batch_inputs(self, image_batches, windows, prompts):
        if not image_batches or not len(image_batches) == len(windows) == len(prompts):
            raise ValueError('Each specialist needs paired images, timestamps and a prompt')
        queries, images = [], []
        for pictures, window, prompt in zip(image_batches, windows, prompts):
            if not pictures or len(pictures) != len(window['times']):
                raise ValueError('Specialist images must match their own timestamps')
            content = []
            for timestamp in window['times']:
                content.extend([dict(type='text', text=f'Sampled frame at {timestamp:.3f} seconds:'),
                                dict(type='image')])
            content.append(dict(type='text', text=prompt))
            queries.append(self.processor.apply_chat_template(
                [dict(role='user', content=content)], tokenize=False, add_generation_prompt=True))
            images.extend(pictures)
        tokenizer = self.processor.tokenizer
        previous_padding = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        try:
            values = self.processor(text=queries, images=images, padding=True, return_tensors='pt',
                                    min_pixels=64 * 64, max_pixels=base.MAX_PIXELS)
        finally:
            tokenizer.padding_side = previous_padding
        if values['image_grid_thw'].shape[0] != len(images):
            raise ValueError('Batched image grid does not match the specialist images')
        mask = values['attention_mask']
        if not bool(mask[:, -1].all()) or bool((mask[:, 1:] < mask[:, :-1]).any()):
            raise ValueError('Decoder-only specialist batches require contiguous left padding')
        device = next(self.model.parameters()).device
        return queries, {key: value.to(device) for key, value in values.items()}

    def _risk_scores(self, image_batches, windows, prompts):
        import torch
        from torch.nn import functional as F
        queries, inputs = self._batch_inputs(image_batches, windows, prompts)
        for query in queries:
            prefix = self.processor.tokenizer.encode(query, add_special_tokens=False)
            for digit, token in zip(('0', '1'), self.answer_ids):
                completed = self.processor.tokenizer.encode(query + digit, add_special_tokens=False)
                if completed != prefix + [token]:
                    raise ValueError('Binary completion changes the prompt tokenization')
        with torch.inference_mode():
            hidden = self.model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
            head = self.model.get_output_embeddings()
            ids = torch.tensor(self.answer_ids, device=head.weight.device)
            bias = head.bias[ids].float() if head.bias is not None else None
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                logits = F.linear(hidden.float(), head.weight[ids].float(), bias)
            scores = (logits[:, 1] - logits[:, 0]).tolist()
        if not all(math.isfinite(score) for score in scores):
            raise ValueError('Nonfinite specialist score')
        return scores, dict(input_tokens=inputs['attention_mask'].sum(dim=1).tolist(),
                            padded_input_tokens=inputs['input_ids'].shape[1], model_calls=1)

    def _generate_facts(self, image_batches, windows, prompts):
        import torch
        _, inputs = self._batch_inputs(image_batches, windows, prompts)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                             use_cache=True)
        output = generated[:, inputs['input_ids'].shape[1]:]
        texts = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (tuple, list)) else [eos]) if eos is not None else set()
        lengths = []
        for tokens in output.tolist():
            lengths.append(next((i + 1 for i, token in enumerate(tokens) if token in eos), len(tokens)))
        return texts, dict(input_tokens=inputs['attention_mask'].sum(dim=1).tolist(),
                           padded_input_tokens=inputs['input_ids'].shape[1],
                           generated_tokens=lengths, model_calls=1)

    def _facts_many(self, image_batches, windows, prompts):
        texts, profile = self._generate_facts(image_batches, windows, prompts)
        if len(texts) != len(windows):
            raise ValueError('Batched generation returned the wrong number of specialist answers')
        facts, statuses = [], []
        for text, window in zip(texts, windows):
            try:
                facts.append(base.validate_observations(text, window))
                statuses.append('valid')
            except (ValueError, TypeError, KeyError):
                facts.append(base.unknown_observations())
                statuses.append('invalid_json_or_schema')
        return facts, statuses, profile

    def _analyze_window(self, window, memory):
        from PIL import Image
        import torch
        self._load()
        began = time.monotonic()
        images, cropped = [], []
        try:
            for path in window['paths']:
                with Image.open(path) as source:
                    images.append(source.convert('RGB'))
            crop = window['counter_crop']
            if crop is not None:
                cropped = [picture.crop((math.floor(crop[0] * picture.width), math.floor(crop[1] * picture.height),
                                         math.ceil(crop[2] * picture.width), math.ceil(crop[3] * picture.height)))
                           for picture in images]
            pictures = [images, cropped if crop is not None else images]
            windows = [base._view_window(window, view) for view in VIEWS]
            prepared = time.monotonic()
            raw_scores, risk_profile = self._risk_scores(
                pictures, windows, [base.build_prompt(w, view, memory, 'risk') for w, view in zip(windows, VIEWS)])
            risk_finished = time.monotonic()
            try:
                raw_facts, raw_statuses, fact_profile = self._facts_many(
                    pictures, windows, [base.build_prompt(w, view, memory, 'facts') for w, view in zip(windows, VIEWS)])
            except torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as exc:
                if 'out of memory' in str(exc).lower():
                    raise
                raw_facts = [base.unknown_observations() for _ in VIEWS]
                raw_statuses = ['generation_failed'] * len(VIEWS)
                fact_profile = dict(model_calls=1, generation_completed=False)
            facts_finished = time.monotonic()
            scores, facts, statuses = dict(zip(VIEWS, raw_scores)), dict(zip(VIEWS, raw_facts)), dict(zip(VIEWS, raw_statuses))
            reasons = []
            if any(status != 'valid' for status in statuses.values()):
                reasons.append('invalid_or_incomplete_observations')
            if any(event['kind'] == 'unclassified_activity' for value in facts.values() for event in value['events']):
                reasons.append('unclassified_activity')
            if not any(track['role'] == 'cashier_candidate' for track in window['tracks']):
                reasons.append('cashier_role_unknown')
            if any(value['service_state'] == 'unknown' or value['cash_action'] == 'unknown'
                   or any(actor['visibility'] in ('occluded', 'unknown') or actor['activity'] == 'unknown'
                          for actor in value['actors']) for value in facts.values()):
                reasons.append('insufficient_observations')
            disagreement = scores['context'] * scores['cash'] < 0
            if disagreement:
                reasons.append('uncalibrated_specialist_sign_disagreement')
            evidence = []
            for view, value in facts.items():
                for event in value['events']:
                    closest = min(range(len(window['times'])), key=lambda i: abs(window['times'][i] - event['time_sec']))
                    evidence.append(dict(kind=event['kind'], event_time_sec=event['time_sec'],
                                         frame_time_sec=window['times'][closest], frame_path=window['frame_paths'][closest],
                                         view=view, crop=window['counter_crop'] if view == 'cash' else None,
                                         track_ids=event['track_ids'], verified=False))
            return dict(index=window['index'], start_sec=window['start_sec'], end_sec=window['end_sec'], status='done',
                        score=(scores['context'] + scores['cash']) / 2, context_score=scores['context'],
                        cash_score=scores['cash'], observations=facts, parse_status=statuses,
                        sampled_frames=len(images), sampled_times=window['times'], input_mode='sampled_image_sequence',
                        cash_view='configured_counter_crop' if crop is not None else 'full_scene',
                        actor_roles_verified=False, memory_verified=False, evidence_frames=evidence,
                        review_required=bool(reasons), review_reasons=reasons, view_disagreement=disagreement,
                        production_approved=False, model_passes=2, elapsed_sec=round(time.monotonic() - began, 3),
                        profile=dict(execution='batched_views_v1', view_batch_size=2,
                                     image_preparation_sec=round(prepared - began, 3),
                                     risk_sec=round(risk_finished - prepared, 3),
                                     facts_sec=round(facts_finished - risk_finished, 3),
                                     risk=risk_profile, facts=fact_profile))
        finally:
            for picture in cropped + images:
                picture.close()

    def analyze(self, index, root, previous_results=None, on_window=None):
        old_settings = dict(max_frames=self.max_frames, max_new_tokens=self.max_new_tokens, max_pixels=base.MAX_PIXELS)
        settings = dict(old_settings, view_batch_size=2, execution='batched_views_v1')
        previous = None
        if previous_results is not None:
            if previous_results.get('version') != VERSION or previous_results.get('settings') != settings:
                raise ValueError('Previous expert result belongs to a different execution or configuration')
            previous = dict(previous_results, version=base.VERSION, settings=old_settings)

        def current(payload):
            return dict(payload, version=VERSION, settings=settings)

        callback = (lambda value: on_window(current(value))) if on_window is not None else None
        return current(super().analyze(index, root, previous_results=previous, on_window=callback))
