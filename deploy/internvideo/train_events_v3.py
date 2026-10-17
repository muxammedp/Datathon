import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from evaluate_dataset import check_dataset, file_hash, save_json
from event_diagnostics import audit_reports, load_verified_cache
from event_learning import binary_head_logits, negative_window_loss, positive_bag_coefficients, positive_bag_loss
from lora_binary import adapter_state, attach_adapters, restore_adapter
from train_events import calibration, report as base_report
from train_lora import emit, lock_protocol
from training_protocol import make_folds, metrics, objective
from video_windows import annotated_windows

ROOT = Path(__file__).resolve().parent


def binary_logits(model, inputs, answer_ids):
    inputs = inputs.detach()
    if model.training:
        inputs.requires_grad_(True)
    hidden = model.model(inputs_embeds=inputs, use_cache=False, output_attentions=False,
                         output_hidden_states=False, return_dict=True).last_hidden_state[:, -1, :]
    return binary_head_logits(hidden, model.get_output_embeddings(), answer_ids)


def window_score(model, path, answer_ids):
    device = next(model.parameters()).device
    inputs = load_file(str(path), device=str(device))['embeds']
    logits = binary_logits(model, inputs, answer_ids)
    score = (logits[0, 1] - logits[0, 0]).reshape(())
    if not torch.isfinite(score):
        raise ValueError('Nonfinite window score')
    return score


def window_paths(cache, windows):
    return [cache / (window['cache_key'] + '.safetensors') for window in windows]


def score_windows(model, indices, windows, cache, answer_ids):
    model.eval()
    with torch.no_grad():
        return {i: [float(window_score(model, path, answer_ids)) for path in window_paths(cache, windows[i])]
                for i in indices}


def rng_state(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def backward_video(model, paths, label, answer_ids, divisor, temperature):
    if not paths or label not in (0, 1) or divisor <= 0:
        raise ValueError('Invalid video training batch')
    if not label:
        total = 0.
        for path in paths:
            loss = negative_window_loss(window_score(model, path, answer_ids), len(paths))
            total += float(loss.detach())
            (loss / divisor).backward()
        return total
    device = next(model.parameters()).device
    states, values = [], []
    checkpointing = getattr(model.model, 'gradient_checkpointing', None)
    if checkpointing is not None:
        model.model.gradient_checkpointing = False
    try:
        with torch.no_grad():
            for path in paths:
                states.append(rng_state(device))
                values.append(window_score(model, path, answer_ids))
            scores = torch.stack(values)
            loss = float(positive_bag_loss(scores, temperature))
            coefficients = positive_bag_coefficients(scores, temperature) / divisor
    finally:
        if checkpointing is not None:
            model.model.gradient_checkpointing = checkpointing
    after_first_pass = rng_state(device)
    try:
        for path, state, reference, coefficient in zip(paths, states, scores, coefficients):
            restore_rng(state, device)
            score = window_score(model, path, answer_ids)
            if not torch.allclose(score.detach(), reference, atol=1e-5, rtol=1e-5):
                raise RuntimeError('Dropout replay changed the positive window score; refusing inexact gradients')
            (score * coefficient).backward()
    finally:
        restore_rng(after_first_pass, device)
    return loss


def traces(windows, values):
    return [dict(index=w['index'], start_sec=w['start_sec'], end_sec=w['end_sec'], score=s)
            for w, s in zip(windows, values)]


def train_fold(model, rows, windows, cache, answer_ids, split, fold, initial, frozen, args):
    restore_adapter(model, initial)
    torch.manual_seed(args.seed + fold)
    rng = random.Random(args.seed + fold)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    best = dict(epoch=0, **calibration(rows, split['calibration'], frozen))
    best_state, history, stale = initial, [], 0
    directory = args.output / f'fold-{fold}'
    directory.mkdir(exist_ok=True)
    eligible = {i: annotated_windows(rows[i], windows[i]) for i in split['train']}
    normal_windows = sum(len(windows[i]) for i in split['train'] if not rows[i]['label'])
    emit(stage='v3_fold_start', fold=fold + 1, normal_training_windows=normal_windows,
         baseline_calibration=best['calibration_metrics'])
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(split['train'])
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total_loss, began, visited = 0., time.monotonic(), 0
        for step, index in enumerate(order):
            choices = eligible[index]
            paths = window_paths(cache, [windows[index][j] for j in choices])
            divisor = min(4, len(order) - step // 4 * 4)
            total_loss += backward_video(model, paths, rows[index]['label'], answer_ids, divisor, args.temperature)
            visited += len(paths)
            if (step + 1) % 4 == 0 or step + 1 == len(order):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if float(norm) == 0:
                    raise RuntimeError('Zero adapter gradient')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                emit(stage='v3_training', fold=fold + 1, epoch=epoch, video_step=step + 1,
                     total_videos=len(order), supervised_windows=visited,
                     mean_loss=round(total_loss / (step + 1), 5), elapsed_sec=round(time.monotonic() - began, 1))
        cal_scores = score_windows(model, split['calibration'], windows, cache, answer_ids)
        record = dict(epoch=epoch, train_loss=total_loss / len(order),
                      supervised_windows=visited, **calibration(rows, split['calibration'], cal_scores))
        train_scores = score_windows(model, split['train'], windows, cache, answer_ids)
        record['train_metrics_at_calibration_threshold'] = metrics(
            [rows[i]['label'] for i in split['train']], [max(train_scores[i]) for i in split['train']],
            record['threshold'])
        record['calibration_window_traces'] = [dict(sha256=rows[i]['sha256'], label=rows[i]['label'],
                                                   windows=traces(windows[i], cal_scores[i]))
                                              for i in split['calibration']]
        history.append(record)
        if (objective(record['calibration_metrics']), record['calibration_loss']) < (
                objective(best['calibration_metrics']), best['calibration_loss']):
            best = {key: record[key] for key in ('epoch', 'threshold', 'calibration_metrics', 'calibration_loss')}
            best_state, stale = adapter_state(model), 0
        else:
            stale += 1
        save_json(directory / 'progress.json', dict(best=best, history=history))
        emit(stage='v3_epoch_complete', fold=fold + 1, selected_epoch=best['epoch'],
             **{k: v for k, v in record.items() if k != 'calibration_window_traces'})
        if epoch >= args.min_epochs and stale >= args.patience:
            emit(stage='v3_early_stop', fold=fold + 1, epoch=epoch, selected_epoch=best['epoch'])
            break
    restore_adapter(model, best_state)
    adapter_path = directory / 'adapter.safetensors'
    temporary = adapter_path.with_suffix('.tmp')
    save_file(adapter_state(model), str(temporary))
    temporary.replace(adapter_path)
    scores = score_windows(model, split['test'], windows, cache, answer_ids)
    frozen_threshold = calibration(rows, split['calibration'], frozen)['threshold']
    predictions = []
    for index in split['test']:
        score = max(scores[index])
        candidates = [dict(start_sec=w['start_sec'], end_sec=w['end_sec'], score=s)
                      for w, s in zip(windows[index], scores[index]) if s >= best['threshold']]
        predictions.append(dict(sha256=rows[index]['sha256'], label=rows[index]['label'], score=score,
                                threshold=best['threshold'], prediction=int(score >= best['threshold']),
                                frozen_score=max(frozen[index]), frozen_threshold=frozen_threshold,
                                candidate_intervals=candidates, windows_processed=len(windows[index]),
                                window_scores=scores[index], window_traces=traces(windows[index], scores[index]),
                                frozen_window_scores=frozen[index]))
    result = dict(fold=fold, best=best, history=history, predictions=predictions,
                  adapter_sha256=file_hash(adapter_path),
                  **{name + '_sha256': [rows[i]['sha256'] for i in indices] for name, indices in split.items()})
    save_json(directory / 'result.json', result)
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    emit(stage='v3_fold_complete', fold=fold + 1, selected_epoch=best['epoch'])
    return result


def report(directory, reports, rows, seed):
    audit = audit_reports(rows, reports, seed=seed)
    value = base_report(directory, reports, rows)
    value.update(method='All normal windows + smooth positive MIL with RNG replay, FP32 binary head',
                 independent_new_dataset=False, automatic_verdict_enabled=False,
                 training_note='All normal windows are supervised. Positive bags use annotated intervals only. '
                               'Decision remains max over all windows; window-count bias is not eliminated. '
                               'This dataset has been reused during development; new-data validation is still required.')
    save_json(directory / 'metrics.json', value)
    save_json(directory / 'summary.json', {k: v for k, v in value.items() if k != 'folds'})
    save_json(directory / 'error-audit.json', audit)
    return value


def main():
    parser = argparse.ArgumentParser(description='Event learning v3; reuses verified frozen inputs from events-v2')
    parser.add_argument('--dataset', type=Path, default=Path('open-dataset'))
    parser.add_argument('--cache-run', type=Path, default=Path('results/events-v2'))
    parser.add_argument('--output', type=Path, default=Path('results/events-v3'))
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--min-epochs', type=int, default=3)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--last-layers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-folds', type=int, default=5)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if (not 1 <= args.epochs <= 20 or not 1 <= args.min_epochs <= args.epochs or not 1 <= args.patience <= 20
            or not math.isfinite(args.lr) or not 0 < args.lr <= .001
            or not math.isfinite(args.temperature) or not .1 <= args.temperature <= 10
            or not 1 <= args.rank <= 64 or not 1 <= args.last_layers <= 32 or not 1 <= args.max_folds <= 5):
        parser.error('Invalid training settings')
    rows = check_dataset(args.dataset)['videos']
    cache, windows, answer_ids, provenance = load_verified_cache(args.cache_run, args.dataset, rows, ROOT)
    splits = make_folds(rows, args.seed)
    files = ('train_events_v3.py', 'event_learning.py', 'event_diagnostics.py', 'train_events.py',
             'train_lora.py', 'training_protocol.py', 'video_windows.py', 'lora_binary.py', 'analyze.py')
    settings = {k: v for k, v in vars(args).items()
                if k not in ('dataset', 'cache_run', 'output', 'check_only', 'max_folds')}
    protocol = dict(version='internvideo_events_v3', settings=settings, splits=splits, input_cache=provenance,
                    dataset_manifest_sha256=file_hash(args.dataset / 'manifest.json'),
                    code_sha256={name: file_hash(ROOT / name) for name in files},
                    model_revision=json.loads((ROOT / 'model-manifest.json').read_text())['revision'])
    emit(stage='v3_cache_verified', videos=len(rows), windows=sum(map(len, windows)),
         reextracting_video=False, training_started=False)
    if args.check_only:
        return
    if args.output.resolve() == args.cache_run.resolve():
        raise ValueError('Use a separate output directory; the source run is read-only')
    lock_protocol(args.output, protocol)
    reports = []
    for fold in range(args.max_folds):
        path = args.output / f'fold-{fold}' / 'result.json'
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf8'))
            if saved['adapter_sha256'] != file_hash(path.parent / 'adapter.safetensors'):
                raise ValueError('Saved adapter checksum mismatch')
            reports.append(saved)
    if reports:
        audit_reports(rows, reports, seed=args.seed)
    if len(reports) == args.max_folds:
        report(args.output, reports, rows, args.seed)
        emit(stage='already_complete')
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
        raise RuntimeError('A CUDA GPU with at least 40 GiB VRAM is required')
    from analyze import load_model
    import transformers
    save_json(args.output / 'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__,
                                                   gpu=torch.cuda.get_device_name(0)))
    full_model, tokenizer = load_model()
    full_model.requires_grad_(False)
    model = full_model.language_model
    attach_adapters(model, args.last_layers, args.rank, args.seed)
    model.model.gradient_checkpointing = True
    model.config.use_cache = False
    initial = adapter_state(model)
    frozen_path = args.output / 'matched-frozen-scores.json'
    frozen = {}
    for index in range(len(rows)):
        frozen.update(score_windows(model, [index], windows, cache, answer_ids))
        emit(stage='v3_matched_baseline', completed=index + 1, total=len(rows))
    save_json(frozen_path, frozen)
    for fold, split in enumerate(splits[:args.max_folds]):
        if any(saved['fold'] == fold for saved in reports):
            continue
        reports.append(train_fold(model, rows, windows, cache, answer_ids, split, fold, initial, frozen, args))
        reports.sort(key=lambda r: r['fold'])
        value = report(args.output, reports, rows, args.seed)
    emit(stage='finished', complete=value['complete'], metrics=value['held_out_metrics'],
         matched_frozen_metrics=value['matched_frozen_metrics'], production_approved=False)


if __name__ == '__main__':
    main()
