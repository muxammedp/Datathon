import argparse
import gc
import json
import random
import time
from pathlib import Path

from evaluate_dataset import check_dataset, file_hash, save_json
from event_cache import binary_prompt, prepare, score_windows
from train_lora import aggregate, emit, lock_protocol
from training_protocol import binary_loss, choose_threshold, make_folds, objective
from video_windows import annotated_windows

ROOT = Path(__file__).resolve().parent


def calibration(rows, indices, scores):
    labels = [rows[i]['label'] for i in indices]
    values = [max(scores[i]) for i in indices]
    threshold, result = choose_threshold(labels, values)
    return dict(threshold=threshold, calibration_metrics=result, calibration_loss=binary_loss(labels, values))


def train_fold(model, rows, windows, cache, answer_ids, split, fold, initial, frozen, args):
    import torch
    from safetensors.torch import load_file, save_file
    from lora_binary import adapter_state, binary_logits, restore_adapter
    device = next(model.parameters()).device
    restore_adapter(model, initial)
    torch.manual_seed(args.seed + fold)
    rng = random.Random(args.seed + fold)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    best = dict(epoch=0, **calibration(rows, split['calibration'], frozen))
    best_state, history = initial, []
    directory = args.output / f'fold-{fold}'
    directory.mkdir(exist_ok=True)
    eligible = {i: annotated_windows(rows[i], windows[i]) for i in split['train']}
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(split['train'])
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        loss_total, began = 0., time.monotonic()
        for step, index in enumerate(order):
            choices = eligible[index]
            if rows[index]['label'] or len(choices) <= 2:
                selected = list(choices)
            else:
                hardest = max(choices, key=lambda j: frozen[index][j])
                selected = [hardest, rng.choice([j for j in choices if j != hardest])]
            window_scores = []
            for j in selected:
                path = cache / (windows[index][j]['cache_key'] + '.safetensors')
                embeds = load_file(str(path), device=str(device))['embeds']
                logits = binary_logits(model, embeds, answer_ids)
                window_scores.append(logits[0, 1] - logits[0, 0])
                del embeds, logits
            score = torch.stack(window_scores).max()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                score, torch.tensor(float(rows[index]['label']), device=device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite event training loss')
            group_size = min(4, len(order) - step // 4 * 4)
            (loss / group_size).backward()
            loss_total += float(loss.detach())
            if (step + 1) % 4 == 0 or step + 1 == len(order):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if float(norm) == 0:
                    raise RuntimeError('Zero adapter gradient')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                emit(stage='event_training', fold=fold + 1, epoch=epoch, video_step=step + 1,
                     total_videos=len(order), mean_loss=round(loss_total / (step + 1), 5),
                     elapsed_sec=round(time.monotonic() - began, 1),
                     peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            del window_scores, score, loss
        scores = score_windows(model, rows, split['calibration'], windows, cache, answer_ids, device)
        record = dict(epoch=epoch, train_loss=loss_total / len(order), **calibration(rows, split['calibration'], scores))
        history.append(record)
        if (objective(record['calibration_metrics']), record['calibration_loss']) < (
                objective(best['calibration_metrics']), best['calibration_loss']):
            best = {key: value for key, value in record.items() if key != 'train_loss'}
            best_state = adapter_state(model)
        save_json(directory / 'progress.json', dict(best=best, history=history))
        emit(stage='event_epoch_complete', fold=fold + 1, selected_epoch=best['epoch'], **record)
    restore_adapter(model, best_state)
    adapter_path = directory / 'adapter.safetensors'
    temporary = adapter_path.with_suffix('.tmp')
    save_file(adapter_state(model), str(temporary))
    temporary.replace(adapter_path)
    scores = score_windows(model, rows, split['test'], windows, cache, answer_ids, device)
    frozen_threshold = calibration(rows, split['calibration'], frozen)['threshold']
    predictions = []
    for index in split['test']:
        score = max(scores[index])
        candidates = [dict(start_sec=w['start_sec'], end_sec=w['end_sec'], score=s)
                      for w, s in zip(windows[index], scores[index]) if s >= best['threshold']]
        predictions.append(dict(sha256=rows[index]['sha256'], label=rows[index]['label'], score=score,
                                threshold=best['threshold'], prediction=int(score >= best['threshold']),
                                frozen_score=max(frozen[index]), frozen_threshold=frozen_threshold,
                                candidate_intervals=candidates, windows_processed=len(windows[index])))
    result = dict(fold=fold, best=best, history=history, predictions=predictions,
                  adapter_sha256=file_hash(adapter_path),
                  **{name + '_sha256': [rows[i]['sha256'] for i in indices] for name, indices in split.items()})
    save_json(directory / 'result.json', result)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    emit(stage='event_fold_complete', fold=fold + 1, selected_epoch=best['epoch'])
    return result


def report(directory, reports, rows):
    value = aggregate(reports, len(rows))
    lookup = {row['sha256']: row for row in rows}
    positive, localized = 0, 0
    for fold in reports:
        for prediction in fold['predictions']:
            row = lookup[prediction['sha256']]
            if row['label']:
                positive += 1
                localized += any(c['start_sec'] <= t < c['end_sec']
                                 for c in prediction['candidate_intervals'] for t in row['event_times'])
    value.update(method='Dense overlapping windows, local motion sampling, sparse visual history, timestamp-guided MIL LoRA',
                 timestamp_hit_recall=localized / positive if positive else None,
                 timestamp_metric_note='At least one candidate window contains an annotation; not action boundaries or category accuracy',
                 actor_tracking=False, cross_window_memory='sparse_visual_history',
                 long_video_accuracy_validated=False)
    save_json(directory / 'metrics.json', value)
    save_json(directory / 'summary.json', {k: v for k, v in value.items() if k != 'folds'})
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('open-dataset'))
    parser.add_argument('--output', type=Path, default=Path('results/events-v1'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--last-layers', type=int, default=12)
    parser.add_argument('--window-sec', type=float, default=16.)
    parser.add_argument('--overlap-sec', type=float, default=4.)
    parser.add_argument('--fps', type=float, default=4.)
    parser.add_argument('--detail-frames', type=int, default=96)
    parser.add_argument('--history-frames', type=int, default=32)
    parser.add_argument('--max-folds', type=int, default=5)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    from video_windows import add_visual_context, plan_windows
    try:
        plan = plan_windows(480, 30., args.window_sec, args.overlap_sec, args.fps, args.detail_frames)
        add_visual_context(plan, [0.] * 480, 30., args.detail_frames, args.history_frames)
    except ValueError as exc:
        parser.error(str(exc))
    if (not 1 <= args.epochs <= 20 or not 0 < args.lr <= .001 or not 1 <= args.rank <= 64
            or not 1 <= args.last_layers <= 32 or not 1 <= args.max_folds <= 5
            or args.detail_frames + args.history_frames > 256):
        parser.error('Invalid training settings')
    rows = check_dataset(args.dataset)['videos']
    splits = make_folds(rows, args.seed)
    files = ('train_events.py', 'event_cache.py', 'video_windows.py', 'train_lora.py',
             'lora_binary.py', 'training_protocol.py', 'analyze.py')
    settings = {k: v for k, v in vars(args).items() if k not in ('dataset', 'output', 'max_folds', 'prepare_only')}
    protocol = dict(version='internvideo_events_v1', settings=settings, splits=splits,
                    dataset_manifest_sha256=file_hash(args.dataset / 'manifest.json'),
                    code_sha256={name: file_hash(ROOT / name) for name in files},
                    model_revision=json.loads((ROOT / 'model-manifest.json').read_text())['revision'])
    lock_protocol(args.output, protocol)
    emit(stage='event_dataset_verified', unique_videos=len(rows), training_started=False)
    if args.prepare_only:
        return
    reports = []
    for fold, split in enumerate(splits[:args.max_folds]):
        path = args.output / f'fold-{fold}' / 'result.json'
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf8'))
            if saved['adapter_sha256'] != file_hash(path.parent / 'adapter.safetensors'):
                raise ValueError('Adapter checksum mismatch')
            for name, indices in split.items():
                if saved[name + '_sha256'] != [rows[i]['sha256'] for i in indices]:
                    raise ValueError('Saved split mismatch')
            reports.append(saved)
    if len(reports) == args.max_folds:
        report(args.output, reports, rows)
        emit(stage='already_complete')
        return
    import torch
    import transformers
    from analyze import load_model, PROMPT
    from lora_binary import adapter_state, attach_adapters
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
        raise RuntimeError('A CUDA GPU with at least 40 GiB VRAM is required')
    save_json(args.output / 'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__,
                                                   gpu=torch.cuda.get_device_name(0)))
    model, tokenizer = load_model()
    model.requires_grad_(False)
    model.eval()
    cache, windows, answer_ids = prepare(model, tokenizer, rows, args, binary_prompt(PROMPT))
    for row, plan in zip(rows, windows):
        annotated_windows(row, plan)
    language = model.language_model
    attach_adapters(language, args.last_layers, args.rank, args.seed)
    language.model.gradient_checkpointing = True
    language.config.use_cache = False
    initial = adapter_state(language)
    frozen = {}
    for index in range(len(rows)):
        frozen.update(score_windows(language, rows, [index], windows, cache, answer_ids, 'cuda'))
        emit(stage='matched_window_baseline', completed=index + 1, total=len(rows))
    save_json(args.output / 'matched-frozen-scores.json', frozen)
    for fold, split in enumerate(splits[:args.max_folds]):
        if any(saved['fold'] == fold for saved in reports):
            continue
        result = train_fold(language, rows, windows, cache, answer_ids, split, fold, initial, frozen, args)
        reports.append(result)
        reports.sort(key=lambda r: r['fold'])
        report(args.output, reports, rows)
    summary = report(args.output, reports, rows)
    emit(stage='finished', complete=summary['complete'], metrics=summary['held_out_metrics'],
         matched_frozen_metrics=summary['matched_frozen_metrics'],
         timestamp_hit_recall=summary['timestamp_hit_recall'], production_approved=False)


if __name__ == '__main__':
    main()
