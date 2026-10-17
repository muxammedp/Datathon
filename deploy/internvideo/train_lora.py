import argparse
import gc
import json
import random
import time
from pathlib import Path

from evaluate_dataset import check_dataset, file_hash, save_json
from training_protocol import binary_loss, choose_threshold, make_folds, metrics, objective

ROOT = Path(__file__).resolve().parent


def emit(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def lock_protocol(directory, protocol):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != protocol:
            raise ValueError('Training protocol changed; choose a new --output directory')
    else:
        if any(directory.iterdir()):
            raise ValueError('Output directory is not empty and has no training protocol')
        save_json(path, protocol)


def load_inputs(directory, row, device):
    from safetensors.torch import load_file
    return load_file(str(directory / (row['sha256'] + '.safetensors')), device=str(device))['embeds']


def predict(language_model, rows, indices, directory, answer_ids, device):
    import torch
    from lora_binary import binary_logits
    language_model.eval()
    scores = []
    with torch.no_grad():
        for i in indices:
            embeds = load_inputs(directory, rows[i], device)
            logits = binary_logits(language_model, embeds, answer_ids)
            value = float((logits[0, 1] - logits[0, 0]).cpu())
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite model logits')
            scores.append(value)
    return scores


def prepare_cache(model, tokenizer, rows, args, prompt):
    import torch
    from safetensors.torch import load_file, save_file
    from analyze import sample_video
    from lora_binary import frozen_inputs
    directory = args.output / 'input-cache'
    directory.mkdir(exist_ok=True)
    answer_ids = None
    for index, row in enumerate(rows):
        began = time.monotonic()
        path = directory / (row['sha256'] + '.safetensors')
        metadata_path = path.with_suffix('.json')
        if path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf8'))
            if file_hash(path) != metadata['cache_sha256']:
                raise ValueError('Cached inputs are corrupted; use a fresh output directory')
            cached = load_file(str(path))['embeds']
            if list(cached.shape) != metadata['shape'] or not torch.isfinite(cached).all():
                raise ValueError('Invalid cached embeddings')
            del cached
        else:
            pixels, times, duration = sample_video(args.dataset / row['file'], args.fps, args.max_frames)
            pixels = pixels.to(device='cuda', dtype=torch.bfloat16)
            embeds, ids = frozen_inputs(model, tokenizer, pixels, times, prompt)
            if not torch.isfinite(embeds).all():
                raise ValueError('Nonfinite visual or text embeddings')
            temporary = path.with_suffix('.tmp')
            save_file({'embeds': embeds}, str(temporary))
            temporary.replace(path)
            metadata = dict(answer_ids=ids, shape=list(embeds.shape), sampled_frames=len(times),
                            duration_sec=duration, first_time_sec=times[0], last_time_sec=times[-1],
                            max_sample_gap_sec=max(b - a for a, b in zip(times, times[1:])),
                            cache_sha256=file_hash(path))
            save_json(metadata_path, metadata)
            del pixels, embeds
        if answer_ids is not None and answer_ids != metadata['answer_ids']:
            raise ValueError('Answer tokenization is inconsistent across videos')
        answer_ids = metadata['answer_ids']
        emit(stage='prepare_inputs', completed=index + 1, total=len(rows),
             frames=metadata['sampled_frames'], elapsed_sec=round(time.monotonic() - began, 2))
    torch.cuda.empty_cache()
    return directory, answer_ids


def train_fold(language_model, rows, split, cache, answer_ids, directory, args, fold, initial, frozen_scores):
    import torch
    from safetensors.torch import save_file
    from lora_binary import adapter_state, binary_logits, restore_adapter
    restore_adapter(language_model, initial)
    torch.manual_seed(args.seed + fold)
    rng = random.Random(args.seed + fold)
    device = next(language_model.parameters()).device
    labels = [rows[i]['label'] for i in split['calibration']]
    baseline_scores = [frozen_scores[i] for i in split['calibration']]
    threshold, baseline_metrics = choose_threshold(labels, baseline_scores)
    best = dict(epoch=0, threshold=threshold, calibration_metrics=baseline_metrics,
                calibration_loss=binary_loss(labels, baseline_scores))
    best_state = initial
    parameters = [p for p in language_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    history = []
    directory.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        language_model.train()
        order = list(split['train'])
        rng.shuffle(order)
        total_loss = 0.
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        for step, index in enumerate(order):
            embeds = load_inputs(cache, rows[index], device)
            logits = binary_logits(language_model, embeds, answer_ids)
            target = torch.tensor([rows[index]['label']], device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            group_start = step // 4 * 4
            group_size = min(4, len(order) - group_start)
            (loss / group_size).backward()
            total_loss += float(loss.detach())
            if (step + 1) % 4 == 0 or step + 1 == len(order):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if float(norm) == 0:
                    raise RuntimeError('Zero adapter gradient; training is not updating the model')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                emit(stage='training', fold=fold + 1, epoch=epoch, step=step + 1, total_steps=len(order),
                     mean_loss=round(total_loss / (step + 1), 5),
                     elapsed_sec=round(time.monotonic() - started, 1),
                     peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            del embeds, logits, loss
        scores = predict(language_model, rows, split['calibration'], cache, answer_ids, device)
        threshold, calibration_metrics = choose_threshold(labels, scores)
        calibration_loss = binary_loss(labels, scores)
        record = dict(epoch=epoch, loss=total_loss / len(order), threshold=threshold,
                      calibration_metrics=calibration_metrics, calibration_loss=calibration_loss)
        history.append(record)
        if (objective(calibration_metrics), calibration_loss) < (objective(best['calibration_metrics']), best['calibration_loss']):
            best = dict(epoch=epoch, threshold=threshold, calibration_metrics=calibration_metrics,
                        calibration_loss=calibration_loss)
            best_state = adapter_state(language_model)
        save_json(directory / 'progress.json', dict(best=best, history=history))
        emit(stage='epoch_complete', fold=fold + 1, **record, selected_epoch=best['epoch'])
    restore_adapter(language_model, best_state)
    language_model.eval()
    adapter_path = directory / 'adapter.safetensors'
    temporary = adapter_path.with_suffix('.tmp')
    save_file(adapter_state(language_model), str(temporary))
    temporary.replace(adapter_path)
    scores = predict(language_model, rows, split['test'], cache, answer_ids, device)
    test_labels = [rows[i]['label'] for i in split['test']]
    baseline_threshold, _ = choose_threshold(labels, [frozen_scores[i] for i in split['calibration']])
    predictions = [dict(sha256=rows[i]['sha256'], label=rows[i]['label'], score=score,
                        threshold=best['threshold'], prediction=int(score >= best['threshold']),
                        frozen_score=frozen_scores[i], frozen_threshold=baseline_threshold)
                   for i, score in zip(split['test'], scores)]
    report = dict(fold=fold, best=best, history=history, predictions=predictions,
                  test_metrics=metrics(test_labels, scores, best['threshold']),
                  matched_frozen_metrics=metrics(test_labels, [frozen_scores[i] for i in split['test']], baseline_threshold),
                  adapter_sha256=file_hash(adapter_path),
                  **{name + '_sha256': [rows[i]['sha256'] for i in indices] for name, indices in split.items()})
    save_json(directory / 'result.json', report)
    emit(stage='fold_complete', fold=fold + 1, selected_epoch=best['epoch'], test_metrics=report['test_metrics'])
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def aggregate(reports, total):
    predictions = [row for report in reports for row in report['predictions']]
    if len({row['sha256'] for row in predictions}) != len(predictions):
        raise ValueError('Duplicate held-out prediction')
    labels = [row['label'] for row in predictions]
    trained = metrics(labels, [row['score'] - row['threshold'] for row in predictions], 0.)
    frozen = metrics(labels, [row['frozen_score'] - row['frozen_threshold'] for row in predictions], 0.)
    complete = len(predictions) == total and len(reports) == 5
    passed = (complete and trained['accuracy'] >= .75 and trained['precision'] >= .8
              and trained['recall'] >= .6 and trained['false_positive_rate'] <= .2)
    return dict(complete=complete, completed_folds=len(reports), unique_test_videos=len(predictions),
                dataset_total=total, held_out_metrics=trained, matched_frozen_metrics=frozen,
                quality_gate_passed=passed, production_approved=False,
                automatic_verdict_enabled=False, categories_evaluated=False, actor_roles_evaluated=False,
                epochs_with_selected_training=[r['best']['epoch'] for r in reports],
                threshold_policy='Minimize (2*FPR + 1-recall)/3 with recall>=0.5 using calibration videos only; '
                                 'break epoch ties by calibration log loss',
                note='Five-fold internal evaluation on a small dataset; same-scene correlations may remain. '
                     'Epoch zero means calibration selected the unchanged model. '
                     'This is not the earlier 256-frame free-generation baseline and is not a role tracker.',
                folds=reports)


def save_report(directory, reports, total):
    report = aggregate(reports, total)
    save_json(directory / 'metrics.json', report)
    save_json(directory / 'summary.json', {key: value for key, value in report.items() if key != 'folds'})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('open-dataset'))
    parser.add_argument('--output', type=Path, default=Path('results/lora-v1'))
    parser.add_argument('--max-frames', type=int, default=128)
    parser.add_argument('--fps', type=float, default=4.)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--last-layers', type=int, default=12)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--max-folds', type=int, default=5)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if (args.max_frames < 4 or args.max_frames > 256 or args.max_frames % 4 or args.fps <= 0
            or not 1 <= args.epochs <= 20 or not 0 < args.lr <= .001
            or not 1 <= args.rank <= 64 or not 1 <= args.last_layers <= 32
            or not 1 <= args.max_folds <= 5):
        parser.error('Invalid training settings')
    manifest = check_dataset(args.dataset)
    rows = manifest['videos']
    splits = make_folds(rows, args.seed)
    model_manifest = json.loads((ROOT / 'model-manifest.json').read_text(encoding='utf8'))
    protocol = dict(version='internvideo_binary_lora_v1', model_revision=model_manifest['revision'],
                    dataset_manifest_sha256=file_hash(args.dataset / 'manifest.json'),
                    code_sha256={name: file_hash(ROOT / name) for name in
                                 ('train_lora.py', 'lora_binary.py', 'training_protocol.py', 'analyze.py')},
                    max_frames=args.max_frames, fps=args.fps, epochs=args.epochs, lr=args.lr,
                    seed=args.seed, last_layers=args.last_layers, rank=args.rank,
                    adapter_format='forte_torch_lora_v1_not_peft', splits=splits)
    lock_protocol(args.output, protocol)
    emit(stage='dataset_verified', unique_videos=len(rows),
         splits=[{k: len(v) for k, v in split.items()} for split in splits],
         training_started=False)
    if args.prepare_only:
        return
    reports = []
    for fold, split in enumerate(splits[:args.max_folds]):
        path = args.output / f'fold-{fold}' / 'result.json'
        if path.exists():
            report = json.loads(path.read_text(encoding='utf8'))
            for name, indices in split.items():
                if report[name + '_sha256'] != [rows[i]['sha256'] for i in indices]:
                    raise ValueError('Saved fold does not match locked split')
            if report['adapter_sha256'] != file_hash(path.parent / 'adapter.safetensors'):
                raise ValueError('Saved adapter checksum mismatch')
            reports.append(report)
    if len(reports) == args.max_folds:
        save_report(args.output, reports, len(rows))
        emit(stage='already_complete', report=str(args.output / 'metrics.json'))
        return
    import torch
    import transformers
    from analyze import load_model, PROMPT
    from lora_binary import adapter_state, attach_adapters
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
        raise RuntimeError('Training requires a CUDA GPU with at least 40 GiB VRAM')
    save_json(args.output / 'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__,
                                                   gpu=torch.cuda.get_device_name(0)))
    model, tokenizer = load_model()
    model.requires_grad_(False)
    model.eval()
    prompt = PROMPT.split('Answer with exactly one digit:')[0] + (
        'Classify the video. Answer with exactly one digit: 0 for no clearly established violation, '
        '1 for a clearly established violation. Do not provide names or descriptions.')
    cache, answer_ids = prepare_cache(model, tokenizer, rows, args, prompt)
    language_model = model.language_model
    attach_adapters(language_model, last_layers=args.last_layers, rank=args.rank, seed=args.seed)
    language_model.model.gradient_checkpointing = True
    language_model.config.use_cache = False
    initial = adapter_state(language_model)
    emit(stage='adapter_ready', trainable_parameters=sum(p.numel() for p in language_model.parameters() if p.requires_grad),
         answer_ids=answer_ids, format=protocol['adapter_format'])
    baseline_path = args.output / 'matched-frozen-scores.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf8'))
        if [r['sha256'] for r in baseline] != [r['sha256'] for r in rows]:
            raise ValueError('Frozen scores do not match dataset')
        frozen_scores = [r['score'] for r in baseline]
    else:
        frozen_scores = []
        for index in range(len(rows)):
            frozen_scores.extend(predict(language_model, rows, [index], cache, answer_ids, 'cuda'))
            emit(stage='matched_frozen_baseline', completed=index + 1, total=len(rows))
        save_json(baseline_path, [dict(sha256=row['sha256'], score=score) for row, score in zip(rows, frozen_scores)])
    for fold, split in enumerate(splits[:args.max_folds]):
        if any(r['fold'] == fold for r in reports):
            continue
        report = train_fold(language_model, rows, split, cache, answer_ids, args.output / f'fold-{fold}',
                            args, fold, initial, frozen_scores)
        reports.append(report)
        reports.sort(key=lambda r: r['fold'])
        save_report(args.output, reports, len(rows))
    result = aggregate(reports, len(rows))
    emit(stage='finished', complete=result['complete'], metrics=result['held_out_metrics'],
         matched_frozen_metrics=result['matched_frozen_metrics'],
         quality_gate_passed=result['quality_gate_passed'], production_approved=False,
         report=str(args.output / 'metrics.json'))


if __name__ == '__main__':
    main()
