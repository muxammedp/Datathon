import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.full_video import VERSION, PROMPT, load_base, prepare_video, binary_logit
from scripts.train_temporal import dataset_items, metrics

CACHE = None
TRAINING_PATHS = None


def choose_threshold(y, scores):
    choices = []
    for threshold in np.unique(np.r_[0., scores]):
        m = metrics(y, scores >= threshold)
        if m['recall'] < .5:
            continue
        choices.append(((2 * m['false_positive_rate'] + 1 - m['recall']) / 3, -m['recall'], float(threshold)))
    return min(choices)[2]


def predict(model, paths, indices):
    model.eval()
    output = []
    with torch.inference_mode():
        for i in indices:
            if CACHE:
                CACHE.select(paths[i].stem)
            inputs = {k: v.cuda() for k, v in load_file(str(paths[i])).items()}
            output.append(float(binary_logit(model, inputs)))
            del inputs
    return np.asarray(output)


def attach_adapter(model, seed):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    depth = model.config.text_config.num_hidden_layers
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                        target_modules=['q_proj', 'v_proj'], bias='none', task_type='CAUSAL_LM',
                        layers_to_transform=list(range(depth - 12, depth)), layers_pattern='layers')
    return get_peft_model(model, config)


def train(model, paths, labels, train_indices, calibration, directory, seed, epochs=2, threshold=None):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=.01)
    best, best_weights, history = None, None, []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train_indices)
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        began = time.monotonic()
        for step, index in enumerate(order):
            path = TRAINING_PATHS[index] if TRAINING_PATHS else paths[index]
            if CACHE:
                CACHE.select(path.stem)
            inputs = {k: v.cuda() for k, v in load_file(str(path)).items()}
            logit = binary_logit(model, inputs)
            target = torch.tensor(float(labels[index]), device='cuda')
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
            (loss / 4).backward()
            losses.append(float(loss.detach()))
            if (step + 1) % 4 == 0 or step + 1 == len(order):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            del inputs, logit, loss
            if (step + 1) % 4 == 0:
                print(json.dumps({'run': directory.name, 'epoch': epoch, 'step': step + 1,
                                  'total_steps': len(order), 'mean_loss': float(np.mean(losses)),
                                  'elapsed_sec': round(time.monotonic() - began, 1),
                                  'peak_vram_mb': round(torch.cuda.max_memory_allocated() / 1048576)}), flush=True)
        record = {'epoch': epoch, 'loss': float(np.mean(losses))}
        if len(calibration):
            scores = predict(model, paths, calibration)
            threshold = choose_threshold(labels[calibration], scores)
            m = metrics(labels[calibration], scores >= threshold)
            objective = (2 * m['false_positive_rate'] + 1 - m['recall']) / 3
            record.update(calibration_metrics=m, threshold=threshold)
            if best is None or objective < best['objective']:
                best = {'epoch': epoch, 'threshold': threshold, 'objective': objective}
                best_weights = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        history.append(record)
        print(json.dumps({'epoch_finished': directory.name, **record}), flush=True)
    if best_weights is not None:
        set_peft_model_state_dict(model, best_weights)
    else:
        best = {'epoch': epochs, 'threshold': threshold}
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    (directory / 'training.json').write_text(json.dumps({'best': best, 'history': history}, indent=2), encoding='utf8')
    del optimizer, best_weights
    return best


def main():
    global CACHE, TRAINING_PATHS
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--base-size', choices=['2B', '4B'], default='4B')
    parser.add_argument('--fold-limit', type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()
    directory = ROOT / 'data' / 'full-video-training'
    directory.mkdir(parents=True, exist_ok=True)
    items = dataset_items(args.dataset)
    from transformers import AutoProcessor
    from app.full_video import BASE_DIR
    processor = AutoProcessor.from_pretrained(str(BASE_DIR), local_files_only=True)
    assert processor.tokenizer.encode('0', add_special_tokens=False) == [15]
    assert processor.tokenizer.encode('1', add_special_tokens=False) == [16]
    paths, preparation = [], []
    for index, item in enumerate(items):
        path = directory / (item['sha256'] + '.safetensors')
        metadata_path = path.with_suffix('.json')
        if not path.exists():
            inputs, metadata = prepare_video(processor, item['path'])
            save_file({k: v.contiguous() for k, v in inputs.items()}, str(path))
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf8')
        metadata = json.loads(metadata_path.read_text(encoding='utf8'))
        if metadata['version'] != VERSION:
            raise ValueError('Prepared video cache has incompatible version')
        preparation.append(dict(name=item['name'], sha256=item['sha256'], **metadata))
        paths.append(path)
        print(json.dumps({'prepared': index + 1, 'total': len(items), 'frames': metadata['sampled_frames'],
                          'context_tokens': metadata['context_tokens']}), flush=True)
    (directory / 'preparation.json').write_text(json.dumps(preparation, ensure_ascii=False, indent=2), encoding='utf8')
    if args.prepare_only:
        return
    train_inputs_dir = directory / 'train64'
    train_inputs_dir.mkdir(exist_ok=True)
    TRAINING_PATHS = []
    for index, item in enumerate(items):
        if preparation[index]['sampled_frames'] <= 64:
            TRAINING_PATHS.append(paths[index])
            continue
        key = hashlib.sha256((item['sha256'] + VERSION + '-train64').encode('ascii')).hexdigest()
        path = train_inputs_dir / (key + '.safetensors')
        if not path.exists():
            inputs, metadata = prepare_video(processor, item['path'], frame_limit=64)
            save_file({k: v.contiguous() for k, v in inputs.items()}, str(path))
            path.with_suffix('.json').write_text(json.dumps(dict(source_sha256=item['sha256'], **metadata), indent=2), encoding='utf8')
        TRAINING_PATHS.append(path)
        print(json.dumps({'prepared_train64': index + 1}), flush=True)
    y = np.array([item['label'] for item in items])
    indices = np.arange(len(items))
    directory = ROOT / 'data' / f'full-video-lora-{args.base_size.lower()}'
    directory.mkdir(exist_ok=True)
    protocol = {'version': VERSION, 'base_size': args.base_size, 'epochs': args.epochs,
                'train_frame_limit': 64, 'evaluation_frame_limit': 96, 'lora_layers': 12,
                'lora_rank': 8, 'learning_rate': 5e-5, 'outer_seed': 2026,
                'prompt_sha256': hashlib.sha256(PROMPT.encode('utf8')).hexdigest(),
                'dataset_sha256': [item['sha256'] for item in items], 'labels': y.tolist()}
    protocol_path = directory / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf8')) != protocol:
            raise ValueError('Experiment protocol changed. Preserve this experiment and use a fresh results directory.')
    elif list(directory.glob('fold-*/training.json')):
        raise ValueError('Cannot resume unversioned fold checkpoints. Preserve them and use a fresh results directory.')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf8')
    base, _ = load_base(args.base_size)
    from scripts.video_feature_cache import FrozenVideoCache
    CACHE = FrozenVideoCache(base, directory / 'frozen-vision')
    baseline_path = directory / 'baseline.json'
    if not baseline_path.exists():
        baseline = []
        for i in indices:
            began = time.monotonic()
            s = predict(base, paths, [i])[0]
            baseline.append({'name': items[i]['name'], 'sha256': items[i]['sha256'], 'label': int(y[i]),
                             'score': float(s), 'predicted': int(s >= 0)})
            print(json.dumps({'baseline': int(i) + 1, 'predicted': int(s >= 0),
                              'label': int(y[i]), 'elapsed_sec': round(time.monotonic() - began, 2)}), flush=True)
        baseline_path.write_text(json.dumps({'records': baseline, 'metrics': metrics(y, [r['predicted'] for r in baseline])}, ensure_ascii=False, indent=2), encoding='utf8')
    if args.baseline_only:
        return
    all_paths = list(dict.fromkeys(paths + TRAINING_PATHS))
    for i, path in enumerate(all_paths):
        feature_path = CACHE.directory / path.name
        if not feature_path.exists():
            prepared = load_file(str(path))
            CACHE.select(path.stem)
            CACHE.get(prepared['pixel_values_videos'].cuda(), prepared['video_grid_thw'].cuda())
            print(json.dumps({'cached_frozen_video': i + 1, 'total': len(all_paths)}), flush=True)
    with torch.inference_mode():
        CACHE.select(paths[0].stem)
        prepared = {k: v.cuda() for k, v in load_file(str(paths[0])).items()}
        cached_score = float(binary_logit(base, prepared))
        wrapper = CACHE.core.get_video_features
        CACHE.core.get_video_features = CACHE.original
        try:
            reference_score = float(binary_logit(base, prepared))
        finally:
            CACHE.core.get_video_features = wrapper
        del prepared
    cache_delta = abs(cached_score - reference_score)
    assert cache_delta < 1e-4, cache_delta
    (directory / 'cache_verification.json').write_text(json.dumps({'binary_logit_delta': cache_delta,
        'features_from_frozen_pretrained_encoder_only': True}, indent=2), encoding='utf8')
    offloaded = CACHE.offload_visual([path.stem for path in all_paths])
    print(json.dumps({'frozen_vision_cached_and_offloaded': offloaded}), flush=True)
    model = attach_adapter(base, 6100)
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    print(json.dumps({'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'vram_mb': round(torch.cuda.memory_allocated() / 1048576)}), flush=True)
    records, folds = [], []
    for fold, (outer_train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(indices, y)):
        if fold >= args.fold_limit:
            break
        tr, calibration = train_test_split(outer_train, test_size=.25, stratify=y[outer_train], random_state=9100 + fold)
        set_peft_model_state_dict(model, initial)
        fold_dir = directory / f'fold-{fold}'
        if (fold_dir / 'training.json').exists():
            set_peft_model_state_dict(model, load_file(str(fold_dir / 'adapter_model.safetensors')))
            best = json.loads((fold_dir / 'training.json').read_text())['best']
        else:
            best = train(model, paths, y, tr, calibration, fold_dir, 6200 + fold, args.epochs)
        scores = predict(model, paths, test)
        m = metrics(y[test], scores >= best['threshold'])
        folds.append({'fold': fold, 'best': best, 'metrics': m,
                      'train_sha256': [items[i]['sha256'] for i in tr],
                      'calibration_sha256': [items[i]['sha256'] for i in calibration],
                      'test_sha256': [items[i]['sha256'] for i in test]})
        for i, s in zip(test, scores):
            records.append({'name': items[i]['name'], 'sha256': items[i]['sha256'], 'label': int(y[i]),
                            'score': float(s), 'predicted': int(s >= best['threshold']),
                            'threshold': best['threshold'], 'fold': fold})
        (directory / 'oof_predictions.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({'finished_fold': fold, **m}), flush=True)
    if args.fold_limit < 5:
        (directory / 'partial.json').write_text(json.dumps({'partial_evaluation': True, 'folds': folds,
            'metrics': metrics([r['label'] for r in records], [r['predicted'] for r in records])}, indent=2), encoding='utf8')
        print('Partial training check finished; no production adapter exported.', flush=True)
        return
    final_epochs = int(np.median([f['best']['epoch'] for f in folds]))
    threshold = float(np.median([f['best']['threshold'] for f in folds]))
    report = {'version': VERSION, 'method': 'Native full-span video (up to 64 training frames, up to 96 evaluation frames), QLoRA last 12 layers q/v rank 8, frozen prefix and visual encoder, 5 SHA-deduplicated outer folds, inner 25% for threshold and epoch. Calibration requires recall >= 0.5.',
              'metrics': metrics([r['label'] for r in records], [r['predicted'] for r in records]),
              'folds': folds, 'unique_videos': len(items), 'final_epochs': final_epochs, 'threshold': threshold,
              'independent_dataset': False, 'unseen_camera_test': False,
              'category_labels': False, 'actor_labels': False,
              'base_size': args.base_size,
              'base_revision': '79cd853d0bc98bb0d67f865123eb49ef8985c2ec' if args.base_size == '4B' else '89644892e4d85e24eaac8bacfd4f463576704203'}
    (directory / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({'cross_validation': report['metrics']}), flush=True)
    set_peft_model_state_dict(model, initial)
    final_dir = ROOT / 'models' / f'forte-video-lora-{args.base_size.lower()}-v1'
    train(model, paths, y, indices, [], final_dir, 6300, final_epochs, threshold)
    (final_dir / 'metadata.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('Full-video adapter training completed.', flush=True)


if __name__ == '__main__':
    main()
