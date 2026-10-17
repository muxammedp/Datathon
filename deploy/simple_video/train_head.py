import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.simple_video.analyzer import DEFAULTS, SimpleVideoAnalyzer, prepare_video
from deploy.simple_video.head import VERSION, digest, file_hash, fit_head, protocol, score
from deploy.internvideo.evaluate_dataset import check_dataset
from deploy.internvideo.training_protocol import choose_threshold, make_folds, metrics


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def train(args):
    began = time.perf_counter()
    dataset, output, model_path = args.dataset.resolve(), args.output.resolve(), args.model.resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (dataset, model_path)):
        raise ValueError('Training output must be separate from dataset and model directories')
    rows = check_dataset(dataset)['videos']
    splits = make_folds(rows)
    print(json.dumps(dict(stage='dataset_verified', unique_videos=len(rows),
                          next_stage='local_model_fingerprint', base_weights_updated=False)), flush=True)
    settings = dict(DEFAULTS, sample_fps=args.sample_fps)
    extraction = protocol(model_path, settings)
    identity = digest(dict(extraction=extraction, dataset_manifest_sha256=file_hash(dataset / 'manifest.json')))
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.training.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        config_path = output / 'protocol.json'
        config = dict(fingerprint=identity, extraction=extraction)
        if config_path.exists() and json.loads(config_path.read_text(encoding='utf8')) != config:
            raise ValueError('Code, model, dataset or settings changed; choose a new output directory')
        save(config_path, config)
        features_dir = output / 'features'
        features_dir.mkdir(exist_ok=True)
        analyzer = SimpleVideoAnalyzer(model_path, device=args.device)
        vectors = []
        for ordinal, row in enumerate(rows, 1):
            started = time.perf_counter()
            target = features_dir / (row['sha256'] + '.json')
            reused = target.exists()
            if reused:
                cached = json.loads(target.read_text(encoding='utf8'))
                payload = cached['payload']
                if cached.get('fingerprint') != identity or cached.get('sha256') != digest(payload) or payload.get('video_sha256') != row['sha256']:
                    raise ValueError('Feature cache identity or checksum mismatch')
            else:
                clips, metadata = prepare_video(dataset / row['file'], settings)
                if analyzer.model is None:
                    print(json.dumps(dict(stage='qwen_model_loading')), flush=True)
                    analyzer.load()
                    print(json.dumps(dict(stage='qwen_model_loaded', elapsed_sec=round(analyzer.load_elapsed_sec, 3))), flush=True)
                vector, profile = analyzer.extract_features(clips, metadata)
                payload = dict(video_sha256=row['sha256'], vector=vector, coverage=metadata, profile=profile)
                save(target, dict(fingerprint=identity, sha256=digest(payload), payload=payload))
                del clips
            vectors.append(payload['vector'])
            print(json.dumps(dict(stage='qwen_features', video=ordinal, total=len(rows), reused=reused,
                                  elapsed_sec=round(time.perf_counter() - started, 3))), flush=True)
        del analyzer
        predictions, reports = [], []
        for fold, split in enumerate(splits):
            trained = fit_head([vectors[i] for i in split['train']], [rows[i]['label'] for i in split['train']])
            cal_scores = [score(trained, vectors[i]) for i in split['calibration']]
            threshold, calibration = choose_threshold([rows[i]['label'] for i in split['calibration']], cal_scores)
            bundle = dict(version=VERSION, head=trained, threshold=threshold, extraction_protocol=extraction,
                          fold=fold, training_kind='frozen_qwen_with_learned_classifier', base_weights_updated=False,
                          **{name + '_sha256': [rows[i]['sha256'] for i in indices] for name, indices in split.items()})
            test = [dict(sha256=rows[i]['sha256'], label=rows[i]['label'], score=score(trained, vectors[i]),
                         threshold=threshold, fold=fold) for i in split['test']]
            for row in test:
                row['prediction'] = int(row['score'] >= row['threshold'])
            predictions.extend(test)
            reports.append(dict(fold=fold, calibration_metrics=calibration,
                                test_metrics=metrics([r['label'] for r in test], [r['score'] for r in test], threshold)))
            if fold == 0:
                save(output / 'predictor.json', bundle)
            print(json.dumps(dict(stage='classifier_fit', fold=fold + 1, gpu_training_passes=0)), flush=True)
        if len({r['sha256'] for r in predictions}) != len(rows):
            raise ValueError('Held-out predictions do not cover each unique video once')
        summary = dict(complete=True, version=VERSION, unique_videos=len(rows),
                       held_out_metrics=metrics([r['label'] for r in predictions],
                                                [r['score'] - r['threshold'] for r in predictions], 0.),
                       elapsed_sec=round(time.perf_counter() - began, 3),
                       trained_component='binary_classifier_head', base_weights_updated=False,
                       exported_fold=0, exported_fold_selected_before_evaluation=True,
                       independent_new_dataset=False, new_angles_validated=False,
                       new_violation_types_validated=False, exact_timestamps_trained=False,
                       production_approved=False,
                       note='One Qwen feature extraction per video clip. Five CPU classifier fits; no repeated GPU folds. '
                            'Exported fold 0 is trained on its training subset, calibrated separately; not fitted on all 40 videos.')
        save(output / 'evaluation.json', dict(summary, folds=reports, predictions=predictions))
        save(output / 'summary.json', summary)
        print(json.dumps(dict(stage='training_finished', **summary)), flush=True)
        return summary
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Train one small classifier on frozen Qwen video features; no model downloads.')
    parser.add_argument('--dataset', type=Path, default=Path('../internvideo/open-dataset'))
    parser.add_argument('--model', type=Path, default=Path('../ensemble/models/qwen'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sample-fps', type=float, default=DEFAULTS['sample_fps'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not .5 <= args.sample_fps <= 2:
        parser.error('Use sample-fps 0.5..2')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    try:
        train(args)
    except Exception as exc:
        print(json.dumps(dict(stage='training_failed', error_type=type(exc).__name__,
                              reason=str(exc) if type(exc) in (ValueError, FileExistsError) else 'Inspect the local runtime environment')), flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
