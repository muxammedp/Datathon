import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file


SOURCE = Path(__file__).resolve().parents[1] / 'deploy/internvideo'


@pytest.fixture
def training(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    return importlib.import_module('train_events_v3')


class TinyDecoder(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([.2, -.3, .1, .4]))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=self.dropout(inputs_embeds * self.scale))


class TinyLanguage(torch.nn.Module):
    def __init__(self, dropout=.35):
        super().__init__()
        self.model = TinyDecoder(dropout)
        self.head = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.head.weight[0].zero_()
            self.head.weight[1].fill_(1.)
        self.head.requires_grad_(False)

    def get_output_embeddings(self):
        return self.head


def cache_paths(directory, vectors):
    paths = []
    for index, vector in enumerate(vectors):
        path = directory / f'window-{index}.safetensors'
        save_file({'embeds': torch.tensor(vector).reshape(1, 1, 4).repeat(1, 3, 1)}, str(path))
        paths.append(path)
    return paths


def test_backward_normal_video_supervises_every_window_with_equal_video_weight(training, tmp_path):
    model = TinyLanguage(dropout=0)
    model.train()
    paths = cache_paths(tmp_path, torch.eye(4).tolist())
    expected_model = copy.deepcopy(model)
    expected_scores = torch.stack([training.window_score(expected_model, path, [0, 1]) for path in paths])
    expected_loss = torch.nn.functional.softplus(expected_scores).mean()
    (expected_loss / 3).backward()
    actual_loss = training.backward_video(model, paths, 0, [0, 1], 3, 1.)
    assert actual_loss == pytest.approx(float(expected_loss.detach()))
    assert torch.all(model.model.scale.grad > 0)
    torch.testing.assert_close(model.model.scale.grad, expected_model.model.scale.grad)


def test_backward_positive_video_replays_dropout_and_matches_direct_mil_gradients(training, tmp_path):
    vectors = [[1., 2., -1., .5], [-.5, 1., .2, 1.], [2., -.5, 1., 2.]]
    paths = cache_paths(tmp_path, vectors)
    model = TinyLanguage(dropout=.5)
    expected_model = copy.deepcopy(model)
    model.train()
    expected_model.train()
    torch.manual_seed(137)
    expected_scores = torch.stack([training.window_score(expected_model, path, [0, 1]) for path in paths])
    expected_loss = training.positive_bag_loss(expected_scores, .7)
    rng_after_direct = torch.get_rng_state().clone()
    (expected_loss / 4).backward()
    torch.manual_seed(137)
    actual_loss = training.backward_video(model, paths, 1, [0, 1], 4, .7)
    assert actual_loss == pytest.approx(float(expected_loss.detach()))
    torch.testing.assert_close(model.model.scale.grad, expected_model.model.scale.grad)
    assert torch.equal(torch.get_rng_state(), rng_after_direct)
    assert model.model.scale.grad.abs().sum() > 0


def test_replay_refuses_forward_changes_between_no_grad_and_backward_pass(training, tmp_path, monkeypatch):
    paths = cache_paths(tmp_path, [[1., 1., 1., 1.]])
    model = TinyLanguage(dropout=0)
    model.train()
    original = training.window_score

    def inconsistent(*args, **kwargs):
        score = original(*args, **kwargs)
        return score + .1 if torch.is_grad_enabled() else score

    monkeypatch.setattr(training, 'window_score', inconsistent)
    with pytest.raises(RuntimeError, match='refusing inexact gradients'):
        training.backward_video(model, paths, 1, [0, 1], 1, 1.)
    assert model.model.scale.grad is None


def synthetic_dataset(directory):
    cache = directory / 'cache'
    cache.mkdir()
    rows, windows = [], []
    for index in range(40):
        label = int(index < 21)
        identity = f'{index:064x}'
        rows.append(dict(sha256=identity, label=label, file=f'video-{index}.mov',
                         aliases=[], event_times=[15.] if label else []))
        plan = []
        for number in range(2 + index % 2):
            key = f'{identity}-{number}'
            values = torch.tensor([float(label * 2 - 1), .2 + index / 100, .1 * number, .5])
            save_file({'embeds': values.reshape(1, 1, 4).repeat(1, 3, 1)}, str(cache / f'{key}.safetensors'))
            plan.append(dict(index=number, start_sec=12. * number, end_sec=12. * number + 16,
                             cache_key=key))
        windows.append(plan)
    return rows, windows, cache


def arguments(directory, **overrides):
    output = directory / 'run'
    output.mkdir()
    values = dict(output=output, seed=123, epochs=2, min_epochs=1, patience=2, lr=.01, temperature=1.)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_five_fold_cpu_training_preserves_held_out_identities_and_exports_complete_traces(
        training, tmp_path, monkeypatch):
    rows, windows, cache = synthetic_dataset(tmp_path)
    args = arguments(tmp_path)
    model = TinyLanguage(dropout=.2)
    initial = training.adapter_state(model)
    frozen = training.score_windows(model, list(range(40)), windows, cache, [0, 1])
    splits = training.make_folds(rows, args.seed)
    original_backward = training.backward_video
    original_calibration = training.calibration
    trained_paths, calibration_indices = [], []

    def observed_backward(model, paths, *extra):
        trained_paths.append([path.stem for path in paths])
        return original_backward(model, paths, *extra)

    def observed_calibration(rows, indices, scores):
        calibration_indices.append(list(indices))
        return original_calibration(rows, indices, scores)

    monkeypatch.setattr(training, 'backward_video', observed_backward)
    monkeypatch.setattr(training, 'calibration', observed_calibration)
    monkeypatch.setattr(training, 'emit', lambda **kwargs: None)
    reports = []
    for fold, split in enumerate(splits):
        trained_paths.clear()
        calibration_indices.clear()
        result = training.train_fold(model, rows, windows, cache, [0, 1], split, fold, initial, frozen, args)
        reports.append(result)
        allowed = {rows[i]['sha256'] for i in split['train']}
        assert all(key.rsplit('-', 1)[0] in allowed for paths in trained_paths for key in paths)
        assert len(trained_paths) == len(split['train']) * len(result['history'])
        normal = {rows[i]['sha256']: len(windows[i]) for i in split['train'] if not rows[i]['label']}
        for paths in trained_paths:
            identity = paths[0].rsplit('-', 1)[0]
            if identity in normal:
                assert len(paths) == normal[identity]
        assert calibration_indices and all(indices == split['calibration'] for indices in calibration_indices)
        assert {p['sha256'] for p in result['predictions']} == {rows[i]['sha256'] for i in split['test']}
        for record in result['history']:
            assert record['train_metrics_at_calibration_threshold']['total'] == 24
            assert {trace['sha256'] for trace in record['calibration_window_traces']} == {
                rows[i]['sha256'] for i in split['calibration']}
        for prediction in result['predictions']:
            assert len(prediction['window_traces']) == prediction['windows_processed']
            assert len(prediction['window_scores']) == len(prediction['frozen_window_scores'])
            assert prediction['score'] == max(prediction['window_scores'])
        assert result['adapter_sha256'] == training.file_hash(args.output / f'fold-{fold}/adapter.safetensors')
    summary = training.report(args.output, reports, rows, args.seed)
    audit = training.audit_reports(rows, reports, args.seed)
    assert summary['complete'] and audit['complete']
    assert audit['videos_evaluated'] == 40
    assert audit['per_window_scores_available']
    assert audit['metrics'] == summary['held_out_metrics']
    assert not summary['production_approved']
    assert not summary['independent_new_dataset']
    assert len(json.loads((args.output / 'error-audit.json').read_text())['records']) == 40


def test_early_stopping_restores_epoch_zero_when_calibration_does_not_improve(training, tmp_path, monkeypatch):
    rows, windows, cache = synthetic_dataset(tmp_path)
    args = arguments(tmp_path, epochs=6, min_epochs=2, patience=1)
    model = TinyLanguage(dropout=0)
    initial = training.adapter_state(model)
    split = training.make_folds(rows, args.seed)[0]
    frozen = training.score_windows(model, list(range(40)), windows, cache, [0, 1])
    fixed = training.calibration(rows, split['calibration'], frozen)
    observed = []

    def unchanged_calibration(*unused):
        observed.append(model.model.scale.detach().clone())
        return copy.deepcopy(fixed)

    monkeypatch.setattr(training, 'calibration', unchanged_calibration)
    monkeypatch.setattr(training, 'emit', lambda **kwargs: None)
    result = training.train_fold(model, rows, windows, cache, [0, 1], split, 0, initial, frozen, args)
    assert result['best']['epoch'] == 0
    assert len(result['history']) == 2
    assert any(not torch.equal(value, initial['model.scale']) for value in observed[1:-1])
    saved = load_file(str(args.output / 'fold-0/adapter.safetensors'))
    torch.testing.assert_close(saved['model.scale'], initial['model.scale'], atol=0, rtol=0)
    torch.testing.assert_close(model.model.scale, initial['model.scale'], atol=0, rtol=0)
