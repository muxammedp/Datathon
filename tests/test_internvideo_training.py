import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'deploy/internvideo'


def module(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / (name + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


protocol = module('training_protocol')
lora = module('lora_binary')


def test_all_videos_are_tested_once_without_cross_split_duplicates():
    rows = [{'sha256': f'{i:064x}', 'label': int(i < 21)} for i in range(40)]
    splits = protocol.make_folds(rows)
    tested = []
    for split in splits:
        train, calibration, test = (set(split[k]) for k in ('train', 'calibration', 'test'))
        assert [len(train), len(calibration), len(test)] == [24, 8, 8]
        assert not train & calibration and not train & test and not calibration & test
        assert len(train | calibration | test) == 40
        for group in (train, calibration, test):
            assert {rows[i]['label'] for i in group} == {0, 1}
        tested.extend(test)
    assert sorted(tested) == list(range(40))
    assert splits == protocol.make_folds(rows)
    with pytest.raises(ValueError, match='Duplicate'):
        protocol.make_folds(rows + [rows[0]])


def test_threshold_counts_false_alarms_and_misses():
    threshold, result = protocol.choose_threshold([0, 0, 1, 1], [-2., -1., 1., 2.])
    assert result['fp'] == result['fn'] == 0
    assert -1 < threshold <= 1
    with pytest.raises(ValueError, match='Nonfinite'):
        protocol.metrics([0, 1], [0., float('nan')], 0.)
    _, poor = protocol.choose_threshold([0, 0, 1, 1], [2., 3., -2., -1.])
    assert poor['recall'] >= .5


def test_lora_changes_only_adapter_and_restores_exact_predictions():
    torch.manual_seed(7)
    base = torch.nn.Linear(6, 4)
    weights = base.weight.detach().clone()
    inputs = torch.randn(3, 6)
    expected = base(inputs).detach()
    layer = lora.LoRALinear(base, rank=2, dropout=0)
    torch.testing.assert_close(layer(inputs), expected)
    initial = lora.adapter_state(layer)
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.05)
    loss = layer(inputs).square().mean()
    loss.backward()
    assert layer.b.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.allclose(layer(inputs), expected)
    torch.testing.assert_close(base.weight, weights, atol=0, rtol=0)
    lora.restore_adapter(layer, initial)
    torch.testing.assert_close(layer(inputs), expected, atol=0, rtol=0)


def test_last_token_scoring_matches_full_language_head_and_backpropagates():
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = lora.LoRALinear(torch.nn.Linear(6, 6), rank=2, dropout=0)

        def forward(self, inputs_embeds, **kwargs):
            return SimpleNamespace(last_hidden_state=self.layer(inputs_embeds))

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()
            self.head = torch.nn.Linear(6, 20)
            self.head.requires_grad_(False)

        def get_output_embeddings(self):
            return self.head

    model = LanguageModel()
    inputs = torch.randn(1, 7, 6)
    result = lora.binary_logits(model, inputs, [2, 9])
    reference = model.head(model.model(inputs_embeds=inputs).last_hidden_state)[:, -1, [2, 9]]
    torch.testing.assert_close(result, reference)
    torch.nn.functional.cross_entropy(result, torch.tensor([1])).backward()
    assert model.model.layer.b.grad.abs().sum() > 0


def test_protocol_refuses_changed_settings_and_unidentified_results(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    training = module('train_lora')
    directory = tmp_path / 'run'
    training.lock_protocol(directory, {'epochs': 2})
    training.lock_protocol(directory, {'epochs': 2})
    with pytest.raises(ValueError, match='protocol changed'):
        training.lock_protocol(directory, {'epochs': 3})
    other = tmp_path / 'unknown'
    other.mkdir()
    (other / 'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='not empty'):
        training.lock_protocol(other, {'epochs': 2})


def test_frozen_inputs_contain_frames_but_not_answers():
    class Template:
        roles = ('user', 'assistant')

        def copy(self):
            return Template()

        def append_message(self, role, message):
            if message is not None:
                self.prompt = message

        def get_prompt(self):
            return self.prompt

    class Tokenizer:
        def __call__(self, text, **kwargs):
            count = text.count('<IMG_CONTEXT>')
            return {'input_ids': torch.tensor([[8] + [7] * count + [3, int(text[-1])]])}

        def convert_tokens_to_ids(self, text):
            return 7

    embedding = torch.nn.Embedding(10, 6)
    model = SimpleNamespace(conv_template=Template(), system_message='', num_image_token=2,
                            config=SimpleNamespace(llm_config=SimpleNamespace(hidden_size=6)),
                            language_model=SimpleNamespace(get_input_embeddings=lambda: embedding),
                            extract_feature=lambda pixels: torch.ones(4, 2, 6))
    inputs, answers = lora.frozen_inputs(model, Tokenizer(), torch.zeros(4, 3, 2, 2), [0, 1, 2, 3], 'binary')
    assert answers == [0, 1]
    assert inputs.shape == (1, 10, 6)
    torch.testing.assert_close(inputs[0, 1:9], torch.ones(8, 6))
    torch.testing.assert_close(inputs[0, -1], embedding.weight[3])
    assert not inputs.requires_grad
