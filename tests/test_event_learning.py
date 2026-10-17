import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F


SOURCE = Path(__file__).resolve().parents[1] / 'deploy/internvideo/event_learning.py'
SPEC = importlib.util.spec_from_file_location('event_learning', SOURCE)
learning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(learning)


def test_positive_bag_updates_every_candidate_instead_of_only_the_maximum():
    values = torch.tensor([-2., .25, 1.5], requires_grad=True)
    learning.positive_bag_loss(values).backward()
    assert torch.all(values.grad < 0)
    assert values.grad[0].abs() < values.grad[-1].abs()
    hard = values.detach().clone().requires_grad_(True)
    F.softplus(-hard.max()).backward()
    assert torch.count_nonzero(hard.grad) == 1


@pytest.mark.parametrize('temperature', [.25, 1., 2.5])
def test_positive_streaming_coefficients_match_full_graph_gradients(temperature):
    features = [torch.tensor([1., 0.]), torch.tensor([0., 2.]), torch.tensor([-.5, 1.])]
    direct = torch.tensor([.3, -.7], requires_grad=True)
    scores = torch.stack([direct @ item for item in features])
    direct_loss = learning.positive_bag_loss(scores, temperature)
    direct_loss.backward()
    streamed = direct.detach().clone().requires_grad_(True)
    with torch.no_grad():
        snapshot = torch.stack([streamed @ item for item in features])
    coefficients = learning.positive_bag_coefficients(snapshot, temperature)
    assert not coefficients.requires_grad
    for coefficient, item in zip(coefficients, features):
        score = streamed @ item
        (coefficient * score).backward()
    torch.testing.assert_close(streamed.grad, direct.grad)


def test_negative_streaming_matches_equal_video_weighted_full_loss():
    normal_videos = [[torch.tensor([1., 0.])],
                     [torch.tensor([0., 1.]), torch.tensor([1., 1.]), torch.tensor([2., -1.])]]
    direct = torch.tensor([.2, -.4], requires_grad=True)
    expected = torch.stack([
        F.softplus(torch.stack([direct @ item for item in video])).mean()
        for video in normal_videos
    ]).mean()
    expected.backward()
    streamed = direct.detach().clone().requires_grad_(True)
    actual = 0.
    for video in normal_videos:
        for item in video:
            contribution = learning.negative_window_loss(streamed @ item, len(video)) / len(normal_videos)
            actual += float(contribution.detach())
            contribution.backward()
    assert actual == pytest.approx(float(expected.detach()))
    torch.testing.assert_close(streamed.grad, direct.grad)
    individual = torch.tensor([-3., 0., 3.], requires_grad=True)
    for score in individual.unbind():
        learning.negative_window_loss(score, len(individual)).backward(retain_graph=True)
    assert torch.all(individual.grad > 0)


@pytest.mark.parametrize('value', [-100., -2., .5, 100.])
def test_bag_score_is_invariant_to_identical_window_duplication(value):
    single = torch.tensor([value])
    duplicated = single.repeat(19)
    torch.testing.assert_close(learning.smooth_bag_score(single), learning.smooth_bag_score(duplicated))
    torch.testing.assert_close(learning.positive_bag_loss(single), learning.positive_bag_loss(duplicated))


def test_two_class_head_preserves_small_bfloat16_margin_in_float32():
    head = torch.nn.Linear(2, 4, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.fill_(1.)
        head.weight[3, 1] = 1.0078125
        head.bias.zero_()
    hidden = torch.tensor([[1., 1.]], dtype=torch.bfloat16, requires_grad=True)
    expected = F.linear(hidden.float(), head.weight.float(), head.bias.float())[:, [1, 3]]
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        actual = learning.binary_head_logits(hidden, head, [1, 3])
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    margin = actual[0, 1] - actual[0, 0]
    assert float(margin.detach()) == .0078125
    rounded = F.linear(hidden, head.weight[[1, 3]], head.bias[[1, 3]]).float()
    assert float((rounded[0, 1] - rounded[0, 0]).detach()) == 0.
    margin.backward()
    assert hidden.grad[0, 1] > 0


@pytest.mark.parametrize('scores', [torch.tensor([]), torch.tensor([[1.]]), torch.tensor([float('nan')])])
def test_invalid_positive_bag_is_rejected(scores):
    with pytest.raises(ValueError):
        learning.positive_bag_loss(scores)


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf'), True])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError):
        learning.positive_bag_loss(torch.tensor([1.]), temperature)


@pytest.mark.parametrize('count', [0, -1, 1.5, True])
def test_invalid_normal_video_count_is_rejected(count):
    with pytest.raises(ValueError):
        learning.negative_window_loss(torch.tensor(0.), count)


@pytest.mark.parametrize('ids', [[0], [0, 0], [-1, 1], [0, 4], [False, 1]])
def test_invalid_answer_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        learning.binary_head_logits(torch.zeros(1, 2), torch.nn.Linear(2, 4), ids)
