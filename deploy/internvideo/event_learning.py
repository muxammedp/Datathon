import math

import torch
from torch.nn import functional as F


def _scores(values):
    if not isinstance(values, torch.Tensor) or values.ndim != 1 or not values.numel():
        raise ValueError('Window scores must be a nonempty one-dimensional tensor')
    if not torch.isfinite(values).all():
        raise ValueError('Window scores must be finite')
    return values.float()


def _temperature(value):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError('MIL temperature must be finite and positive')
    return value


def negative_window_loss(score, window_count):
    if isinstance(window_count, bool) or not isinstance(window_count, int) or window_count < 1:
        raise ValueError('Normal video must have a positive integer window count')
    if not isinstance(score, torch.Tensor) or score.numel() != 1 or not torch.isfinite(score).all():
        raise ValueError('Window score must be a single finite tensor value')
    return F.softplus(score.float().reshape(())) / window_count


def smooth_bag_score(scores, temperature=1.0):
    values = _scores(scores)
    temperature = _temperature(temperature)
    return temperature * (torch.logsumexp(values / temperature, dim=0) - math.log(len(values)))


def positive_bag_loss(scores, temperature=1.0):
    return F.softplus(-smooth_bag_score(scores, temperature))


@torch.no_grad()
def positive_bag_coefficients(scores, temperature=1.0):
    values = _scores(scores)
    temperature = _temperature(temperature)
    bag = smooth_bag_score(values, temperature)
    return -torch.sigmoid(-bag) * torch.softmax(values / temperature, dim=0)


def binary_head_logits(hidden, head, answer_ids):
    if (len(answer_ids) != 2 or any(isinstance(i, bool) or not isinstance(i, int) for i in answer_ids)
            or answer_ids[0] == answer_ids[1]
            or any(i < 0 or i >= head.weight.shape[0] for i in answer_ids)):
        raise ValueError('Binary head requires two distinct valid answer token IDs')
    ids = torch.tensor(answer_ids, device=head.weight.device)
    bias = head.bias[ids].float() if head.bias is not None else None
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.linear(hidden.float(), head.weight[ids].float(), bias)
