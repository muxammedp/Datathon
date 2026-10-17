import math
import random


def make_folds(rows, seed=123):
    identities = [row['sha256'] for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError('Duplicate video identity')
    if any(row['label'] not in (0, 1) for row in rows):
        raise ValueError('Only binary labels are supported')
    groups = [[] for _ in range(5)]
    rng = random.Random(seed)
    for label in (1, 0):
        indices = sorted((i for i, row in enumerate(rows) if row['label'] == label),
                         key=lambda i: rows[i]['sha256'])
        if len(indices) < 5:
            raise ValueError('Five-fold validation needs at least five unique videos per class')
        rng.shuffle(indices)
        counts = [0] * 5
        for index in indices:
            group = min(range(5), key=lambda g: (counts[g], len(groups[g]), g))
            groups[group].append(index)
            counts[group] += 1
    folds = []
    for fold in range(5):
        test, calibration = groups[fold], groups[(fold + 1) % 5]
        train = [i for i in range(len(rows)) if i not in test and i not in calibration]
        folds.append(dict(train=train, calibration=calibration, test=test))
    return folds


def metrics(labels, scores, threshold):
    if len(labels) != len(scores) or not labels:
        raise ValueError('Metrics require nonempty paired labels and scores')
    if not math.isfinite(threshold) or any(not math.isfinite(s) for s in scores):
        raise ValueError('Nonfinite classification score')
    result = dict(tn=0, fp=0, fn=0, tp=0)
    for label, score in zip(labels, scores):
        key = ('tp' if score >= threshold else 'fn') if label else ('fp' if score >= threshold else 'tn')
        result[key] += 1
    negatives, positives = result['tn'] + result['fp'], result['tp'] + result['fn']
    if not positives or not negatives:
        raise ValueError('Both classes are required for evaluation')
    recall, fpr = result['tp'] / positives, result['fp'] / negatives
    return dict(result, total=len(labels), accuracy=(result['tp'] + result['tn']) / len(labels),
                precision=result['tp'] / (result['tp'] + result['fp']) if result['tp'] + result['fp'] else 0.,
                recall=recall, false_positive_rate=fpr, balanced_accuracy=(recall + 1 - fpr) / 2)


def choose_threshold(labels, scores):
    candidates = sorted(set([0.] + list(scores) + [math.nextafter(max(scores), math.inf)]))
    choices = []
    for threshold in candidates:
        result = metrics(labels, scores, threshold)
        if result['recall'] < .5:
            continue
        cost = (2 * result['false_positive_rate'] + 1 - result['recall']) / 3
        choices.append(((cost, -result['recall'], abs(threshold)), threshold, result))
    _, threshold, result = min(choices, key=lambda item: item[0])
    return threshold, result


def objective(result):
    return (2 * result['false_positive_rate'] + 1 - result['recall']) / 3


def binary_loss(labels, scores):
    return sum(max(score, 0.) - label * score + math.log1p(math.exp(-abs(score)))
               for label, score in zip(labels, scores)) / len(labels)
