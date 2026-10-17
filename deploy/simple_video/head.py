import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION = 'single-qwen-frozen-head-v1'
ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def protocol(model_path, settings):
    model_path = Path(model_path)
    weights = sorted(model_path.glob('*.safetensors'))
    if not weights:
        raise ValueError('Local model weights are missing')
    files = weights + sorted(model_path.glob('*.json'))
    value = dict(version=VERSION, settings=settings,
                 model_files={p.name: file_hash(p) for p in files},
                 source_files={name: file_hash(ROOT / name) for name in
                               ('deploy/simple_video/analyzer.py', 'deploy/simple_video/head.py', 'app/rules.py')},
                 feature='final_prompt_hidden_state_duration_weighted_mean', classifier='train_only_standardizer_l2_logistic_C0.01')
    return dict(value, fingerprint=digest(value))


def fit_head(vectors, labels):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(labels) or set(labels) != {0, 1} or not np.isfinite(values).all():
        raise ValueError('Training needs finite vectors and both label classes')
    scaler = StandardScaler().fit(values)
    model = LogisticRegression(C=.01, solver='lbfgs', max_iter=2000, random_state=123)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(values), labels)
    return dict(dimensions=values.shape[1], mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))


def score(head, vector):
    values = [np.asarray(head[key], dtype=np.float64) for key in ('mean', 'scale', 'coef')]
    vector = np.asarray(vector, dtype=np.float64)
    if (vector.shape != (head['dimensions'],) or not np.isfinite(vector).all()
            or any(v.shape != vector.shape or not np.isfinite(v).all() for v in values)
            or np.any(values[1] <= 0) or not math.isfinite(head['intercept'])):
        raise ValueError('Invalid classifier dimensions or coefficients')
    logit = float(((vector - values[0]) / values[1]) @ values[2] + head['intercept'])
    if not math.isfinite(logit):
        raise ValueError('Nonfinite classifier output')
    return float(np.exp(-np.logaddexp(0., -logit)))


def predict(bundle, vector):
    if bundle.get('version') != VERSION or not math.isfinite(bundle['threshold']):
        raise ValueError('Invalid learned classifier')
    value = score(bundle['head'], vector)
    return dict(score=value, threshold=bundle['threshold'],
                prediction='violation' if value >= bundle['threshold'] else 'no_violation',
                category=None, exact_event_verified=False)
