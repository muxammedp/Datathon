import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def save_payload(path, payload, fingerprint):
    save_json(path, dict(fingerprint=fingerprint, sha256=digest(payload), payload=payload))


def load_payload(path, fingerprint):
    value = read_json(path)
    if value.get('fingerprint') != fingerprint or value.get('sha256') != digest(value.get('payload')):
        raise ValueError(f'Cache identity/checksum mismatch: {path}; choose a new output directory')
    return value['payload']


def offline():
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    os.environ['YOLO_VERBOSE'] = 'False'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def model_manifest():
    return read_json(HERE / 'model-manifest.json')


def verify_models(root, names=None):
    manifest = model_manifest()
    for name in names or manifest['models']:
        for relative, expected in manifest['models'][name]['files'].items():
            path = Path(root) / name / relative
            if (not path.is_file() or path.stat().st_size != expected['size']
                    or file_hash(path) != expected['sha256']):
                raise ValueError(f'Missing or mismatched pinned model file: {path}; run download.py')
    return manifest


@contextmanager
def exclusive_output(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.ensemble.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'Output is locked: {lock}. After a crash, remove this file only after confirming the process stopped.') from exc
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def release_gpu():
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
