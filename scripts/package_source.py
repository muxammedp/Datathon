import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = [root / name for name in (
    'README.md', 'VALIDATION.md', 'VALIDATION-INITIAL.md', 'requirements.txt', 'requirements-research.txt', 'setup.ps1', 'start.ps1',
    'Start.cmd', 'Dockerfile', 'compose.yaml', '.gitignore', '.dockerignore',
)]
for directory in ('app', 'tests', 'deploy'):
    files.extend(path for path in (root / directory).rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc')
files.extend(root / 'scripts' / name for name in (
    'download_model.py', 'evaluate.py', 'api_smoke.py', 'check_frontend.py', 'package_source.py', 'research_classifier.py',
    'train_temporal.py', 'train_action.py', 'research_action_features.py', 'download_action_model.py',
    'download_video_model.py', 'resume_video_weights.py', 'train_full_video.py', 'video_feature_cache.py',
    'evaluate_full_context_2b.py',
    'package_internvideo_dataset.py',
))
target = root / 'ForteVision-source.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        if not path.exists():
            raise FileNotFoundError(path)
        archive.write(path, path.relative_to(root).as_posix())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert not any(name.startswith(('data/', 'models/', '.venv/')) for name in archive.namelist())
    print(json.dumps({'archive': target.name, 'files': len(archive.namelist()), 'size_bytes': target.stat().st_size,
                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))
