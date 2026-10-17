import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.ensemble.common import file_hash, model_manifest, verify_models


def main():
    parser = argparse.ArgumentParser(description='Download pinned public model files; no input videos are accessed.')
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--only', nargs='+', choices=['qwen', 'action', 'detector'])
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = model_manifest()
    if not args.verify_only:
        from huggingface_hub import hf_hub_download
        for name in args.only or manifest['models']:
            model = manifest['models'][name]
            for relative, expected in model['files'].items():
                target = args.models / name / relative
                if (target.is_file() and target.stat().st_size == expected['size']
                        and file_hash(target) == expected['sha256']):
                    continue
                hf_hub_download(model['repo'], relative, revision=model['revision'],
                                local_dir=str(args.models / name), force_download=target.exists())
    verify_models(args.models, args.only)
    print('Pinned models verified. No training, inference or video upload was performed.', flush=True)


if __name__ == '__main__':
    main()
