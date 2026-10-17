import argparse
from pathlib import Path


def connect(source, destination):
    source = source.expanduser().resolve(strict=True)
    destination = destination.resolve(strict=True)
    if source == destination:
        raise ValueError('Existing installation and Git checkout must be different directories')
    names = ('.venv', 'models', 'open-dataset', 'results')
    for relative in ('.venv/bin/python', 'open-dataset/manifest.json',
                     'models/InternVideo2_5_Chat_8B/config.json'):
        if not (source / relative).is_file():
            raise ValueError('Missing required installation file: ' + relative)
    for name in names:
        target, link = source / name, destination / name
        if not target.is_dir():
            raise ValueError('Missing existing directory: ' + name)
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise ValueError('Refusing to replace existing path: ' + str(link))
    for name in names:
        link = destination / name
        if not link.is_symlink():
            link.symlink_to(source / name, target_is_directory=True)
        print(f'{name}: connected to existing installation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connect existing Brev data to a separate Git checkout')
    parser.add_argument('installation', type=Path, nargs='?', default=Path.home() / 'forte-internvideo')
    args = parser.parse_args()
    connect(args.installation, Path(__file__).resolve().parent)
