import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_temporal import dataset_items
from scripts.evaluate import labels_from_xlsx, normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    items = dataset_items(args.dataset)
    lookup = {normalize(path.stem): path for path in (args.dataset / 'videos').glob('*.mov')}
    aliases = {}
    for row in labels_from_xlsx(args.dataset / 'open_labels.xlsx'):
        path = lookup[normalize(row['name'])]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        aliases.setdefault(digest, []).append(path.name)
    manifest = {'dataset': 'provided_open_dataset', 'deduplicated': True,
                'category_labels': False, 'actor_labels': False, 'videos': []}
    for item in items:
        manifest['videos'].append({'file': 'videos/' + Path(item['path']).name,
                                  'sha256': item['sha256'], 'label': item['label'],
                                  'event_times': item['event_times'], 'aliases': aliases[item['sha256']]})
    target = ROOT / 'ForteVision-open-dataset.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr('open-dataset/manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        for item in items:
            archive.write(item['path'], 'open-dataset/videos/' + Path(item['path']).name)
        archive.write(args.dataset / 'open_labels.xlsx', 'open-dataset/open_labels.xlsx')
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(items) + 2
    print(json.dumps({'archive': target.name, 'unique_videos': len(items),
                      'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
