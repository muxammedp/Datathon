from html.parser import HTMLParser
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1] / 'app' / 'static'


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.assets = []
        self.inputs = []
        self.labels = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.append(attrs['id'])
        if tag in ('script', 'link'):
            path = attrs.get('src') or attrs.get('href')
            if path:
                self.assets.append(path)
        if tag == 'input':
            self.inputs.append(attrs.get('id'))
        if tag == 'label':
            self.labels.append(attrs.get('for'))


parser = PageParser()
parser.feed((root / 'index.html').read_text(encoding='utf8'))
assert len(parser.ids) == len(set(parser.ids)), 'Duplicate HTML IDs'
script = (root / 'app.js').read_text(encoding='utf8')
references = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)(?:[ '\)])", script))
assert references.issubset(set(parser.ids)), f'Missing JS targets: {references - set(parser.ids)}'
assert set(parser.inputs).issubset(set(parser.labels)), 'Unlabelled input'
for asset in parser.assets:
    assert asset.startswith('/static/'), f'External runtime asset: {asset}'
    assert (root / asset.removeprefix('/static/')).exists(), f'Missing asset: {asset}'
css = (root / 'styles.css').read_text(encoding='utf8')
for font in re.findall(r'url\("([^"\)]+)"\)', css):
    assert (root / font).exists(), f'Missing font: {font}'
assert 'innerHTML' not in script, 'Use safe DOM text construction'
assert 'localStorage' not in script, 'Do not persist API credentials'
print(f'PASS: {len(parser.ids)} unique IDs, {len(references)} JS targets, local assets, labelled inputs and safe text rendering')
