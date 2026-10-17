import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class FrozenVideoCache:
    def __init__(self, base, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.core = base.model
        self.original = self.core.get_video_features
        self.key = None
        self.strict = False
        owner = self

        def cached(core, pixel_values_videos, video_grid_thw=None):
            return owner.get(pixel_values_videos, video_grid_thw)

        self.core.get_video_features = types.MethodType(cached, self.core)

    def select(self, key):
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid feature cache key')
        self.key = key

    def get(self, pixels, grid):
        if self.key is None:
            raise ValueError('No video selected for feature cache')
        path = self.directory / (self.key + '.safetensors')
        if not path.exists():
            if self.strict:
                raise ValueError('Missing frozen visual features')
            self.core.visual.eval()
            with torch.inference_mode(False), torch.no_grad():
                embeds, deep = self.original(pixels, grid)
                tensors = {'pooled': torch.cat(embeds).detach().cpu().contiguous(),
                           'grid': grid.detach().cpu().contiguous()}
                tensors.update({f'deep_{i}': v.detach().cpu().contiguous() for i, v in enumerate(deep)})
                save_file(tensors, str(path))
        with torch.inference_mode(False):
            data = load_file(str(path), device='cuda')
        if not torch.equal(data['grid'], grid):
            raise ValueError('Frozen visual cache grid mismatch')
        sizes = (grid.prod(-1) // self.core.visual.spatial_merge_size ** 2).tolist()
        embeds = tuple(torch.split(data['pooled'], sizes))
        deep = [data[f'deep_{i}'] for i in range(len(data) - 2)]
        return embeds, deep

    def offload_visual(self, keys):
        if not all((self.directory / (key + '.safetensors')).exists() for key in keys):
            return False
        self.core.visual.cpu()
        self.strict = True
        torch.cuda.empty_cache()
        return True
