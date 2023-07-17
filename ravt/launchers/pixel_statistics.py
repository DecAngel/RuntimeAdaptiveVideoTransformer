import random
import sys
from pathlib import Path
from typing import Union, Dict, Any

import torch
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image, UnidentifiedImageError


class ImageStatisticsCounter:
    def __init__(self, mode: str = 'RGB'):
        self.mode = mode
        self.transform = ToTensor()
        self.means = []
        self.stds = []
        self.heights = []
        self.widths = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.print()

    def count_image(self, image: Image.Image) -> None:
        image = image.convert(self.mode)
        self.heights.append(image.height)
        self.widths.append(image.width)
        image = self.transform(image)
        image = torch.flatten(image, 1, 2)
        std, mean = torch.std_mean(image, dim=1)
        self.means.append(mean)
        self.stds.append(std)

    def count_file(self, image_path: Union[Path, str]) -> None:
        self.count_image(Image.open(image_path))

    def count_dir(self, dir_path: Union[Path, str], max_limit: int = 1000) -> int:
        return max_limit - self._count_dir(Path(dir_path), max_limit)

    def _count_dir(self, dir_path: Path, max_limit: int) -> int:
        seq = list(dir_path.iterdir())
        random.shuffle(seq)
        for p in seq:
            if p.is_dir():
                max_limit = self._count_dir(p, max_limit)
            else:
                try:
                    image = Image.open(p)
                except UnidentifiedImageError:
                    pass
                else:
                    self.count_image(image)
                    max_limit -= 1
            if max_limit == 0:
                return 0
        return max_limit

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'max_height': int(np.max(self.heights)),
            'avg_height': float(np.mean(self.heights)),
            'min_height': int(np.min(self.heights)),
            'bin_heights': list(np.histogram(self.heights)[0]),
            'max_width': int(np.max(self.widths)),
            'avg_width': float(np.mean(self.widths)),
            'min_width': int(np.min(self.widths)),
            'bin_widths': list(np.histogram(self.widths)[0]),
            'avg_pixel': torch.mean(torch.stack(self.means, dim=0), dim=0).tolist(),
            'std_pixel': torch.mean(torch.stack(self.stds, dim=0), dim=0).tolist(),
        }

    def print(self, file=sys.stdout):
        for k, v in self.get_statistics().items():
            print(f'{k}:\t{v}', file=file)


def run_pixel_statistics(image_or_path: Union[np.ndarray, Image.Image, Path, str], max_images: int = 1000) -> Dict[str, Any]:
    c = ImageStatisticsCounter()
    if isinstance(image_or_path, np.ndarray) or isinstance(image_or_path, Image.Image):
        if isinstance(image_or_path, np.ndarray):
            image = Image.fromarray(image_or_path)
        else:
            image = image_or_path
        c.count_image(image)
        return c.get_statistics()

    else:
        path = Path(image_or_path)
        if not path.exists():
            raise FileNotFoundError(f'Cannot find {str(path)}')
        elif path.is_file():
            c.count_file(path)
            return c.get_statistics()
        else:
            c.count_dir(path, max_images)
