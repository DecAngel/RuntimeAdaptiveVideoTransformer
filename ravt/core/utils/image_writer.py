from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ravt.core.configs import output_visualize_dir


class ImageWriter:
    def __init__(
            self,
            tag: str,
            mode: Literal['show_opencv', 'show_plt', 'show_tensorboard', 'write_image', 'write_video'] = 'write_image',
            visualization_dir: Path = output_visualize_dir
    ):
        self.tag = tag
        self.mode = mode
        self.visualization_dir = visualization_dir

        self.counter = defaultdict(lambda: 0)
        self.video_writer = None
        self.video_shape = None

    def __enter__(self):
        return self

    def write(self, image: np.ndarray, subtag: Optional[str] = None):
        identifier = self.tag if subtag is None else f'{self.tag}_{subtag}'
        self.counter[subtag] += 1
        count = self.counter[subtag]
        if self.mode == 'show_opencv':
            cv2.imshow(identifier, image)
            cv2.waitKey(1)
        elif self.mode == 'show_plt':
            plt.close(identifier)
            plt.figure(identifier)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        elif self.mode == 'write_image':
            cv2.imwrite(str(self.visualization_dir.joinpath(f'{identifier}_{count:05d}.jpg')), image)
        elif self.mode == 'write_video':
            if self.video_writer is None:
                self.video_writer = cv2.VideoWriter(
                    str(self.visualization_dir.joinpath(f'{identifier}.mp4')),
                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                    30,
                    (image.shape[1], image.shape[0])
                )
                self.video_shape = image.shape[:2]
            else:
                assert self.video_shape == image.shape[:2]
            self.video_writer.write(image)
        else:
            raise ValueError(f'Unsupported mode {self.mode}')

    def close(self):
        if self.mode == 'write_video' and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
