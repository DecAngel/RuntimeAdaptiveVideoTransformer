from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from ravt.core.configs import output_visualize_dir


class ImageWriter:
    def __init__(
            self,
            tag: str,
            mode: Literal['show_opencv', 'write_image', 'write_video'] = 'write_image',
            visualization_dir: Path = output_visualize_dir
    ):
        self.tag = tag
        self.mode = mode
        self.visualization_dir = visualization_dir

        self.counter = 0
        self.video_writer = None
        self.video_shape = None

    def __enter__(self):
        return self

    def write(self, image: np.ndarray):
        if self.mode == 'show_opencv':
            cv2.imshow(self.tag, image)
            cv2.waitKey(1)
        elif self.mode == 'write_image':
            cv2.imwrite(str(self.visualization_dir.joinpath(f'{self.tag}_{self.counter:05d}.jpg')), image)
        elif self.mode == 'write_video':
            if self.video_writer is None:
                self.video_writer = cv2.VideoWriter(
                    str(self.visualization_dir.joinpath(f'{self.tag}.mp4')),
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
        self.counter += 1

    def close(self):
        if self.mode == 'write_video' and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
