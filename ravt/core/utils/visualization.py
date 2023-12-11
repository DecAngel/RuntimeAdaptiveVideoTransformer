from typing import Tuple

import cv2
import torch
import numpy as np
from torchvision.utils import flow_to_image

from ravt.core.constants import (
    ImageComponentNDict, BBoxComponentNDict, FlowComponentNDict, FeatureComponentNDict,
)


def draw_image(image: ImageComponentNDict, size: Tuple[int, int] = (600, 960)):
    img = image['image'].transpose(1, 2, 0)
    return cv2.resize(img, size[::-1], interpolation=cv2.INTER_LINEAR)


def draw_bbox(bbox: BBoxComponentNDict, size: Tuple[int, int] = (600, 960)):
    resize_ratio = (np.array(size, dtype=np.float32) / bbox['current_size'])[[1, 0, 1, 0]]
    img = np.zeros((*size, 3), dtype=np.int8)
    for c, p, l in zip(
            bbox['coordinate'] * resize_ratio,
            bbox['probability'],
            bbox['label']
    ):
        if p > 1e-3:
            img = cv2.rectangle(
                img,
                *(c.astype(int).reshape(2, 2).tolist()),
                color=(0, 255, 0),
                thickness=2,
            )
        else:
            break
    return img


def draw_flow(flow: FlowComponentNDict, size: Tuple[int, int] = (600, 960)):
    img = flow_to_image(torch.from_numpy(flow['flow'])).numpy().transpose(1, 2, 0)
    return cv2.resize(img, size[::-1], interpolation=cv2.INTER_LINEAR)


def draw_feature(feature: FeatureComponentNDict, size: Tuple[int, int] = (600, 960)):
    A = torch.from_numpy(feature['feature'])
    C, H, W = A.size()
    A = A.permute(1, 2, 0).flatten(0, 1)  # HW, C
    U, S, V = torch.pca_lowrank(A)
    img = torch.matmul(A, V[:, :3])
    img = img.unflatten(0, (H, W)).numpy()
    img = ((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5) * 255).astype(np.uint8)
    return cv2.resize(img, size[::-1], interpolation=cv2.INTER_LINEAR)


def add_clip_id(image: np.ndarray, clip_id: int):
    image = np.pad(image, ((50, 10), (5, 5), (0, 0)), constant_values=(0, 0))
    image = cv2.putText(
        image, f'Frame T{"+" if clip_id >= 0 else "-"}{abs(clip_id)}', (0, 40), fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=2.5, color=(255, 255, 255), thickness=2
    )
    return image
