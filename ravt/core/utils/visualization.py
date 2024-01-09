from typing import Tuple, List

import cv2
import torch
import numpy as np
from torchvision.utils import flow_to_image

from ravt.core.constants import (
    ImageComponentNDict, BBoxComponentNDict, FlowComponentNDict, FeatureComponentNDict,
)

feature_enhance_gamma = 0.5


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
    img = flow_to_image(torch.from_numpy(np.nan_to_num(flow['flow'])).to(dtype=torch.float32)).numpy().transpose(1, 2, 0)
    return cv2.resize(img, size[::-1], interpolation=cv2.INTER_LINEAR)


def draw_feature(feature: FeatureComponentNDict, size: Tuple[int, int] = (600, 960)):
    A = torch.from_numpy(feature['feature']).to(dtype=torch.float32)
    C, H, W = A.size()
    A = A.permute(1, 2, 0).flatten(0, 1)  # HW, C
    U, S, V = torch.pca_lowrank(A)
    img = torch.matmul(A, V[:, :3])
    img = img.unflatten(0, (H, W)).numpy()
    img = (img - np.mean(img)) / np.std(img)
    img = np.float_power(np.abs(img), feature_enhance_gamma) * np.sign(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
    img = (img * 255).astype(np.uint8)
    return cv2.resize(img, size[::-1], interpolation=cv2.INTER_LINEAR)


def draw_feature_batch(features: List[FeatureComponentNDict], size: Tuple[int, int] = (600, 960)):
    A = torch.stack([torch.from_numpy(f['feature']) for f in features]).to(dtype=torch.float32)
    B, C, H, W = A.size()
    A = A.permute(0, 2, 3, 1).flatten(0, 2)  # BHW, C
    U, S, V = torch.pca_lowrank(A)
    img = torch.matmul(A, V[:, :3])
    img = img.unflatten(0, (B, H, W)).numpy()
    img = (img - np.mean(img)) / np.std(img)
    img = np.float_power(np.abs(img), feature_enhance_gamma) * np.sign(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
    img = (img * 255).astype(np.uint8)
    return [cv2.resize(i, size[::-1], interpolation=cv2.INTER_LINEAR) for i in img]


def add_clip_id(image: np.ndarray, clip_id: int):
    image = np.pad(image, ((50, 10), (5, 5), (0, 0)), constant_values=(0, 0))
    image = cv2.putText(
        image, f'Frame T{"+" if clip_id >= 0 else "-"}{abs(clip_id)}', (0, 40), fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=2.5, color=(255, 255, 255), thickness=2
    )
    return image


def draw_grid_clip_id(image_list: List[List[np.ndarray]], clip_ids: List[int]):
    # constants
    pad_width = 2
    H, W, C = image_list[0][0].shape
    x, y = len(image_list[0]), len(image_list)

    res = []

    title = np.zeros((10, (W+2*pad_width)*x, 3), dtype=np.uint8)
    for i, c in enumerate(clip_ids):
        title = cv2.putText(
            title, f'Frame T{"+" if c >= 0 else "-"}{abs(c)}', ((H+2*pad_width)*i, 40), fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2.5, color=(255, 255, 255), thickness=2
        )
    res.append(title)

    for il in image_list:
        res.append(np.concatenate(
            [np.pad(i, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), constant_values=(0, 0)) for i in il],
            axis=1,
        ))

    return np.concatenate(res, axis=0)
