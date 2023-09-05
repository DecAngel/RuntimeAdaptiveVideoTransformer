import torch.nn as nn
import kornia.augmentation as ka


transform = ka.AugmentationSequential(ka.VideoSequential(
    ka.RandomHorizontalFlip(),
    ka.Resize((600, 960)),
))

train(
    dataset=dataset,
    model=model,
    transform=transform,
    backbone=,
    neck=,
    metric=,
)
