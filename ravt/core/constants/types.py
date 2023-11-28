from typing import Literal

ComponentTypes = Literal[
    'image', 'bbox'
]
SubsetTypes = Literal[
    'train', 'eval', 'test'
]
VisualizationTypes = Literal[
    'bbox', 'flow', 'feature'
]
