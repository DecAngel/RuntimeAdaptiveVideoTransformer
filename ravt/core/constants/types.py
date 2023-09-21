from typing import Literal

ComponentTypes = Literal[
    'image', 'bbox'
]
PhaseTypes = Literal[
    'environment', 'launcher', 'dataset', 'model', 'evaluation', 'visualization'
]
SubsetTypes = Literal[
    'train', 'eval', 'test'
]
StageTypes = Literal[
    'fit', 'validate', 'test', 'predict'
]
