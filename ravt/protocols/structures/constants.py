from typing import Literal


ConfigTypes = Literal[
    'environment', 'launcher', 'dataset', 'model', 'evaluation', 'visualization', 'summary'
]
SubsetTypes = Literal['train', 'eval', 'test']
StageTypes = Literal['fit', 'validate', 'test', 'predict']
ComponentTypes = Literal['image', 'bbox']
