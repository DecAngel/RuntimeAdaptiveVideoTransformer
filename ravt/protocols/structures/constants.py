from typing import Literal


ConfigTypes = Literal[
    'environment', 'launcher', 'dataset', 'preprocess', 'model', 'postprocess', 'evaluation', 'summary'
]
SubsetTypes = Literal['train', 'eval', 'test']
StageTypes = Literal['fit', 'validate', 'test', 'predict']
ComponentTypes = Literal['meta', 'image', 'bbox']
