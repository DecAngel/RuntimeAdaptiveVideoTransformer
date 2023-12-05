from typing import Literal

ComponentLiteral = Literal[
    'meta', 'image', 'bbox', 'flow', 'feature'
]
SubsetLiteral = Literal[
    'train', 'eval', 'test'
]
