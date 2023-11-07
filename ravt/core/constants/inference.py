import numpy as np
from jaxtyping import UInt, Float


ImageInferenceType = UInt[np.ndarray, 'height width channels=3']
BBoxesInferenceType = Float[np.ndarray, 'time_f objects xyxypc=6']
BBoxInferenceType = Float[np.ndarray, 'objects xyxypc=6']
