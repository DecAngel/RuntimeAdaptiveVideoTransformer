from pathlib import Path
from typing import TypedDict, List, Callable, Optional

from pycocotools.coco import COCO


class EnvironmentConfigs(TypedDict, total=False):
    root_dir: Path
    ravt_dir: Path
    output_dir: Path
    output_ckpt_dir: Path
    output_result_dir: Path
    output_visualize_dir: Path
    output_train_log_dir: Path
    output_sap_log_dir: Path
    weight_dir: Path
    weight_benchmark_dir: Path
    weight_pretrained_dir: Path
    weight_trained_dir: Path
    dataset_dir: Path

    tensorboard_port: int


class DatasetConfigsRequiredKeysComponent(TypedDict, total=False):
    meta: List[int]
    image: List[int]
    bbox: List[int]


class DatasetConfigsRequiredKeys(TypedDict, total=False):
    interval: int
    margin: int
    components: DatasetConfigsRequiredKeysComponent


class DatasetConfigs(TypedDict, total=False):
    required_keys_train: DatasetConfigsRequiredKeys
    required_keys_eval: DatasetConfigsRequiredKeys


class PreprocessConfigs(TypedDict, total=False):
    pass


class ModelConfigs(TypedDict, total=False):
    pass


class PostprocessConfigs(TypedDict, total=False):
    pass


class EvaluationConfigs(TypedDict, total=False):
    coco_factory: Callable[[], Optional[COCO]]


class InternalConfigs(TypedDict):
    environment: EnvironmentConfigs
    dataset: DatasetConfigs
    preprocess: PreprocessConfigs
    model: ModelConfigs
    postprocess: PostprocessConfigs
    evaluation: EvaluationConfigs
