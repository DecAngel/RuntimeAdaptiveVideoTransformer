from pathlib import Path
from typing import TypedDict, List, Dict, Tuple

from ravt.core.constants.types import StageTypes


class EnvironmentConfigs(TypedDict, total=False):
    root_dir: Path
    ravt_dir: Path
    output_dir: Path
    output_ckpt_dir: Path
    output_result_dir: Path
    output_visualize_dir: Path
    output_train_log_dir: Path
    output_sap_log_dir: Path
    dataset_dir: Path

    tensorboard_port: int
    shared_memory_port: int


class BatchKeys(TypedDict, total=False):
    interval: int
    margin: int
    image: List[int]
    bbox: List[int]


class InternalConfigs(TypedDict, total=False):
    exp_tag: str
    stage: StageTypes
    original_size: Tuple[int, int]
    required_keys_train: BatchKeys
    required_keys_eval: BatchKeys
    produced_keys: BatchKeys


class AllConfigs(TypedDict):
    environment: EnvironmentConfigs
    internal: InternalConfigs
    extra: Dict


all_configs_empty: AllConfigs = {
    'environment': {},
    'internal': {},
    'extra': {},
}
