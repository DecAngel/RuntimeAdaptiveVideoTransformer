import sys
from pathlib import Path

# Path configs
s = set(locals().keys())

root_dir: Path = Path(__file__).parents[1].resolve()
ravt_dir: Path = root_dir.joinpath('ravt')
output_dir: Path = root_dir.joinpath('outputs')
output_ckpt_dir: Path = output_dir.joinpath('checkpoints')
output_result_dir: Path = output_dir.joinpath('results')
output_visualize_dir: Path = output_dir.joinpath('visualizations')
output_train_log_dir: Path = output_dir.joinpath('train_logs')
output_sap_log_dir: Path = output_dir.joinpath('sap_logs')
weight_dir: Path = root_dir.joinpath('weights')
weight_benchmark_dir: Path = weight_dir.joinpath('benchmark')
weight_pretrained_dir: Path = weight_dir.joinpath('pretrained')
weight_trained_dir: Path = weight_dir.joinpath('trained')
dataset_dir: Path = root_dir.joinpath('datasets')

s = set(locals().keys()) - s
for name in s:
    if name[-3:] == 'dir' and isinstance(locals()[name], Path):
        locals()[name].mkdir(exist_ok=True, parents=True)

sys.path.append(str(ravt_dir))

# Other configs
tensorboard_port = 8189
