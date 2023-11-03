import sys
import os
from pathlib import Path

# Path configs
# Input paths, must exist.
s = set(locals().keys())
root_dir: Path = Path(__file__).parents[2].resolve()
ravt_dir: Path = root_dir.joinpath('ravt')
dataset_dir: Path = root_dir.joinpath('datasets')
s = set(locals().keys()) - s - set('s')
for name in s:
    if name[-3:] == 'dir' and isinstance(locals()[name], Path):
        assert locals()[name].exists(), f'{locals()[name]} does not exist!'

# Output paths, mkdir if not exist.
s = set(locals().keys())
output_dir: Path = root_dir.joinpath('outputs')
output_ckpt_dir: Path = output_dir.joinpath('checkpoints')
output_result_dir: Path = output_dir.joinpath('results')
output_visualize_dir: Path = output_dir.joinpath('visualizations')
output_train_log_dir: Path = output_dir.joinpath('train_logs')
output_sap_log_dir: Path = output_dir.joinpath('sap_logs')
s = set(locals().keys()) - s - set('s')
for name in s:
    if name[-3:] == 'dir' and isinstance(locals()[name], Path):
        locals()[name].mkdir(exist_ok=True, parents=True)


sys.path.append(str(root_dir))
os.environ['PATH'] = f'{str(ravt_dir)}:{os.environ["PATH"]}'

# Other configs
tensorboard_port = 8189
shared_memory_port = 8200
