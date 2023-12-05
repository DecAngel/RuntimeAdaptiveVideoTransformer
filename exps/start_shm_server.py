import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parents[1].resolve())
os.chdir(root_dir)
sys.path.append(root_dir)
print(f'Working Directory: {root_dir}')

from ravt.core.launchers.shared_memory import run_shm_server


def main():
    run_shm_server()


if __name__ == '__main__':
    main()
