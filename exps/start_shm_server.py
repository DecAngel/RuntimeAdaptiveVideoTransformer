import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parents[1].resolve())
os.chdir(root_dir)
sys.path.append(root_dir)
print(f'Working Directory: {root_dir}')

from ravt.launchers import SharedMemoryLauncher


def main():
    server = SharedMemoryLauncher()
    server.run()


if __name__ == '__main__':
    main()
