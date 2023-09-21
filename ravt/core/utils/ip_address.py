import sys
import re
import socket
import subprocess
from typing import List


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_all_local_ip() -> List[str]:
    proc = subprocess.Popen(
        'ipconfig' if sys.platform.startswith('win') else 'ifconfig',
        stdout=subprocess.PIPE, shell=True, text=True,
    )
    x = proc.communicate()[0]
    return list(filter(
        lambda i: not i.endswith(('.0', '.255')),
        re.findall(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', x)
    ))
