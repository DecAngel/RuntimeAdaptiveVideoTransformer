import logging
import sys

lightning_logger = logging.getLogger('pytorch_lightning')

ravt_logger = logging.getLogger('ravt')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter(fmt='{levelname}:{filename}:{funcName}:{asctime}:\n\t{message:s}', style='{'))
handler.setLevel(logging.INFO)
ravt_logger.setLevel(logging.INFO)
ravt_logger.addHandler(handler)
ravt_logger.propagate = False


def get_start_string(name: str):
    return f'{name:>^70}'


def get_end_string(name: str):
    return f'{name:<^70}'

