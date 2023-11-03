import logging
import sys


class RAVTFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s:%(name)s:%(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


lightning_logger = logging.getLogger('pytorch_lightning')

ravt_logger = logging.getLogger('ravt')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(RAVTFormatter())
handler.setLevel(logging.INFO)
ravt_logger.setLevel(logging.INFO)
ravt_logger.addHandler(handler)
ravt_logger.propagate = False
