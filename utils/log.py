
import logging
from pathlib import Path


def fileLog(fpath):
    Path(fpath).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name='a')  # set root logger if not set name
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(levelname)s] [%(asctime)s]: %(message)s',
        datefmt='%H:%M:%S')
    # output to file by using FileHandler
    fh = logging.FileHandler(f"{fpath}/log.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # output to screen by using StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # add Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def mylogger(flag, fpath):

    if flag == 1:
        Path(fpath).mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(name='a')  # set root logger if not set name
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(levelname)s] [%(asctime)s]: %(message)s',
            datefmt='%H:%M:%S')
        # output to file by using FileHandler

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(f"{fpath}/log.txt")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    elif flag == 2:
        logger = logging.getLogger(name='b')  # set root logger if not set name
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(levelname)s] [%(asctime)s]: %(message)s',
            datefmt='%H:%M:%S')
        # output to screen by using StreamHandler
        c = logging.StreamHandler()
        c.setLevel(logging.DEBUG)
        c.setFormatter(formatter)
        logger.addHandler(c)

    else:
        raise NameError

    return logger

