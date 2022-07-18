import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


# borrowed from Hoel
class ConsoleColor:
    """Simple shortcut to use colors in the console."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m'
    RED = '\033[91m'
    ENDCOLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColoredFormatter(logging.Formatter):
    """Formatter adding colors to levelname."""
    def format(self, record):
        levelno = record.levelno
        if logging.ERROR == levelno:
            levelname_color = ConsoleColor.RED + record.levelname + ConsoleColor.ENDCOLOR
        elif logging.WARNING == levelno:
            levelname_color = ConsoleColor.ORANGE + record.levelname + ConsoleColor.ENDCOLOR
        elif logging.INFO == levelno:
            levelname_color = ConsoleColor.GREEN + record.levelname + ConsoleColor.ENDCOLOR
        elif logging.DEBUG == levelno:
            levelname_color = ConsoleColor.BLUE + record.levelname + ConsoleColor.ENDCOLOR
        else:
            levelname_color = record.levelname
        record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def set_verbose_level(verbose_level: str) -> int:
    match verbose_level:
        case "info":
            level = logging.INFO
        case "debug":
            level = logging.DEBUG
        case "error":
            level = logging.ERROR
        case "_":
            raise NotImplementedError
    return level


def create_logger(logs_dir: Optional[Path] = None, name: Union[Path, str] = None, verbose_level: str = None):
    """Create logger."""
    logger = logging.getLogger()
    logger.setLevel(set_verbose_level(verbose_level))
    sh = logging.StreamHandler()
    colored_log_formatter = ColoredFormatter("%(asctime)s %(name)s %(levelname)s:%(message)s")
    sh.setFormatter(colored_log_formatter)
    logger.addHandler(sh)
    if logs_dir is not None:
        logs_dir = logs_dir / datetime.now().strftime("%Y-%m-%d")
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(filename=str(logs_dir / name) + ".log")
        fh.setFormatter(colored_log_formatter)
        logger.addHandler(fh)

    return logger
