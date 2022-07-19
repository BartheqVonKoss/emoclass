from pathlib import Path
from typing import List, Union


class TrainingConfig:
    """Hold variables regarding training setup."""
    # Data to use while training part
    DATAFILE: Path = Path("data/train.tsv")

    EMOTIONS_FILE: Path = Path("data/emotions.txt")
    LABELS_FILE: Path = None #Path("data/")
    # EVAL_DATAFILE: Path 
    # EVAL_DATAPATH: List[Path] = [Path("data/val_synthetic")]  # a list of paths to validation data
    # "labels_file.csv" is automatically genereated and will be added

    NAME = Path("default")
    TB_DIR = Path("workdir/tensorboard") / NAME
    CHECKPOINTS_DIR = Path("workdir/checkpoints") / NAME
    LOGS_DIR = Path("workdir")

    # General part
    BATCH_SIZE = 32
    REG_FACTOR = 1e-5
    LR = 1e-3
    LR_DECAY = 0.998
    LOSS_NAME = "mse"  # "l1"

    # Metrics part
    EVAL_FREQ = 10
    SAVE_FREQ = 50

    # Resume training part
    START_EPOCH: int = None
    MAX_EPOCHS: int = 500

    MODEL_PATH: Path = None  # Path("path/to/model.pt")
    MODEL_TO_USE: str = "one_fcn2"


def get_training_config() -> TrainingConfig:
    return TrainingConfig()
