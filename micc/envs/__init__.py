from absl import flags
from micc.envs.smac.smac_logger import SMACLogger


FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "smac": SMACLogger,
}
