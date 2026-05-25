import os
import warnings

from .__version__ import __version__

warnings.filterwarnings(
    "ignore",
    message=".*TorchScript type system.*instance-level annotations.*",
)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
