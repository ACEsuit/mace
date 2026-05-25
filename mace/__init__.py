import os
import warnings

from .__version__ import __version__

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.jit",
    message=r"The TorchScript type system doesn't support instance-level annotations "
    r"on empty non-base types in `__init__`",
)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
