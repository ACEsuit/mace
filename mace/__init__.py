import os

from .__version__ import __version__

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

__all__ = ["__version__"]
