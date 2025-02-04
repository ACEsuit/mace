from .__version__ import __version__

import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
