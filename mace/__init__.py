import torch

from .__version__ import __version__

# e3nn 0.4.4 (the pinned version) loads its Wigner constants with
# ``torch.load`` and no ``weights_only`` argument. That file stores ``slice``
# objects, which PyTorch >= 2.6's safe unpickler rejects by default. Allowlist
# that single harmless type instead of disabling safe loading process-wide via
# ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD``; MACE's own whole-pickle checkpoints
# pass ``weights_only=False`` at the load site that needs it.
torch.serialization.add_safe_globals([slice])
