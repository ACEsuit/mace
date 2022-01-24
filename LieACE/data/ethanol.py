import logging
import os
from typing import Dict

from .utils import Configurations, unpack_configs_from_archive

# Atomic energies (in eV)
atomic_energies = {
    1: -13.568422383046626,
    6: -1025.2770951782686,
    8: -2035.5709809589698,
}


def load(directory: str) -> Dict[str, Configurations]:
    logging.info('Loading ethanol dataset')
    path = os.path.join(directory, 'dataset_ethanol.tar.gz')
    return unpack_configs_from_archive(path=path)