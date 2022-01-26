import logging
import os
from typing import Dict
from LieACE.data.utils import unpack_configs_from_archive

from LieACE.tools.config import Configurations


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