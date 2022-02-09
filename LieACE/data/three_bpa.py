import logging
import os
from typing import Dict

from .utils import Configurations, unpack_configs_from_archive

# "Linear Atomic Cluster Expansion Force Fields for Organic Molecules: beyond RMSE"
# Kovacs, D. P.; Oord, C. van der; Kucera, J.; Allen, A.; Cole, D.; Ortner, C.; Csanyi, G. 2021.
# https://doi.org/10.33774/chemrxiv-2021-7qlf5-v3

# Atomic energies (in eV)
atomic_energies = {
    1: -13.587222780835477,
    6: -1029.4889999855063,
    7: -1484.9814568572233,
    8: -2041.9816003861047,
}


def load(directory: str) -> Dict[str, Configurations]:
    logging.info('Loading 3BPA dataset')
    path = os.path.join(directory, 'dataset_3BPA.tar.gz')
    return unpack_configs_from_archive(path=path)