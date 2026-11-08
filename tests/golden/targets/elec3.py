"""Explicit capture of the initial LC/LSC external-oracle goldens."""

from tests.golden.targets._electrostatics_oracle import PINS as PINS
from tests.golden.targets._electrostatics_oracle import run_capture

ORDER = 39
HELP = "LC/LSC initial references from pinned mace-scf and MACE 0.3.14"
IN_ALL = False


def run():
    run_capture("tests.parity.elec3")
