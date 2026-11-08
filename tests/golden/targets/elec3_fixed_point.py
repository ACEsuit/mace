"""Initial affine FixedPointSCF references, separate from all existing targets."""

from tests.golden.targets._electrostatics_oracle import run_capture

ORDER = 40
HELP = "Affine FixedPointSCF initial references from pinned mace-scf"
IN_ALL = False


def run():
    run_capture("tests.parity.elec3_fixed_point")
