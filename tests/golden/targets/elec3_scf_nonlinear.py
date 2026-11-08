"""Initial nonlinear FixedPointSCF references; existing families stay frozen."""

from tests.golden.targets._electrostatics_oracle import run_capture

ORDER = 41
HELP = "Nonlinear FixedPointSCF initial references from pinned mace-scf"
IN_ALL = False


def run():
    run_capture("tests.parity.elec3_scf_nonlinear")
