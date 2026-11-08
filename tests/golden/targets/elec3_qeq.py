"""Initial QEq reference; never regenerates the LC/LSC artifacts."""

from tests.golden.targets._electrostatics_oracle import run_capture

ORDER = 39
HELP = "QEq initial references from pinned mace-scf and MACE 0.3.14"
IN_ALL = False


def run():
    run_capture("tests.parity.elec3_qeq")
