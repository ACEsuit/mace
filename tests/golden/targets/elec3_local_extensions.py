"""Initial LSC oxidation and direct response references."""

from tests.golden.targets._electrostatics_oracle import run_capture

ORDER = 42
HELP = "LSC oxidation mixers, charge transfer and polarizability references"
IN_ALL = False


def run():
    run_capture("tests.parity.elec3_local_extensions")
