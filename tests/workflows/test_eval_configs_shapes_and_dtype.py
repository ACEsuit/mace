"""Two things `mace_eval_configs` could not do, both fixed by copying itself.

`--return_node_energies` accumulated one entry per *batch* and then built a
rectangular array out of it, so it worked only while every structure had the
same number of atoms. `--descriptors`, four lines above in the same function,
already accumulated one entry per structure precisely because its shapes vary.

`--default_dtype` reached the forward pass without being reconciled against the
checkpoint, so a mismatch died inside a scripted tensor product on "both inputs
should have same dtype", naming neither the flag nor the model. The ase
calculator, given the same request, warns and converts.

Both tests therefore assert agreement with something already in the tree rather
than a number of their own.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from tests.helpers import REPO_ROOT, run_mace_train

EVAL_CONFIGS = REPO_ROOT / "mace" / "cli" / "eval_configs.py"


def _water(count):
    """`count` water molecules in one cell, so structures differ in atom count."""
    positions, symbols = [], []
    for index in range(count):
        origin = np.array([4.0 * index, 0.0, 0.0])
        positions += [origin, origin + [0.95, 0, 0], origin + [-0.24, 0.93, 0]]
        symbols += ["O", "H", "H"]
    return Atoms(
        symbols, positions=np.array(positions), cell=[6 + 4 * count, 8, 8], pbc=True
    )


@pytest.fixture(name="mixed_configs")
def fixture_mixed_configs(tmp_path):
    path = tmp_path / "mixed.xyz"
    write(path, [_water(1), _water(2), _water(1)])   # 3, 6, 3 atoms
    return path


@pytest.fixture(name="uniform_configs")
def fixture_uniform_configs(tmp_path):
    path = tmp_path / "uniform.xyz"
    write(path, [_water(1), _water(1), _water(1)])   # 3, 3, 3 atoms
    return path


def _evaluate(model, configs, output, **flags):
    params = {
        "configs": str(configs),
        "model": str(model),
        "output": str(output),
        "device": "cpu",
    }
    params.update(flags)
    result = run_mace_train(
        params, script=EVAL_CONFIGS, capture_output=True, text=True
    )
    return read(output, index=":"), result.stdout + result.stderr


@pytest.mark.parametrize("fixture_name", ["uniform_configs", "mixed_configs"])
def test_node_energies_are_written_per_structure_whatever_its_size(
    tmp_path, trained_tiny_model_path, fixture_name, request
):
    """Asserted by value, not by exit code: each structure's per-atom energies
    have that structure's length and sum to the total reported for it."""
    configs = request.getfixturevalue(fixture_name)
    frames, _ = _evaluate(
        trained_tiny_model_path,
        configs,
        tmp_path / f"{fixture_name}_out.xyz",
        default_dtype="float64",
        return_node_energies=None,
    )

    for atoms in frames:
        node_energies = atoms.arrays["MACE_node_energies"]
        assert node_energies.shape == (len(atoms),)
        assert np.isclose(node_energies.sum(), atoms.info["MACE_energy"])


def test_a_dtype_that_disagrees_with_the_checkpoint_is_converted_not_fatal(
    tmp_path, trained_tiny_model_path, uniform_configs
):
    """The defect: this used to die inside a scripted tensor product on "both
    inputs should have same dtype", naming neither the flag nor the model.

    What is asserted exactly is that it now runs and that it took the conversion
    path, by the warning it emits. The energies are only compared loosely, and
    deliberately so: the CLI runs in a subprocess while the calculator runs in
    this one, and float32 kernel selection depends on the shapes a process has
    already seen, so the two agree to about a float32 ulp rather than to the bit.
    An earlier version of this test asserted equality and failed in CI at exactly
    that: 1.1e-7 relative, against a float32 eps of 1.19e-7.
    """
    from mace.calculators import MACECalculator

    frames, output = _evaluate(
        trained_tiny_model_path,
        uniform_configs,
        tmp_path / "f32.xyz",
        default_dtype="float32",
    )

    assert "does not match model dtype" in output, (
        "the run succeeded without converting, so this no longer exercises the "
        "mismatch:\n" + output[-2000:]
    )

    atoms = read(uniform_configs, index=0)
    atoms.calc = MACECalculator(
        model_paths=str(trained_tiny_model_path), device="cpu", default_dtype="float32"
    )
    assert float(frames[0].info["MACE_energy"]) == pytest.approx(
        float(atoms.get_potential_energy()), rel=1e-5
    )
