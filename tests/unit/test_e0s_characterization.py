"""Where the isolated-atom energies come from, and what happens when they do not.

E0s are the largest numbers in a MACE energy by two or three orders of
magnitude, and they enter in exactly one place: `AtomicEnergiesBlock` looks
them up with a one-hot matmul and the result is the first term of the readout
sum (`mace/modules/models.py:359` for plain MACE, `:581` for the scale-shift
class, where the E0 term is added *outside* the scale and the shift, which
were applied one line earlier). Everything
upstream of that lookup -- the four ways a user can supply E0s, and the
several ways the code quietly supplies its own -- is characterized here.

Three of the behaviours below are **silent fallbacks**: an isolated-atom
configuration with no energy becomes E0 = 0.0 with a `logging.warning`; a
NaN reference energy propagates NaN E0s with no log line at all; and a
species that never appears in the training set gets whatever the minimum-norm
least-squares solution hands back, which is 0.0. The rewrite is expected to
turn these into hard errors. Characterizing them first is the point: a
behaviour nobody wrote down cannot be deliberately changed.

The `"average"` fit and the foundation-`"estimated"` correction are both
asserted **exactly** against a hand-built `numpy.linalg.lstsq` on the same
design matrix -- the hand-built version is the executable specification, and
neither test involves a random seed.
"""

import json
import logging

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write

from mace import data
from mace.data.utils import (
    Configuration,
    KeySpecification,
    compute_average_E0s,
    config_from_atoms,
    estimate_e0s_from_foundation,
    load_from_xyz,
)
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.scripts_utils import get_atomic_energies
from tests.golden import harness
from tests.golden.anchors import anchor_graph, load_anchor, load_training_structures

TOL = harness.FP64_CPU_REFERENCE

Z_TABLE = utils.AtomicNumberTable([1, 6, 8])


def _config(atomic_numbers, energy):
    """A Configuration carrying nothing but what the E0 fit reads."""
    count = len(atomic_numbers)
    return Configuration(
        atomic_numbers=np.array(atomic_numbers),
        positions=np.zeros((count, 3)),
        properties={"energy": energy},
        property_weights={"energy": 1.0},
    )


def _design_matrix(configs, z_table):
    """The A and B a hand-written least squares would build. The spec."""
    matrix = np.zeros((len(configs), len(z_table)))
    target = np.zeros(len(configs))
    for row, config in enumerate(configs):
        target[row] = config.properties["energy"]
        for column, z in enumerate(z_table.zs):
            matrix[row, column] = np.count_nonzero(config.atomic_numbers == z)
    return matrix, target


# ---------------------------------------------------------------------------
# Isolated atoms in the training file
# ---------------------------------------------------------------------------


def _write_xyz(path, structures):
    write(path, structures, format="extxyz")
    return str(path)


def _isolated(symbol, energy=None):
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 6.0)
    atoms.info["config_type"] = "IsolatedAtom"
    if energy is not None:
        atoms.info["REF_energy"] = energy
    return atoms


def _molecule(symbols, energy):
    atoms = Atoms(
        symbols,
        positions=np.arange(3 * len(symbols)).reshape(-1, 3) * 0.5,
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    atoms.info["REF_energy"] = energy
    return atoms


def test_isolated_atom_configs_become_exact_e0s(tmp_path):
    """The first and most direct route: one config per element, tagged.

    Both conditions are required (`mace/data/utils.py:325-327`): exactly one
    atom *and* `config_type == "IsolatedAtom"`. A two-atom config carrying
    the tag is training data, not an E0.
    """
    path = _write_xyz(
        tmp_path / "train.xyz",
        [
            _isolated("H", -13.6),
            _isolated("O", -432.1),
            _molecule("H2O", -444.0),
        ],
    )
    e0s, configs = load_from_xyz(
        path, KeySpecification.from_defaults(), extract_atomic_energies=True
    )
    assert e0s == {1: -13.6, 8: -432.1}
    # and the isolated atoms are dropped from the training configurations
    assert len(configs) == 1


def test_an_isolated_atom_that_carries_no_energy_becomes_zero_with_a_warning(
    tmp_path, caplog
):
    """Silent fallback #1, and the one most likely to be hit by accident.

    A typo in `--energy_key` makes every isolated atom look energy-less, and
    the run then trains against E0 = 0 for that element -- a shift of
    hundreds of eV -- having logged one warning.
    """
    path = _write_xyz(
        tmp_path / "train.xyz",
        [_isolated("H", -13.6), _isolated("O"), _molecule("H2O", -444.0)],
    )
    with caplog.at_level(logging.WARNING):
        e0s, _ = load_from_xyz(
            path, KeySpecification.from_defaults(), extract_atomic_energies=True
        )
    assert e0s == {1: -13.6, 8: 0.0}
    assert any(
        "IsolatedAtom" in record.message and "Zero energy" in record.message
        for record in caplog.records
    ), caplog.text


def test_keep_isolated_atoms_leaves_them_in_the_training_set(tmp_path):
    path = _write_xyz(
        tmp_path / "train.xyz",
        [_isolated("H", -13.6), _isolated("O", -432.1), _molecule("H2O", -444.0)],
    )
    keyspec = KeySpecification.from_defaults()
    _, dropped = load_from_xyz(path, keyspec, extract_atomic_energies=True)
    _, kept = load_from_xyz(
        path,
        KeySpecification.from_defaults(),
        extract_atomic_energies=True,
        keep_isolated_atoms=True,
    )
    assert len(dropped) == 1
    assert len(kept) == 3


def test_a_single_atom_without_the_tag_is_not_an_e0(tmp_path):
    lonely = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 6.0)
    lonely.info["REF_energy"] = -13.6
    path = _write_xyz(tmp_path / "train.xyz", [lonely, _molecule("H2O", -444.0)])
    e0s, configs = load_from_xyz(
        path, KeySpecification.from_defaults(), extract_atomic_energies=True
    )
    assert e0s == {}
    assert len(configs) == 2


# ---------------------------------------------------------------------------
# "average": least squares over the training set
# ---------------------------------------------------------------------------


def test_average_e0s_are_exactly_a_least_squares_fit():
    """Not approximately: the same floats, bit for bit.

    `compute_average_E0s` builds a composition matrix and calls
    `numpy.linalg.lstsq(rcond=None)`. The hand-built version here is the
    executable specification a port is checked against; there is no seed and
    no tolerance, because there is nothing stochastic about it.
    """
    configs = [
        _config([1, 1, 8], -10.0),
        _config([6, 8, 8], -20.0),
        _config([1, 6, 1, 8], -15.0),
        _config([6, 6], -7.0),
    ]
    matrix, target = _design_matrix(configs, Z_TABLE)
    expected = np.linalg.lstsq(matrix, target, rcond=None)[0]

    got = compute_average_E0s(configs, Z_TABLE)
    assert sorted(got) == list(Z_TABLE.zs)
    for column, z in enumerate(Z_TABLE.zs):
        assert got[z] == expected[column], z


def test_average_e0s_reproduce_an_exactly_determined_system():
    """A case whose answer can be read off, so the fit is not self-referential.

    Three configurations, three elements, consistent by construction with
    E0(H) = -1, E0(C) = -2, E0(O) = -3: H2O -> -5, CO2 -> -8, CH4 -> -6.
    """
    configs = [
        _config([1, 1, 8], -5.0),
        _config([6, 8, 8], -8.0),
        _config([6, 1, 1, 1, 1], -6.0),
    ]
    got = compute_average_E0s(configs, Z_TABLE)
    assert got[1] == pytest.approx(-1.0, abs=TOL.atol)
    assert got[6] == pytest.approx(-2.0, abs=TOL.atol)
    assert got[8] == pytest.approx(-3.0, abs=TOL.atol)


def test_an_element_absent_from_the_training_set_gets_zero_silently():
    """Silent fallback #3. `lstsq` returns the minimum-norm solution.

    A z_table wider than the data -- exactly what `--foundation_model_elements`
    produces -- therefore yields E0 = 0.0 for the unseen elements, with no
    warning and no rank report.
    """
    configs = [_config([1, 1], -2.0), _config([1, 1, 1], -3.0)]
    got = compute_average_E0s(configs, Z_TABLE)
    assert got[6] == 0.0
    assert got[8] == 0.0


def test_a_nan_reference_energy_is_not_caught_by_the_linalgerror_fallback(caplog):
    """Silent fallback #2, and the platform-dependent one.

    `compute_average_E0s` wraps its `lstsq` in `except LinAlgError` and falls
    back to zeros. NaN input does not necessarily raise: on macOS/Accelerate
    it returns NaN E0s and the fallback never runs, so the training starts
    with NaN atomic energies and the first loss is NaN with nothing in the
    log to say why.

    The assertion is written so that it pins *this* platform's behaviour
    without asserting one LAPACK: either outcome is accepted, but the pairing
    is not -- a NaN result must come with no log line (that is what makes it
    silent), and a zeroed result must come with the error the fallback logs.
    """
    configs = [
        _config([1, 1, 8], float("nan")),
        _config([6, 8, 8], -20.0),
        _config([1, 6, 1, 8], -15.0),
    ]
    with caplog.at_level(logging.ERROR):
        got = compute_average_E0s(configs, Z_TABLE)
    values = np.array([got[z] for z in Z_TABLE.zs])
    logged = any("least squares" in record.message for record in caplog.records)
    if np.isnan(values).any():
        assert not logged, "a NaN result that logs is not the silent path"
    else:
        assert np.array_equal(values, np.zeros(3))
        assert logged, "the zeros fallback logs; nothing else may reach it"


# ---------------------------------------------------------------------------
# get_atomic_energies: the --E0s command line
# ---------------------------------------------------------------------------


def test_e0s_average_goes_through_the_least_squares_fit():
    configs = [_config([1, 1, 8], -10.0), _config([6, 8, 8], -20.0)]
    assert get_atomic_energies("average", configs, Z_TABLE) == compute_average_E0s(
        configs, Z_TABLE
    )


def test_e0s_average_without_a_training_set_is_a_runtime_error():
    with pytest.raises(RuntimeError, match="Could not compute average E0s"):
        get_atomic_energies("average", None, Z_TABLE)


def test_e0s_from_a_literal_dict_and_from_a_json_file(tmp_path):
    literal = get_atomic_energies("{1: -13.6, 8: -432.1}", None, Z_TABLE)
    assert literal == {1: -13.6, 8: -432.1}

    path = tmp_path / "e0s.json"
    path.write_text(json.dumps({"1": -13.6, "8": -432.1}), encoding="utf-8")
    from_json = get_atomic_energies(str(path), None, Z_TABLE)
    # the JSON route stringifies its keys and the loader casts them back
    assert from_json == literal
    assert all(isinstance(key, int) for key in from_json)


def test_an_unparseable_e0s_string_is_a_runtime_error():
    with pytest.raises(RuntimeError, match="E0s specified invalidly"):
        get_atomic_energies("not a dict", None, Z_TABLE)


def test_no_e0s_at_all_is_a_runtime_error():
    with pytest.raises(RuntimeError, match="E0s not found"):
        get_atomic_energies(None, None, Z_TABLE)


# ---------------------------------------------------------------------------
# The foundation-model "estimated" path
# ---------------------------------------------------------------------------


def _training_configs(limit=5):
    keyspec = KeySpecification.from_defaults()
    return [
        config_from_atoms(atoms, keyspec)
        for atoms in load_training_structures(limit=limit)
    ]


def test_estimated_e0s_are_the_foundation_e0s_plus_a_least_squares_correction():
    """The specification, again hand-built on the same design matrix.

    `estimate_e0s_from_foundation` runs the foundation model on every
    training configuration, fits the *residual* energy per element, and adds
    the fit to the foundation's own E0s. The committed anchor stands in for
    the foundation model, which is what makes this testable with no network
    and no download.
    """
    foundation = load_anchor("tiny_scaleshift", torch.float64)
    foundation_e0s = {1: -1.0, 6: -2.0, 8: -3.0}
    with torch_tools.default_dtype("float64"):
        got = estimate_e0s_from_foundation(
            foundation, foundation_e0s, _training_configs(), Z_TABLE, device="cpu"
        )

        configs = _training_configs()
        matrix = np.zeros((len(configs), len(Z_TABLE)))
        residual = np.zeros(len(configs))
        foundation_z_table = utils.AtomicNumberTable(
            [int(z) for z in foundation.atomic_numbers]
        )
        for row, config in enumerate(configs):
            graph = data.AtomicData.from_config(
                config, z_table=foundation_z_table, cutoff=float(foundation.r_max)
            )
            batch = next(
                iter(
                    torch_geometric.dataloader.DataLoader(
                        [graph], batch_size=1, shuffle=False
                    )
                )
            )
            with torch.no_grad():
                predicted = float(
                    foundation(
                        batch.to_dict(),
                        training=False,
                        compute_force=False,
                        compute_virials=False,
                        compute_stress=False,
                    )["energy"]
                )
            residual[row] = config.properties["energy"] - predicted
            for column, z in enumerate(Z_TABLE.zs):
                matrix[row, column] = np.sum(config.atomic_numbers == z)
    corrections = np.linalg.lstsq(matrix, residual, rcond=None)[0]

    for column, z in enumerate(Z_TABLE.zs):
        assert got[z] == foundation_e0s[z] + corrections[column], z


def test_estimated_e0s_are_deterministic_at_cpu_float64():
    """Two runs on freshly rebuilt config objects: bit-identical.

    Rebuilt rather than reused, so that a path which mutated its inputs
    would show up as a difference rather than being hidden by the second run
    reading the first one's leftovers.
    """
    foundation = load_anchor("tiny_scaleshift", torch.float64)
    foundation_e0s = {1: -1.0, 6: -2.0, 8: -3.0}
    with torch_tools.default_dtype("float64"):
        first = estimate_e0s_from_foundation(
            foundation, foundation_e0s, _training_configs(), Z_TABLE, device="cpu"
        )
        second = estimate_e0s_from_foundation(
            foundation, foundation_e0s, _training_configs(), Z_TABLE, device="cpu"
        )
    assert set(first) == set(second)
    for z, value in first.items():
        assert value == second[z], z


def test_estimated_e0s_fall_back_to_the_foundation_when_nothing_has_an_energy(
    caplog,
):
    foundation = load_anchor("tiny_scaleshift", torch.float64)
    foundation_e0s = {1: -1.0, 6: -2.0, 8: -3.0}
    configs = _training_configs(limit=2)
    for config in configs:
        config.properties["energy"] = None
    with caplog.at_level(logging.WARNING):
        got = estimate_e0s_from_foundation(
            foundation, foundation_e0s, configs, Z_TABLE, device="cpu"
        )
    assert got == foundation_e0s
    assert got is not foundation_e0s, "the fallback returns a copy"
    assert any("No configurations with energy" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Where the E0s land in the energy
# ---------------------------------------------------------------------------


@pytest.fixture(name="fixtures_water")
def fixture_water():
    atoms = harness.load_fixtures(names=["water_cluster"])["water_cluster"]
    return atoms, load_anchor("tiny_scaleshift", torch.float64)


def test_the_e0s_enter_the_total_energy_once_per_atom(fixtures_water):
    """Additive, exact, and outside every scaling.

    Adding a constant d to every element's E0 moves the total energy by
    n_atoms * d and nothing else -- the E0 term is a plain sum over the
    readout, not something the scale or the shift touches.
    """
    atoms, model = fixtures_water
    with torch_tools.default_dtype("float64"):
        before = float(model(anchor_graph(model, atoms), compute_force=False)["energy"])
        delta = 1.25
        model.atomic_energies_fn.atomic_energies = (
            model.atomic_energies_fn.atomic_energies + delta
        )
        after = float(model(anchor_graph(model, atoms), compute_force=False)["energy"])
    assert after - before == pytest.approx(len(atoms) * delta, abs=TOL.atol)


def test_the_e0_contribution_is_the_composition_dot_the_table(fixtures_water):
    """Read the whole E0 term off the composition, and subtract it.

    Zeroing the table has to remove exactly `sum_z count(z) * E0(z)`, which
    is what makes the E0s separable from the fitted part at all -- and what
    the `"average"` fit above assumes.
    """
    atoms, model = fixtures_water
    table = model.atomic_energies_fn.atomic_energies.detach().clone()
    numbers = [int(z) for z in model.atomic_numbers]
    expected = sum(
        float(torch.atleast_2d(table)[0][numbers.index(z)])
        for z in atoms.get_atomic_numbers()
    )
    with torch_tools.default_dtype("float64"):
        with_e0s = float(
            model(anchor_graph(model, atoms), compute_force=False)["energy"]
        )
        model.atomic_energies_fn.atomic_energies = torch.zeros_like(table)
        without = float(
            model(anchor_graph(model, atoms), compute_force=False)["energy"]
        )
    assert with_e0s - without == pytest.approx(expected, abs=TOL.atol)


# ---------------------------------------------------------------------------
# --mean / --std as explicit statistics overrides
#
# The inventory gap assigned to this ticket. These are the two knobs that
# bypass the dataset-statistics pass entirely, and all three behaviours below
# are surprising in the same direction: the flag is accepted and then ignored.
# ---------------------------------------------------------------------------


BASE_ARGV = [
    "--name",
    "characterization",
    "--train_file",
    "unused.xyz",
    "--model",
    "ScaleShiftMACE",
    "--r_max",
    "4.0",
    "--max_L",
    "0",
    "--num_channels",
    "4",
    "--hidden_irreps",
    "4x0e",
    "--num_interactions",
    "2",
    "--default_dtype",
    "float64",
]


def _configure(extra_argv):
    """Parse a real command line and build the model it asks for.

    `configure_model` is only ever called from `run_train.run()`, which sets
    the five `compute_*` flags itself before calling (mace/cli/run_train.py:
    618-620); they are reproduced here rather than faked, so the arguments
    under test are the parsed ones.
    """
    from mace.tools.arg_parser import build_default_arg_parser  # noqa: PLC0415
    from mace.tools.model_script_utils import configure_model  # noqa: PLC0415

    structures = load_training_structures(limit=4)
    atomic_energies = np.array([-0.1, -0.2, -0.3])
    with torch_tools.default_dtype("float64"):
        graphs = [
            data.AtomicData.from_config(
                config_from_atoms(atoms, KeySpecification.from_defaults()),
                z_table=Z_TABLE,
                cutoff=4.0,
            )
            for atoms in structures
        ]
        loader = torch_geometric.dataloader.DataLoader(
            graphs, batch_size=2, shuffle=False
        )
        args = build_default_arg_parser().parse_args(BASE_ARGV + extra_argv)
        args.compute_energy = True
        args.compute_dipole = False
        args.compute_polarizability = False
        args.compute_magforces = False
        model, _ = configure_model(
            args, loader, atomic_energies, heads=["Default"], z_table=Z_TABLE
        )
    return model


def test_mean_and_std_together_override_the_dataset_statistics():
    """`--std` becomes the scale and `--mean` the shift, verbatim."""
    model = _configure(["--mean", "1.5", "--std", "2.5"])
    assert float(model.scale_shift.scale) == pytest.approx(2.5)
    assert float(model.scale_shift.shift) == pytest.approx(1.5)
    # and they are not what the data would have produced
    computed = _configure([])
    assert float(computed.scale_shift.shift) != pytest.approx(1.5)


def test_mean_without_std_is_accepted_and_then_discarded():
    """Both are recomputed unless *both* are given.

    The guard is `if args.mean is None or args.std is None` (mace/tools/
    model_script_utils.py:78), so a run that sets only one of them silently
    gets the dataset statistics for both. Measured here: `--mean 1.5` gives a
    shift of 0.175, which is the fitted value.
    """
    partial = _configure(["--mean", "1.5"])
    computed = _configure([])
    assert float(partial.scale_shift.shift) == float(computed.scale_shift.shift)
    assert float(partial.scale_shift.shift) != pytest.approx(1.5)


def test_no_scaling_overrides_even_an_explicit_std():
    """`--scaling no_scaling` sets std to 1.0 before anything else looks.

    So `--scaling no_scaling --std 7.0` trains with a scale of 1.0 and no
    warning. The mean is untouched, which is the asymmetry.
    """
    model = _configure(["--scaling", "no_scaling", "--std", "7.0", "--mean", "1.5"])
    assert float(model.scale_shift.scale) == pytest.approx(1.0)
    assert float(model.scale_shift.shift) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# The dataset statistics the scale and the shift come from
#
# These are the numbers `--mean` and `--std` replace, and they are computed
# from the E0s: every one of them subtracts the atomic energies first, so an
# E0 mistake propagates into the scale as well as into the energies. Each is
# asserted against the same arithmetic done in numpy on the same structures.
# ---------------------------------------------------------------------------


def _labelled_loader(structures, batch_size=2, cutoff=4.0):
    keyspec = KeySpecification.from_defaults()
    with torch_tools.default_dtype("float64"):
        graphs = [
            data.AtomicData.from_config(
                config_from_atoms(atoms, keyspec), z_table=Z_TABLE, cutoff=cutoff
            )
            for atoms in structures
        ]
        return torch_geometric.dataloader.DataLoader(
            graphs, batch_size=batch_size, shuffle=False
        )


def _per_atom_interaction_energies(structures, e0s):
    """The same quantity in numpy: (E - sum_z n_z E0_z) / n_atoms."""
    values = []
    for atoms in structures:
        e0 = sum(e0s[Z_TABLE.z_to_index(int(z))] for z in atoms.get_atomic_numbers())
        values.append((atoms.info["REF_energy"] - e0) / len(atoms))
    return np.array(values)


def test_the_mean_and_std_scaling_is_the_per_atom_interaction_energy():
    """`--scaling std_scaling`, which is the default.

    The mean is the mean per-atom interaction energy and the std is its
    sample standard deviation (Bessel-corrected -- `scatter_std` defaults to
    `unbiased=True`, which differs from the population value by a factor
    sqrt(n/(n-1)) and is the kind of detail a port gets wrong silently).
    """
    from mace.modules.utils import (  # noqa: PLC0415
        compute_mean_std_atomic_inter_energy,
    )

    structures = load_training_structures(limit=6)
    e0s = np.array([-0.1, -0.2, -0.3])
    with torch_tools.default_dtype("float64"):
        mean, std = compute_mean_std_atomic_inter_energy(
            _labelled_loader(structures), e0s
        )
    expected = _per_atom_interaction_energies(structures, e0s)
    assert float(np.atleast_1d(mean)[0]) == pytest.approx(
        expected.mean(), abs=TOL.atol
    )
    assert float(np.atleast_1d(std)[0]) == pytest.approx(
        expected.std(ddof=1), abs=TOL.atol
    )
    assert float(np.atleast_1d(std)[0]) != pytest.approx(
        expected.std(ddof=0), abs=TOL.atol
    ), "ddof=0 would also be plausible; it is not what the code does"


def test_the_rms_forces_scaling_is_the_root_mean_square_force_component():
    """`--scaling rms_forces_scaling`: the shift is an energy, the scale is a force.

    Two different physical quantities in one pair, which is why `--mean` and
    `--std` have the help strings they do.
    """
    from mace.modules.utils import compute_mean_rms_energy_forces  # noqa: PLC0415

    structures = load_training_structures(limit=6)
    e0s = np.array([-0.1, -0.2, -0.3])
    with torch_tools.default_dtype("float64"):
        mean, rms = compute_mean_rms_energy_forces(_labelled_loader(structures), e0s)
    expected_mean = _per_atom_interaction_energies(structures, e0s).mean()
    forces = np.concatenate([atoms.arrays["REF_forces"] for atoms in structures])
    assert float(np.atleast_1d(mean)[0]) == pytest.approx(expected_mean, abs=TOL.atol)
    assert float(np.atleast_1d(rms)[0]) == pytest.approx(
        np.sqrt(np.mean(forces**2)), abs=TOL.atol
    )


def test_a_zero_spread_becomes_a_scale_of_one_with_a_warning(caplog):
    """Silent fallback #4, in the statistics rather than in the E0s.

    A single training configuration has no spread, so the std is 0 and
    dividing by it would be fatal. `_check_non_zero` replaces it with 1.0 and
    logs -- so a one-config debug run trains unscaled and looks normal.
    """
    from mace.modules.utils import (  # noqa: PLC0415
        compute_mean_std_atomic_inter_energy,
    )

    structures = load_training_structures(limit=1)
    with torch_tools.default_dtype("float64"):
        with caplog.at_level(logging.WARNING):
            _, std = compute_mean_std_atomic_inter_energy(
                _labelled_loader(structures, batch_size=1), np.zeros(3)
            )
    assert float(np.atleast_1d(std)[0]) == 1.0
    assert any("Standard deviation" in record.message for record in caplog.records)


def test_the_average_neighbour_count_skips_atoms_that_have_no_neighbours():
    """A counting convention with a real effect on the model's normalisation.

    `compute_avg_num_neighbors` counts with `torch.unique(receivers)`, so an
    atom that appears in no edge contributes nothing at all -- it is not
    counted as a zero. Adding an isolated atom to the batch therefore leaves
    the average unchanged instead of pulling it down, which is a difference a
    port that counts per node rather than per edge would get wrong.
    """
    from mace.modules.utils import compute_avg_num_neighbors  # noqa: PLC0415

    structures = load_training_structures(limit=3)
    lonely = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 50.0, pbc=True)
    lonely.info["REF_energy"] = 0.0
    lonely.arrays["REF_forces"] = np.zeros((1, 3))

    with torch_tools.default_dtype("float64"):
        without = compute_avg_num_neighbors(_labelled_loader(structures))
        with_lonely = compute_avg_num_neighbors(_labelled_loader(structures + [lonely]))
    assert without > 1.0
    assert with_lonely == pytest.approx(without, abs=TOL.atol)


def test_compute_statistics_returns_three_numbers_and_the_third_is_not_a_std():
    """What `mace_prepare_data` writes into statistics.json, and its one lie.

    The signature says `Tuple[float, float, float, float]` and the function
    returns **three** values; `preprocess_data.py:43` unpacks them as
    `avg_num_neighbors, mean, std`, but the third is the root-mean-square
    force component, not a standard deviation of anything. It agrees with
    `compute_mean_rms_energy_forces`, which is what `--scaling
    rms_forces_scaling` (the default) uses, so the number is the right one
    for the default path and misnamed everywhere it is read.
    """
    from mace.modules.utils import (  # noqa: PLC0415
        compute_avg_num_neighbors,
        compute_mean_rms_energy_forces,
        compute_mean_std_atomic_inter_energy,
        compute_statistics,
    )

    structures = load_training_structures(limit=6)
    e0s = np.array([-0.1, -0.2, -0.3])
    with torch_tools.default_dtype("float64"):
        returned = compute_statistics(_labelled_loader(structures), e0s)
        expected_neighbors = compute_avg_num_neighbors(_labelled_loader(structures))
        expected_mean, expected_rms = compute_mean_rms_energy_forces(
            _labelled_loader(structures), e0s
        )
        _, actual_std = compute_mean_std_atomic_inter_energy(
            _labelled_loader(structures), e0s
        )
    assert len(returned) == 3, "the annotation says four; the code returns three"
    avg_neighbors, mean, third = returned
    assert avg_neighbors == pytest.approx(expected_neighbors, abs=TOL.atol)
    assert float(np.atleast_1d(mean)[0]) == pytest.approx(
        float(np.atleast_1d(expected_mean)[0]), abs=TOL.atol
    )
    assert float(np.atleast_1d(third)[0]) == pytest.approx(
        float(np.atleast_1d(expected_rms)[0]), abs=TOL.atol
    )
    assert float(np.atleast_1d(third)[0]) != pytest.approx(
        float(np.atleast_1d(actual_std)[0]), abs=TOL.atol
    ), "the two are different quantities; if they coincide the test is vacuous"


def test_the_dipole_scaling_entry_cannot_be_used_the_way_the_other_two_are():
    """`scaling_classes["rms_dipoles_scaling"]` does not have the same shape.

    The other two entries return `(mean, std)` and `configure_model` unpacks
    the call site as such (mace/tools/model_script_utils.py:81). This one
    returns a single float, so that unpacking raises TypeError. It is not
    reachable from `--scaling`, whose choices are the other three, so the
    entry is only usable through a YAML config -- where it fails. Pinned as
    characterization: a port that "fixes" the return shape changes what a
    config file does, and a port that copies the registry faithfully
    inherits a dead entry it should know about.
    """
    from mace.modules import scaling_classes  # noqa: PLC0415
    from mace.modules.utils import compute_rms_dipoles  # noqa: PLC0415

    structures = load_training_structures(limit=4)
    for index, atoms in enumerate(structures):
        atoms.info["dipole"] = np.array([0.5 * (index + 1), 0.0, -1.0])
    with torch_tools.default_dtype("float64"):
        returned = compute_rms_dipoles(_labelled_loader(structures))
    dipoles = np.stack([atoms.info["dipole"] for atoms in structures])
    assert isinstance(returned, float)
    assert returned == pytest.approx(np.sqrt(np.mean(dipoles**2)), abs=TOL.atol)
    assert scaling_classes["rms_dipoles_scaling"] is compute_rms_dipoles
    with pytest.raises(TypeError):
        _mean, _std = returned  # what configure_model does


def test_the_zero_spread_guard_only_works_on_arrays():
    """`_check_non_zero` assigns into its argument, so a scalar 0.0 raises.

    The two array-returning statistics reach it safely; `compute_rms_dipoles`
    hands it a Python float, so the guard that exists to prevent a division
    by zero is itself a TypeError in exactly the case it was written for.
    """
    from mace.modules.utils import _check_non_zero  # noqa: PLC0415

    assert _check_non_zero(np.array([0.0, 2.0])).tolist() == [1.0, 2.0]
    assert _check_non_zero(3.5) == 3.5
    with pytest.raises(TypeError):
        _check_non_zero(0.0)


def test_the_per_batch_statistics_helpers_agree_with_the_loader_versions():
    """`_compute_mean_std_atomic_inter_energy` and its rms sibling.

    They exist so a caller holding one batch can get the same per-graph
    quantities the loader-level functions reduce -- the HDF5/LMDB
    preprocessing path is what needs that. Nothing pins them together, so
    this does: fed the whole dataset as one batch, the private helper's
    per-graph energies reduce to exactly what the public function returns.
    """
    from mace.modules.blocks import AtomicEnergiesBlock  # noqa: PLC0415
    from mace.modules.utils import (  # noqa: PLC0415
        _compute_mean_rms_energy_forces,
        _compute_mean_std_atomic_inter_energy,
        compute_mean_std_atomic_inter_energy,
    )

    structures = load_training_structures(limit=6)
    e0s = np.array([-0.1, -0.2, -0.3])
    with torch_tools.default_dtype("float64"):
        loader = _labelled_loader(structures, batch_size=len(structures))
        batch = next(iter(loader))
        block = AtomicEnergiesBlock(atomic_energies=e0s)
        per_graph = _compute_mean_std_atomic_inter_energy(batch, block)
        also_per_graph, forces = _compute_mean_rms_energy_forces(batch, block)
        mean, std = compute_mean_std_atomic_inter_energy(
            _labelled_loader(structures), e0s
        )
    expected = _per_atom_interaction_energies(structures, e0s)
    assert per_graph.detach().numpy() == pytest.approx(expected, abs=TOL.atol)
    assert torch.equal(per_graph, also_per_graph)
    assert forces.shape[0] == sum(len(atoms) for atoms in structures)
    assert float(np.atleast_1d(mean)[0]) == pytest.approx(
        float(per_graph.mean()), abs=TOL.atol
    )
    assert float(np.atleast_1d(std)[0]) == pytest.approx(
        float(per_graph.std(unbiased=True)), abs=TOL.atol
    )
