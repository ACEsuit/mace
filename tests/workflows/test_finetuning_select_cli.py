"""`mace_finetuning_select` through its command line.

`tests/workflows/test_finetuning_select.py` builds `SelectionSettings` in-process,
which covers the selection logic and leaves the CLI itself unexercised: nothing
passed `--model`, `--output`, `--device`, `--default_dtype`, `--seed` or the YAML
`--config`, so a flag could stop reaching the code it names and every existing
test would still pass.

Two of those flags are only observable through the descriptor pass, so these tests
go through `--subselect fps`. With `fpsample` absent the sampling step logs
"FPS failed, selecting random configurations instead" and falls back, but the
model is still loaded and the descriptors are still computed and saved first,
which is the part that pins `--model` and `--default_dtype`. That fallback is
itself worth pinning: a user who asked for farthest-point sampling and silently
got random selection has no other signal than a log line.
"""

import shutil
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms

from tests.helpers import run_mace_train

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECT = REPO_ROOT / "mace" / "cli" / "fine_tuning_select.py"


@pytest.fixture(name="pool")
def fixture_pool(tmp_path):
    """A pretraining pool to select from, written where the CLI can read it."""
    rng = np.random.default_rng(0)
    configs = []
    for _ in range(8):
        atoms = Atoms(
            numbers=[8, 1, 1],
            positions=rng.random((3, 3)) * 3,
            cell=[5, 5, 5],
            pbc=[True] * 3,
        )
        atoms.info["REF_energy"] = float(rng.normal())
        atoms.arrays["REF_forces"] = rng.normal(size=(3, 3))
        configs.append(atoms)
    path = tmp_path / "pt.xyz"
    ase.io.write(path, configs, format="extxyz")
    return path


@pytest.fixture(name="model")
def fixture_model(trained_tiny_model_path, tmp_path):
    """A local checkpoint, so `--model` is exercised without a download.

    Its default is the string "small", which resolves to a foundation model over
    the network; a test that took the default would be testing the download.
    """
    destination = tmp_path / "tiny.model"
    shutil.copy(trained_tiny_model_path, destination)
    return destination


def select(tmp_path, pool, **flags):
    params = {"configs_pt": str(pool), "output": str(tmp_path / "out.xyz")}
    params.update({k: str(v) for k, v in flags.items()})
    return run_mace_train(
        params,
        script=SELECT,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def read_selection(tmp_path, name="out.xyz"):
    selected = ase.io.read(tmp_path / name, index=":")
    return [tuple(a.get_positions().round(8).ravel()) for a in selected]


# ---------------------------------------------------------------------------
# --output, --num_samples
# ---------------------------------------------------------------------------


def test_the_cli_writes_the_selection_and_the_combined_file(tmp_path, pool):
    select(tmp_path, pool, num_samples=3, subselect="random")

    assert (tmp_path / "out.xyz").exists()
    assert (tmp_path / "out_combined.xyz").exists(), "the combined file is part of the contract"
    assert len(read_selection(tmp_path)) == 3


def test_the_output_path_is_where_the_flag_says(tmp_path, pool):
    """`--output` also decides the combined file and the descriptor npy names."""
    elsewhere = tmp_path / "nested"
    elsewhere.mkdir()
    target = elsewhere / "chosen.xyz"

    run_mace_train(
        {
            "configs_pt": str(pool),
            "output": str(target),
            "num_samples": "2",
            "subselect": "random",
        },
        script=SELECT,
        cwd=tmp_path,
    )

    assert target.exists()
    assert (elsewhere / "chosen_combined.xyz").exists()


def test_an_output_that_is_not_extxyz_is_refused(tmp_path, pool):
    result = run_mace_train(
        {
            "configs_pt": str(pool),
            "output": str(tmp_path / "out.traj"),
            "num_samples": "2",
            "subselect": "random",
        },
        script=SELECT,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


# ---------------------------------------------------------------------------
# --seed
# ---------------------------------------------------------------------------


def test_the_same_seed_selects_the_same_configurations(tmp_path, pool):
    """Reproducibility is the whole point of the flag, and random selection is
    the mode where it is observable."""
    select(tmp_path, pool, num_samples=3, subselect="random", seed=11)
    first = read_selection(tmp_path)
    (tmp_path / "out.xyz").unlink()
    select(tmp_path, pool, num_samples=3, subselect="random", seed=11)

    assert read_selection(tmp_path) == first


def test_a_different_seed_selects_differently(tmp_path, pool):
    """Otherwise the flag could be read and dropped and the test above would
    still pass."""
    select(tmp_path, pool, num_samples=3, subselect="random", seed=11)
    first = read_selection(tmp_path)
    (tmp_path / "out.xyz").unlink()
    select(tmp_path, pool, num_samples=3, subselect="random", seed=12)

    assert read_selection(tmp_path) != first


# ---------------------------------------------------------------------------
# --model, --device, --default_dtype: the descriptor pass
# ---------------------------------------------------------------------------


def test_a_local_checkpoint_is_used_for_the_descriptors(tmp_path, pool, model):
    """`--model` pointing at a file, rather than the "small" default that would
    reach for the network."""
    select(tmp_path, pool, num_samples=3, subselect="fps", model=model, device="cpu")

    descriptors = tmp_path / "pt_descriptors.npy"
    assert descriptors.exists(), "the model was never asked for descriptors"
    assert len(read_selection(tmp_path)) == 3


def test_a_model_path_that_does_not_exist_is_refused(tmp_path, pool):
    """The counterpart: the flag is read, so a bad value has to fail."""
    result = run_mace_train(
        {
            "configs_pt": str(pool),
            "output": str(tmp_path / "out.xyz"),
            "num_samples": "2",
            "subselect": "fps",
            "model": str(tmp_path / "absent.model"),
        },
        script=SELECT,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_descriptors_follow_the_requested_dtype(tmp_path, pool, model, dtype):
    """`--default_dtype` reaches the calculator, observed where it lands: the
    saved descriptors are float32 or float64 accordingly."""
    select(
        tmp_path,
        pool,
        num_samples=3,
        subselect="fps",
        model=model,
        device="cpu",
        default_dtype=dtype,
    )

    saved = np.load(tmp_path / "pt_descriptors.npy", allow_pickle=True)
    first = saved[0]
    array = first[sorted(first)[0]] if isinstance(first, dict) else first

    assert np.asarray(array).dtype == np.dtype(dtype)


def test_a_device_outside_the_choices_is_rejected(tmp_path, pool):
    """`--device` is a `choices=` argument, so a typo must not reach torch."""
    result = run_mace_train(
        {
            "configs_pt": str(pool),
            "output": str(tmp_path / "out.xyz"),
            "num_samples": "2",
            "subselect": "random",
            "device": "cpu0",
        },
        script=SELECT,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_fps_without_fpsample_falls_back_and_says_so(tmp_path, pool, model):
    """The degradation is silent in the result and loud only in the log, so this
    pins the log. A user who asked for farthest-point sampling and received a
    random subset has nothing else to go on."""
    result = select(
        tmp_path, pool, num_samples=3, subselect="fps", model=model, device="cpu"
    )
    combined = result.stdout + result.stderr

    try:
        import fpsample  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        assert "FPS failed" in combined
        assert "selecting random configurations instead" in combined
    else:
        assert "FPS failed" not in combined


# ---------------------------------------------------------------------------
# --config
# ---------------------------------------------------------------------------


def test_the_yaml_config_is_read(tmp_path, pool):
    """configargparse's `--config`, which is the documented way to drive this CLI
    from a file. Same settings through YAML must give the same selection."""
    pytest.importorskip("configargparse")
    select(tmp_path, pool, num_samples=3, subselect="random", seed=5)
    by_flags = read_selection(tmp_path)
    (tmp_path / "out.xyz").unlink()

    config = tmp_path / "select.yaml"
    config.write_text(
        f"configs_pt: {pool}\n"
        f"output: {tmp_path / 'out.xyz'}\n"
        "num_samples: 3\n"
        "subselect: random\n"
        "seed: 5\n"
    )
    run_mace_train({"config": str(config)}, script=SELECT, cwd=tmp_path)

    assert read_selection(tmp_path) == by_flags


# ---------------------------------------------------------------------------
# --head_pt, --head_ft, --weight_pt, --weight_ft, --filter_atomic_numbers_pt
# ---------------------------------------------------------------------------


@pytest.fixture(name="mixed_pool")
def fixture_mixed_pool(tmp_path):
    """A pool of two chemistries, so the element filter has something to remove."""
    rng = np.random.default_rng(1)
    configs = []
    for symbols, positions in (
        ("H2O", [[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]]),
        ("H2O", [[0, 0, 0], [0.98, 0, 0], [-0.26, 0.95, 0]]),
        ("CH4", [[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]]),
        ("CH4", [[0, 0, 0], [1.10, 0, 0], [-0.37, 1.04, 0], [-0.37, -0.52, 0.90], [-0.37, -0.52, -0.90]]),
    ):
        for _ in range(2):
            atoms = Atoms(symbols, positions=positions, cell=[9, 9, 9], pbc=[True] * 3)
            atoms.positions += rng.normal(0, 0.02, atoms.positions.shape)
            atoms.info["REF_energy"] = float(rng.normal())
            atoms.arrays["REF_forces"] = rng.normal(0, 0.1, atoms.positions.shape)
            configs.append(atoms)
    path = tmp_path / "mixed.xyz"
    ase.io.write(path, configs, format="extxyz")
    return path


def written(tmp_path, name="out.xyz"):
    return ase.io.read(tmp_path / name, index=":")


def combined(tmp_path):
    return ase.io.read(tmp_path / "out_combined.xyz", index=":")


def test_the_pretraining_weight_is_written_onto_the_selection(tmp_path, pool):
    """`--weight_pt` ends up in `atoms.info["config_weight"]`, which is what the
    training loss reads, so a flag that stopped arriving would quietly weight the
    replay data at 1.0."""
    select(tmp_path, pool, num_samples=3, subselect="random", weight_pt=0.25)

    assert {atoms.info["config_weight"] for atoms in written(tmp_path)} == {0.25}


def test_the_selection_is_marked_as_pretraining(tmp_path, pool):
    """The other half of the same metadata: the flag is only meaningful because
    the configurations it lands on are labelled as the pretraining set."""
    select(tmp_path, pool, num_samples=3, subselect="random", weight_pt=0.25)

    assert all(atoms.info["pretrained"] for atoms in written(tmp_path))


def test_the_finetuning_weight_applies_to_the_finetuning_configs(tmp_path, pool):
    """`--weight_ft` is written onto `--configs_ft`, not onto the selection, and
    the two arrive together in the combined file. Checked there, because that is
    the file training consumes."""
    select(
        tmp_path,
        pool,
        num_samples=2,
        subselect="random",
        configs_ft=pool,
        weight_pt=0.25,
        weight_ft=0.75,
    )

    weights = {
        (atoms.info["pretrained"], atoms.info["config_weight"])
        for atoms in combined(tmp_path)
    }
    assert (True, 0.25) in weights
    assert (False, 0.75) in weights


def test_the_default_weights_are_one(tmp_path, pool):
    select(tmp_path, pool, num_samples=3, subselect="random")

    assert {atoms.info["config_weight"] for atoms in written(tmp_path)} == {1.0}


def test_the_heads_label_each_half(tmp_path, pool):
    """`--head_pt` and `--head_ft` name the level of theory each set belongs to,
    which is how multihead training tells them apart."""
    select(
        tmp_path,
        pool,
        num_samples=2,
        subselect="random",
        configs_ft=pool,
        head_pt="pbe_mp",
        head_ft="my_dft",
    )

    heads = {(atoms.info["pretrained"], atoms.info["head"]) for atoms in combined(tmp_path)}
    assert (True, "pbe_mp") in heads
    assert (False, "my_dft") in heads


def test_no_head_key_is_written_when_none_is_given(tmp_path, pool):
    """`None` means "do not label", not "label it None": the key is absent, and a
    consumer distinguishes the two."""
    select(tmp_path, pool, num_samples=3, subselect="random")

    assert all("head" not in atoms.info for atoms in written(tmp_path))


def test_the_atomic_number_filter_restricts_the_pool(tmp_path, mixed_pool):
    """`--filter_atomic_numbers_pt` decides which chemistries can be selected. It
    only applies when no finetuning configurations were given, since those supply
    the element list themselves.
    """
    select(
        tmp_path,
        mixed_pool,
        num_samples=4,
        subselect="random",
        filter_atomic_numbers_pt="[1,8]",
        filtering_type="exclusive",
    )

    elements = {symbol for atoms in written(tmp_path) for symbol in atoms.get_chemical_symbols()}
    assert elements <= {"H", "O"}, elements
    assert "C" not in elements


def test_without_the_filter_both_chemistries_are_available(tmp_path, mixed_pool):
    """Otherwise the test above would pass on a pool that had no carbon in it."""
    select(tmp_path, mixed_pool, num_samples=8, subselect="random")

    elements = {symbol for atoms in written(tmp_path) for symbol in atoms.get_chemical_symbols()}
    assert "C" in elements
