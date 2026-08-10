"""The two tiny parity anchors reproduce their committed references.

These are the numbers everything downstream is measured against: the
converted-architecture gate, the live in-process parity run, and the backend
parity goldens all reduce to "does this checkpoint still produce these
values". They run per pull request, on CPU, with no network and no optional
dependency, and they carry no capability marker on purpose -- an anchor that
could skip would be an anchor that could rot.
"""

import copy
import json

import numpy as np
import pytest
import torch

from mace import data
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools import torch_geometric, utils
from tests.golden import harness

ANCHORS = {
    "tiny_scaleshift": {
        "model": harness.MODELS_DIR / "tiny_scaleshift.model",
        "sidecar": harness.MODELS_DIR / "tiny_scaleshift.build.json",
        "reference": harness.REFERENCES_DIR / "tiny_scaleshift_e3nn_cpu_fp64.json",
        "class": "ScaleShiftMACE",
    },
    "tiny_mace": {
        "model": harness.MODELS_DIR / "tiny_mace.model",
        "sidecar": harness.MODELS_DIR / "tiny_mace.build.json",
        "reference": harness.REFERENCES_DIR / "tiny_mace_e3nn_cpu_fp64.json",
        "class": "MACE",
    },
}


def _load(name):
    return torch.load(
        ANCHORS[name]["model"], weights_only=False, map_location="cpu"
    ).to(torch.float64)


def _batch(model, atoms):
    """One structure as the graph batch the model consumes.

    AtomicData reads the process-wide default dtype, which is float32 under
    pytest, so the graph is cast to the anchors' float64 explicitly rather
    than left to whatever ran before it in the session.
    """
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    config = data.config_from_atoms(atoms)
    atomic_data = data.AtomicData.from_config(
        config, z_table=z_table, cutoff=float(model.r_max)
    )
    loader = torch_geometric.dataloader.DataLoader(
        [atomic_data], batch_size=1, shuffle=False
    )
    graph = next(iter(loader)).to_dict()
    return {
        key: (
            value.to(torch.float64)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value
        )
        for key, value in graph.items()
    }


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures()


@pytest.mark.parametrize("name", sorted(ANCHORS))
def test_anchor_reproduces_its_reference(name, fixtures):
    from mace.calculators import MACECalculator  # noqa: PLC0415

    model = _load(name)
    calc = MACECalculator(models=[model], device="cpu", default_dtype="float64")
    snapshot = harness.snapshot_outputs(
        calc, fixtures, dtype="float64", device="cpu", backend="e3nn"
    )
    reference = harness.load_reference(ANCHORS[name]["reference"])
    harness.compare_to_reference(
        snapshot, reference, row=harness.FP64_CPU_REFERENCE.name
    )


@pytest.mark.parametrize("name", sorted(ANCHORS))
def test_anchor_is_the_class_it_claims_to_be(name):
    """The whole reason there are two anchors.

    The training CLI cannot emit a plain MACE: `--model MACE` returns a
    ScaleShiftMACE with the scale taken from the dataset std and the shift
    zeroed (mace/tools/model_script_utils.py:279-296). If this assertion ever
    starts failing on tiny_mace, the anchor was rebuilt through the CLI and
    silently became the other class.
    """
    model = _load(name)
    assert type(model).__name__ == ANCHORS[name]["class"]
    assert hasattr(model, "pair_repulsion"), "both anchors carry ZBL by design"


@pytest.mark.parametrize("name", sorted(ANCHORS))
def test_reference_carries_dtype_units_and_provenance(name):
    reference = harness.load_reference(ANCHORS[name]["reference"])
    assert reference["dtype"] == "float64"
    assert reference["device"] == "cpu"
    assert reference["backend"] == "e3nn"
    assert reference["units"]["energy"] == "eV"
    provenance = reference["provenance"]
    assert provenance["source"].endswith(ANCHORS[name]["model"].name)
    assert provenance["tolerance_row"] == harness.FP64_CPU_REFERENCE.name
    for entry in reference["fixtures"].values():
        for channel in entry["outputs"].values():
            assert channel["unit"]
            assert channel["kind"] in harness.KINDS


@pytest.mark.parametrize("name", sorted(ANCHORS))
def test_sidecar_records_how_the_anchor_was_built(name):
    sidecar = json.loads(ANCHORS[name]["sidecar"].read_text(encoding="utf-8"))
    assert sidecar["model"] == ANCHORS[name]["model"].name
    assert sidecar["dtype"] == "float64"
    assert sidecar["seed"]
    assert sidecar["command"]
    assert "regenerate.py" in sidecar["regenerate_with"]


def test_the_repulsion_term_is_scaled_in_one_class_and_raw_in_the_other(fixtures):
    """The divergence the two anchors exist to turn into a number.

    Plain MACE appends the pair term to `energies` next to `e0`
    (mace/modules/models.py:359-361) and never scales it. ScaleShiftMACE
    seeds its readout sum with `[pair_node_energy]` (`:539`) and puts the
    whole sum through `scale_shift` (`:579`). Removing the term therefore
    moves the total energy by the raw pair sum in one case and by
    `scale * pair_sum` in the other, and the short dimer fixture makes that
    difference large enough to be unmistakable.
    """
    atoms = fixtures["dimer_short"]
    tol = harness.FP64_CPU_REFERENCE

    ratios = {}
    for name in ANCHORS:
        model = _load(name)
        graph = _batch(model, atoms)
        _, lengths = get_edge_vectors_and_lengths(
            positions=graph["positions"],
            edge_index=graph["edge_index"],
            shifts=graph["shifts"],
        )
        pair_sum = float(
            model.pair_repulsion_fn(
                lengths,
                graph["node_attrs"],
                graph["edge_index"],
                model.atomic_numbers,
            ).sum()
        )
        assert pair_sum > 1.0, "the dimer fixture should be deep in the repulsion"

        stripped = copy.deepcopy(model)
        del stripped.pair_repulsion
        with_pair = float(
            model(_batch(model, atoms), compute_force=False)["energy"].detach()
        )
        without_pair = float(
            stripped(_batch(model, atoms), compute_force=False)["energy"].detach()
        )
        ratios[name] = (with_pair - without_pair) / pair_sum

    assert ratios["tiny_mace"] == pytest.approx(1.0, abs=tol.atol)

    scale = float(_load("tiny_scaleshift").scale_shift.scale[0])
    assert scale != pytest.approx(1.0, abs=1e-3), (
        "a scale of one would make this test vacuous; regenerate the anchor "
        "on a training set whose energy std is not unity"
    )
    assert ratios["tiny_scaleshift"] == pytest.approx(scale, abs=tol.atol)


def test_the_plain_anchor_is_not_convertible_and_says_so():
    """Why the accelerated-backend goldens run on the other anchor.

    `extract_config_mace_model` whitelists ScaleShiftMACE and the extension
    classes; a plain MACE comes back as an error payload rather than a
    config. Pinned here as a contract so the GPU parity work is choosing the
    ScaleShiftMACE anchor for a stated reason rather than working around a
    surprise.
    """
    from mace.tools.scripts_utils import extract_config_mace_model  # noqa: PLC0415

    refused = extract_config_mace_model(_load("tiny_mace"))
    assert isinstance(refused, dict) and "error" in refused
    accepted = extract_config_mace_model(_load("tiny_scaleshift"))
    assert "error" not in accepted


def test_training_error_reference_carries_mae_rmse_and_loss():
    """GATE-3 reads this file to say the rewrite trains comparably."""
    path = harness.REFERENCES_DIR / "tiny_scaleshift_training_errors.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload["error_table"]
    assert {"train_Default", "valid_Default"} <= set(table)
    for row in table.values():
        assert any("RMSE E" in key for key in row)
        assert any("RMSE F" in key for key in row)
    final = payload["final_eval_record"]
    for key in ("loss", "mae_e", "rmse_e", "mae_f", "rmse_f"):
        assert isinstance(final[key], float), key


def test_anchor_checkpoints_stay_small():
    """A committed checkpoint is a permanent cost in every clone."""
    for entry in ANCHORS.values():
        size_mb = entry["model"].stat().st_size / 1e6
        assert size_mb < 1.5, f"{entry['model'].name} is {size_mb:.2f} MB"


def test_zero_edge_and_degenerate_cell_fixtures_produce_finite_numbers(fixtures):
    """The two fixtures most likely to produce a NaN rather than a wrong number."""
    from mace.calculators import MACECalculator  # noqa: PLC0415

    model = _load("tiny_scaleshift")
    calc = MACECalculator(models=[model], device="cpu", default_dtype="float64")
    subset = {
        name: fixtures[name] for name in ("isolated_atom", "slab_zero_vacuum")
    }
    snapshot = harness.snapshot_outputs(calc, subset)
    for name, entry in snapshot["fixtures"].items():
        for channel, payload in entry["outputs"].items():
            values = np.asarray(payload["value"], dtype=float)
            assert np.isfinite(values).all(), f"{name}/{channel} is not finite"
    # the zero-vacuum slab must still get a stress, which is only true
    # because the neighbour-list layer patches its all-zero row
    assert "stress" in snapshot["fixtures"]["slab_zero_vacuum"]["outputs"]
