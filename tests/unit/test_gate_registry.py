"""Every `gate_dict` entry is reachable and does something.

`gate_dict` is one of the string to callable registries that connect a CLI value
to an implementation, so `--gate abs` and `--gate None` are part of the surface.
Only `silu` was ever built in a test, which leaves three ways to break the other
two without a failure: drop the key, point it at the wrong callable, or stop
threading it into the readout.

`None` is the one worth spelling out. It is the string `"None"` on the command
line and a real `None` in the dict, and it has to survive `nn.Activation`, which
is not obviously true of a null activation.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.tools import torch_geometric

TABLE = tools.AtomicNumberTable([1, 8])
ATOMIC_ENERGIES = np.array([1.0, 3.0], dtype=float)


def build(gate):
    """A tiny MACE with a NonLinear readout, which is what consumes the gate."""
    return modules.MACE(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=gate,
        atomic_energies=ATOMIC_ENERGIES,
        avg_num_neighbors=8,
        atomic_numbers=TABLE.zs,
        correlation=3,
        radial_type="bessel",
    )


@pytest.fixture(name="batch")
def fixture_batch():
    config = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        properties={"forces": np.zeros((3, 3)), "energy": -1.5},
        property_weights={"forces": 1.0, "energy": 1.0},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=TABLE, cutoff=3.0)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False, drop_last=False
    )
    return next(iter(loader))


def test_the_registry_maps_every_cli_value_to_its_callable():
    """The mapping itself, since the CLI passes strings and nothing else checks
    which callable each one lands on."""
    assert modules.gate_dict["abs"] is torch.abs
    assert modules.gate_dict["tanh"] is torch.tanh
    assert modules.gate_dict["silu"] is torch.nn.functional.silu
    assert modules.gate_dict["None"] is None


@pytest.mark.parametrize("name", ["abs", "tanh", "silu", "None"])
def test_a_model_builds_and_runs_with_every_gate(name, batch):
    """Reachability: a registry entry that cannot be built is not a feature."""
    model = build(modules.gate_dict[name])

    out = model(batch.to_dict(), training=False)

    assert out["energy"] is not None
    assert torch.isfinite(out["energy"]).all()


def test_the_gate_changes_the_energy(batch):
    """Not just "it builds". Two gates over the same weights must disagree, or
    the argument is being accepted and dropped.

    Seeded per build, because the readout weights are random and two models with
    the same seed differ only in the gate.
    """
    energies = {}
    for name in ("abs", "tanh", "silu", "None"):
        torch.manual_seed(0)
        model = build(modules.gate_dict[name])
        energies[name] = float(model(batch.to_dict(), training=False)["energy"])

    assert len(set(energies.values())) == len(energies), (
        f"two gates produced the same energy, so at least one is not applied: "
        f"{energies}"
    )
