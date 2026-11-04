"""Projection algebra, graph handling, persistence and fixed-point composition."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from conftest import assert_close, fp64_only
from mace_core.config import FixedPointSpec
from mace_core.config.model import MagneticConfig, ModelConfig
from mace_core.graph import AutoPadding
from mace_core.observables import DEFAULT_CATALOGUE
from mace_core.outputs import MACEOutput
from mace_torch.calculators import MACECalculator
from mace_torch.deploy.loader import DeployError, load_deployed
from mace_torch.models.magnetic import MagneticModel
from mace_torch.physics import DerivativeEngine, FixedPointDriver
from mace_torch.serialization import canonical_state
from mace_torch.train import ModelStageError, write_model
from test_mace_torch_magnetic_model import (
    CLUSTER,
    configuration,
    engine,
    graph,
    iron_cluster,
    moments,
)

ENERGY = DEFAULT_CATALOGUE.observable("energy")
MAGMOM = DEFAULT_CATALOGUE.input("magmom")


class Quadratic(MagneticModel):
    """A non-even surface with a known projection and relaxed state."""

    def __init__(self, mode):
        torch.nn.Module.__init__(self)
        self.config = ModelConfig(magnetic={"time_reversal": mode})
        self.coupling = torch.nn.Parameter(torch.tensor(0.7))

    def _forward(self, graph):
        data = graph
        positions, spins = data["positions"], data["magmom"]
        site = positions.square().sum(-1) + spins.square().sum(-1)
        site = site + self.coupling * (positions * spins).sum(-1)
        total = site.new_zeros(data["ptr"].shape[0] - 1).index_add(
            0, data["batch"], site
        )
        return MACEOutput(
            total_energy=total,
            node_energies=site,
            extras={"interaction_energy": total, "node_interaction_energy": site},
        )


def project(model, mode):
    result = deepcopy(model)
    settings = result.backbone.config.magnetic.model_copy(
        update={"time_reversal": mode}
    )
    result.backbone.config = result.backbone.config.model_copy(
        update={"magnetic": settings}
    )
    return result


def compare(left, right):
    assert left.names() == right.names()
    for name in left.names():
        assert_close(left.get(name), right.get(name).detach().cpu().numpy(), name)


@pytest.mark.parametrize("mode", ["batched", "sequential"])
def test_projection_removes_an_odd_term_and_keeps_shared_leaves(mode):
    data = graph(CLUSTER, moments())
    data = {
        key: value.to(torch.get_default_dtype()) if value.is_floating_point() else value
        for key, value in data.items()
    }
    data["positions"].requires_grad_(True)
    data["magmom"].requires_grad_(True)
    model = Quadratic(mode)
    native = DerivativeEngine(model, ENERGY, inputs=[MAGMOM])
    bare = Quadratic("none")(data)
    flipped = Quadratic("none")({**data, "magmom": -data["magmom"]})
    assert not torch.allclose(bare.total_energy, flipped.total_energy)
    output = native(data, compute=("forces", "magforces"), training=True)
    expected = data["positions"].square().sum() + data["magmom"].square().sum()
    assert_close(output.total_energy, expected.reshape(1).detach().numpy())
    assert_close(output.forces, (-2 * data["positions"]).detach().numpy())
    assert_close(output.extras["magforces"], (-2 * data["magmom"]).detach().numpy())
    gradients = torch.autograd.grad(
        output.total_energy.sum(),
        (data["positions"], data["magmom"]),
        create_graph=True,
    )
    assert_close(gradients[1], -output.extras["magforces"].detach().numpy())
    gradient = torch.autograd.grad(
        output.extras["magforces"].square().sum(), model.coupling
    )[0]
    assert_close(gradient, 0.0)


@fp64_only
@pytest.mark.parametrize("mode", ["batched", "sequential"])
def test_projection_supports_double_backward_and_fullgraph(mode):
    data = graph(CLUSTER, moments())
    model = Quadratic(mode)
    positions = data["positions"].requires_grad_()
    spins = data["magmom"].requires_grad_()

    def energy(pos, magmom):
        return model({**data, "positions": pos, "magmom": magmom}).total_energy

    assert torch.autograd.gradgradcheck(energy, (positions, spins))
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    compare(compiled(data), model(data))


@fp64_only
@pytest.mark.parametrize("mode", ["batched", "sequential"])
def test_projection_acts_before_the_fixed_point(mode):
    native = DerivativeEngine(Quadratic(mode), ENERGY, inputs=[MAGMOM])
    driver = FixedPointDriver(
        native, FixedPointSpec(variable="magmom", tolerance=1e-10)
    )
    data = graph(CLUSTER, moments())
    result = driver(data, compute=("forces", "magforces"))
    assert_close(result.extras["converged_magmom"], np.zeros((4, 3)))
    assert_close(result.total_energy, np.array([(CLUSTER**2).sum()]))
    assert_close(result.forces, -2 * CLUSTER)
    with pytest.raises(NotImplementedError, match="second derivative"):
        driver(data, compute=("hessian",))


@pytest.mark.parametrize("mode", ["batched", "sequential"])
@pytest.mark.parametrize("bad", [None, "wrong_shape"])
def test_missing_or_malformed_moments_are_refused(mode, bad):
    data = graph(CLUSTER, moments())
    if bad is None:
        del data["magmom"]
    else:
        data["magmom"] = data["magmom"][:, 0]
    with pytest.raises(ValueError, match=r"magmom.*shape"):
        Quadratic(mode)(data)


@pytest.mark.parametrize("mode", ["none", "batched", "sequential"])
def test_mode_roundtrip_and_unknown_mode(mode):
    config = MagneticConfig(time_reversal=mode)
    assert MagneticConfig.model_validate_json(config.model_dump_json()) == config
    with pytest.raises(ValueError, match="time_reversal"):
        MagneticConfig.model_validate({"time_reversal": "typo"})


@fp64_only
@pytest.mark.parametrize("mode", ["batched", "sequential"])
def test_actual_magnetic_model_symmetries_and_batch(mode):
    base = engine()
    native = project(base, mode)
    data = graph(CLUSTER, moments())
    before = {key: value.clone() for key, value in data.items()}
    wanted = ("forces", "magforces")
    output = native(data, compute=wanted)
    matrix = torch.tensor([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])
    transformed = graph(CLUSTER @ matrix.numpy().T + 2.5, -moments() @ matrix.numpy().T)
    turned = native(transformed, compute=wanted)
    assert_close(turned.total_energy, output.total_energy.detach().numpy())
    assert_close(turned.forces, (output.forces @ matrix.T).detach().numpy())
    assert_close(
        turned.extras["magforces"],
        (-output.extras["magforces"] @ matrix.T).detach().numpy(),
    )
    order = np.array([2, 0, 3, 1])
    permuted = native(
        graph(CLUSTER[order], moments()[order], np.array([8, 26, 26, 26])[order]),
        compute=wanted,
    )
    assert_close(permuted.total_energy, output.total_energy.detach().numpy())
    assert_close(
        permuted.extras["magforces"], output.extras["magforces"][order].detach().numpy()
    )
    for key, value in before.items():
        assert torch.equal(data[key], value) and not data[key].requires_grad
    zero = native(graph(CLUSTER, np.zeros((4, 3))), compute=("magforces",))
    assert_close(zero.extras["magforces"], np.zeros((4, 3)))
    with torch.no_grad():
        energy_only = native(data, compute=())
    assert energy_only.forces is None and "magforces" not in energy_only.extras


@fp64_only
@pytest.mark.parametrize("batched", [False, True])
def test_loaded_projection_persists_and_calculator_padding_is_inert(tmp_path, batched):
    from dataclasses import replace

    from mace_torch.finetune.stages import build

    built = build(configuration(tmp_path))
    write_model(tmp_path / "base", built.model, built.metadata)
    original = load_deployed(tmp_path / "base.json")
    assert isinstance(original.engine, DerivativeEngine)
    original.engine.compile_model(backend="eager")
    original.compute(graph(CLUSTER, moments()), compute=())
    projected = original.with_time_reversal(batched=batched)
    assert projected.engine._compiled_model is None
    assert original.config.model.magnetic.time_reversal == "none"
    old_state, new_state = (
        canonical_state(original.engine),
        canonical_state(projected.engine),
    )
    assert old_state.keys() == new_state.keys()
    for path in old_state:
        for name in old_state[path]:
            assert torch.equal(old_state[path][name], new_state[path][name])
            if old_state[path][name].numel():
                assert (
                    old_state[path][name].data_ptr() != new_state[path][name].data_ptr()
                )
    write_model(tmp_path / "projected", projected.engine, projected.metadata)
    reread = load_deployed(tmp_path / "projected.json")
    assert reread.config == projected.config
    data = graph(CLUSTER, moments())
    compare(
        reread.compute(data, compute=("forces", "magforces")),
        projected.compute(data, compute=("forces", "magforces")),
    )
    reread.engine.compile_model(backend="eager")
    compare(
        reread.compute(data, compute=("forces", "magforces")),
        projected.compute(data, compute=("forces", "magforces")),
    )
    with pytest.raises(DeployError, match="already"):
        projected.with_time_reversal()
    with pytest.raises(DeployError, match="boolean"):
        original.with_time_reversal(batched="False")
    outside = replace(
        original,
        engine=FixedPointDriver(original.engine, FixedPointSpec(variable="magmom")),
    )
    with pytest.raises(DeployError, match="before the fixed-point"):
        outside.with_time_reversal()
    atoms = iron_cluster()
    calculator = MACECalculator(models=reread, pad_num_atoms=8, pad_num_edges=32)
    calculator.calculate(atoms)
    direct = reread.compute(
        graph(CLUSTER, atoms.arrays["REF_magmom"]), compute=("forces", "magforces")
    )
    assert_close(calculator.results["energy"], direct.total_energy[0].detach().numpy())
    assert_close(
        calculator.results["magforces"], direct.extras["magforces"].detach().numpy()
    )
    atoms.arrays["REF_magmom"] *= -1
    calculator.calculate(atoms)
    assert_close(
        calculator.results["magforces"], -direct.extras["magforces"].detach().numpy()
    )


@fp64_only
def test_nonmagnetic_models_refuse_projection(tmp_path):
    from mace_torch.finetune.stages import build

    config = configuration(tmp_path, model="scale_shift", observables=["energy"])
    built = build(config)
    write_model(tmp_path / "energy", built.model, built.metadata)
    with pytest.raises(DeployError, match="magnetic"):
        load_deployed(tmp_path / "energy.json").with_time_reversal()
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"magnetic": MagneticConfig(time_reversal="batched")}
            )
        }
    )
    with pytest.raises(ModelStageError, match=r"magnetic\.time_reversal"):
        build(config)


@fp64_only
@pytest.mark.parametrize("mode", ["batched", "sequential"])
def test_unequal_structures_and_heads_remain_independent(tmp_path, mode):
    from mace_core.data.configuration import Configuration
    from mace_torch.finetune.stages import build
    from mace_torch.graph.collate import build_batch
    from mace_torch.graph.padding import pad_batch
    from mace_torch.neighbors import resolve_neighbor_backend

    config = configuration(tmp_path)
    head = config.data.heads["default"]
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"heads": {"default": head, "other": head}}
            )
        }
    )
    built = build(config)
    native = project(built.model, mode)
    frames = [
        Configuration(
            atomic_numbers=np.array([8, 26, 26, 26])[:count],
            positions=CLUSTER[:count],
            properties={"magmom": moments()[:count]},
            head=head,
        )
        for count, head in ((4, "default"), (2, "other"))
    ]

    def batch(samples):
        return build_batch(
            samples,
            atomic_number_table=built.data.z_table,
            cutoff=4.0,
            neighbor_backend=resolve_neighbor_backend(),
            heads=("default", "other"),
        )

    joint = batch(frames)
    padded, _ = pad_batch(
        joint, AutoPadding(), 4.0, atomic_number_table=built.data.z_table
    )
    before = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in padded.items()
    }
    counts = []
    hook = native.backbone.backbone.register_forward_pre_hook(
        lambda module, args: counts.append(
            (len(args[0]["positions"]), len(args[0]["ptr"]) - 1)
        )
    )
    actual = native(padded, compute=("forces", "magforces"))
    hook.remove()
    n, g = len(padded["positions"]), len(padded["ptr"]) - 1
    assert counts == ([(2 * n, 2 * g)] if mode == "batched" else [(n, g), (n, g)])
    offset = 0
    for index, frame in enumerate(frames):
        single = native(batch([frame]), compute=("forces", "magforces"))
        size = len(frame.positions)
        assert_close(
            actual.total_energy[index], single.total_energy[0].detach().numpy()
        )
        assert_close(
            actual.forces[offset : offset + size], single.forces.detach().numpy()
        )
        assert_close(
            actual.extras["magforces"][offset : offset + size],
            single.extras["magforces"].detach().numpy(),
        )
        offset += size
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(padded[key], value)
        else:
            assert padded[key] == value
