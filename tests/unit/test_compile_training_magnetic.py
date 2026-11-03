"""Compiled training on the magnetic family: fresh construction, FX tracing,
numerical parity against the eager model, and dynamic-shape reuse.

The magnetic anchor cannot be used here: it was built with e3nn codegen
enabled, so its tensor products carry generated paths that symbolic tracing
refuses. Real compiled training constructs the model under
``disable_e3nn_codegen`` — the same thing ``run_train`` does via
``_configure_model_for_training`` — and that is what this test reproduces.
"""

import pytest
import torch

pytestmark = pytest.mark.magnetic

from mace.data import utils as data_utils  # noqa: E402
from mace.data.atomic_data import AtomicData  # noqa: E402
from mace.tools import compile as mace_compile  # noqa: E402
from mace.tools import torch_geometric  # noqa: E402
from mace.tools import torch_tools  # noqa: E402
from mace.tools.compile_training import compile_training_model  # noqa: E402
from tests.golden import magnetic_surfaces as ms  # noqa: E402
from tests.golden.build_magnetic_anchor import build_model  # noqa: E402


def _multi_graph_batch(model, atoms_list):
    z_table = data_utils.AtomicNumberTable(
        [int(z) for z in model.atomic_numbers]
    )
    keyspec = data_utils.KeySpecification(
        info_keys={}, arrays_keys={"magmom": ms.MAGMOM_KEY}
    )
    with torch_tools.default_dtype("float64"):
        data = [
            AtomicData.from_config(
                data_utils.config_from_atoms(atoms, key_specification=keyspec),
                z_table=z_table,
                cutoff=float(model.r_max),
            )
            for atoms in atoms_list
        ]
        loader = torch_geometric.dataloader.DataLoader(
            data, batch_size=len(data)
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


def test_magnetic_model_compiles_with_eager_parity():
    pytest.importorskip("sphericart.torch")
    fixtures = ms.magnetic_fixtures()
    names = sorted(fixtures)

    with torch_tools.default_dtype("float64"), mace_compile.disable_e3nn_codegen():
        model = build_model()

    example = _multi_graph_batch(
        model, [fixtures[names[0]], fixtures[names[1]], fixtures[names[2]]]
    )
    validation = _multi_graph_batch(model, [fixtures[names[3]], fixtures[names[4]]])

    with torch_tools.default_dtype("float64"):
        reference = model(
            {key: value.clone() for key, value in example.items()},
            training=True,
            compute_force=True,
        )
        compiled = compile_training_model(
            model,
            example,
            {"energy": True, "forces": True},
            mode="default",
            dynamic=True,
            validation_batch=validation,
        )

        assert compiled.is_compiled, compiled.compilation_error

        out = compiled(example)
        torch.testing.assert_close(out["energy"], reference["energy"])
        torch.testing.assert_close(
            out["forces"], reference["forces"], rtol=1e-12, atol=1e-12
        )

        # A different batch shape must flow through the same compiled facade
        # and still match the eager model (the magnetic family reports
        # per-graph energies with shape (N,), unlike the standard (N, 1)).
        reference_validation = model(
            {key: value.clone() for key, value in validation.items()},
            training=True,
            compute_force=True,
        )
        reshaped = compiled(validation)
        torch.testing.assert_close(
            reshaped["energy"], reference_validation["energy"]
        )

        # The compiled graph must be differentiable end to end.
        out["energy"].sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(bool(torch.isfinite(g).all()) for g in grads)
