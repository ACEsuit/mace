"""Explicit provider selection must not execute unrelated optional imports."""

from types import SimpleNamespace

import pytest
from mace_core.data import registry as datasets
from mace_core.data.keys import KeySpecification
from mace_core.electrostatics import registry as solvers
from mace_core.kernels import registry as kernels


@pytest.mark.parametrize("registry", [kernels, solvers, datasets])
@pytest.mark.parametrize("name", ["selected", "missing"])
def test_named_resolution_only_imports_the_selected_plugin(monkeypatch, registry, name):
    imported = []
    result = object()

    class Entry:
        def __init__(self, name):
            self.name = name
            self.value = f"example:{name}"

        def load(self):
            imported.append(self.name)
            if self.name == "unrelated":
                raise ImportError("optional runtime absent")
            if registry is datasets:
                return SimpleNamespace(open=lambda *args, **kwargs: result)
            return lambda: result

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda **kwargs: [Entry("unrelated"), Entry("selected")],
    )
    if registry is kernels:

        def resolve():
            return kernels.get_backend(name)

        error = kernels.BackendNotAvailableError
    elif registry is solvers:

        def resolve():
            return solvers.get_solver(name)

        error = solvers.SolverNotAvailableError
    else:

        def resolve():
            return datasets.open_dataset(
                "unused.xyz", format=name, key_spec=KeySpecification()
            )

        error = datasets.UnknownBackendError
    if name == "selected":
        assert resolve() is result
        assert imported == ["selected"]
    else:
        with pytest.raises(error, match="selected"):
            resolve()
        assert imported == []
