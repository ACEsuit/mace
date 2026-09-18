"""Two adapters that let a non-energy model reach the golden harness.

The harness accepts three shapes (``harness._evaluate``), and for an energy
model the middle one -- "an ase calculator" -- is the whole story: hand it the
calculator and it drives ``get_potential_energy`` / ``get_forces`` / (for a
periodic structure) ``get_stress`` and scrapes ``results``.

**That route does not exist for a model with no energy.** ``MACECalculator``
built with ``model_type="DipoleMACE"`` or ``"DipolePolarizabilityMACE"``
leaves no ``energy`` in ``results`` at all (``mace/calculators/mace.py:756``
onwards populates only what the forward returned), so the very first accessor
raises ``PropertyNotImplementedError`` and the snapshot never happens. This is
not a gap in the schema -- every key these families emit resolves to a channel
-- it is that the *driving* convention is energy-shaped. The harness already
provides the way out: an object exposing ``golden_outputs(atoms)`` is called
instead, and ``golden_surface`` says which vocabulary the returned dict is in.
So :class:`CalculatorRoute` is a calculator-surface evaluation that is driven
by ``calculate()`` directly rather than through an ase property accessor.

The second adapter is the model surface. It exists here rather than in a test
file because two goldens in this ticket need it (the tiny dipole anchor, whose
``atomic_dipoles`` never reach any calculator's ``results``; and the MACE-MDP
cross-check, which has to compare the calculator's numbers against the
forward's), and because the graph has to be built inside a float64 scope --
a detail that is silently wrong rather than loud when it is missed, and that
should be got right once.

Both are deliberately family-agnostic: the magnetic (P0-3b) and LES (P0-3c)
goldens meet the same two problems and should import these rather than grow a
third and fourth copy.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from tests.golden import harness


def graph_batch(model: torch.nn.Module, atoms, dtype: str = "float64") -> Dict[str, Any]:
    """One ``ase.Atoms`` as the graph batch ``model.forward`` consumes.

    Built inside a ``default_dtype`` scope, not cast afterwards.
    ``AtomicData`` reads the process-wide default dtype, which is float32
    under pytest; casting a float32 graph up afterwards has already rounded
    the positions, and the forward then agrees with the calculator only to
    about 2e-8 relative -- inside the fp64 tolerance row, so it reads as
    agreement while making a bit-exact comparison impossible. The trailing
    cast is a belt-and-braces for any tensor the scope does not reach.
    """
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    torch_dtype = getattr(torch, dtype)
    with torch_tools.default_dtype(dtype):
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
            value.to(torch_dtype)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value
        )
        for key, value in graph.items()
    }


class CalculatorRoute:
    """An ase calculator driven by ``calculate()`` instead of by an accessor.

    Attribute lookups fall through to the wrapped calculator, and that is
    load-bearing rather than convenience: the harness recovers what an
    evaluation *reads* by inspecting the object it was handed
    (``info_keys`` / ``arrays_keys``, and the ``charges_key`` / ``magmom_key``
    fallbacks -- see ``tests/golden/calculator_keys.py``). An adapter that
    only forwarded the outputs would present a bare object with no key
    mapping, the harness would fall back to its literal defaults, and every
    reference taken through here would record the inputs of a reader other
    than the one that ran. Forwarding everything also means a probe added by
    a later ticket works here with no change.
    """

    golden_surface = harness.SURFACE_CALCULATOR

    def __init__(self, calc: Any) -> None:
        self._calc = calc

    def __getattr__(self, name: str) -> Any:
        # Only public names fall through: `self._calc` itself must never be
        # resolved through here, or a lookup before __init__ finishes recurses
        # until the stack ends.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_calc"], name)

    def golden_outputs(self, atoms) -> Dict[str, Any]:
        # A copy, because `Calculator.calculate` stores the structure it was
        # given on the instance and the harness hands out the fixtures it also
        # reads the inputs from.
        self._calc.calculate(atoms.copy())
        return dict(self._calc.results)


class ForwardRoute:
    """A model ``forward`` presented as a golden evaluation.

    ``project`` turns the raw forward dict into the snapshot's vocabulary. It
    is required rather than defaulted because graph-level channels are
    declared per graph -- ``dipole`` is ``(3,)``, not ``(n_graphs, 3)`` -- so
    the one graph has to be indexed out, and doing that inside the harness
    would silently accept a two-graph batch as a one-graph result (see
    ``tests/golden/model_keys.py``, note 4).
    """

    golden_surface = harness.SURFACE_MODEL

    def __init__(
        self,
        model: torch.nn.Module,
        project: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        dtype: str = "float64",
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.project = project
        self.dtype = dtype
        self.forward_kwargs = dict(forward_kwargs or {})

    def golden_outputs(self, atoms) -> Dict[str, Any]:
        # The scope has to cover the forward too, not only the graph: an
        # extension that creates a tensor mid-forward without a dtype follows
        # the process-wide default, and the mismatch then surfaces from a
        # linalg op in the backward rather than here.
        with torch_tools.default_dtype(self.dtype):
            out = self.model(
                graph_batch(self.model, atoms, dtype=self.dtype),
                **self.forward_kwargs,
            )
        return self.project(out)


def as_numpy(value):
    """Detach a forward's tensor into the array the harness encodes."""
    return value.detach().cpu().numpy()
