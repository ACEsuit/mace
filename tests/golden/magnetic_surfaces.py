"""How the magnetic anchor is driven, on each of the three surfaces.

Shared by ``regenerate.py`` and by the two magnetic golden test modules, and
that sharing is the point: a reference and the test that checks it have to
drive the model the same way, and the only reliable way to guarantee that is
for there to be one driver.

Three surfaces, because the magnetic family is the first one where they are
not interchangeable:

* **the model forward** is the only door ``magforces`` comes through.
  ``MagneticMACECalculator`` computes it -- the forward defaults to
  ``compute_magforces=True`` -- and then throws it away: its results dict
  carries energy, free_energy, node_energy, forces, stress and, for an
  SCF-wrapped model, the relaxed moments. So the reference this family exists
  to provide cannot be taken through a calculator at all.
* **the calculator** is what an ase-driven workflow sees, and it is checked
  against the same reference rather than given one of its own.
* **the evaluation CLI** is what ``--return_magforces`` writes, and it is the
  only surface a user gets ``dE/dm`` out of without writing python.

All three are pinned against one reference file. That is what the alias
registry is for, and it is not decoration here: the calculator renames the
relaxed moments to ``MACE_magmoms``, and the three surfaces disagree about
what ``node_energy`` means (see ``model_keys.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import ase.io
import numpy as np
import torch

from mace import data as mace_data
from mace.modules.extensions import MagneticSCFMACE
from mace.tools import torch_geometric, torch_tools, utils
from tests.golden import harness
from tests.golden.build_magnetic_anchor import (
    ATOMIC_NUMBERS,
    MODEL_PATH,
    SCF_CONFIG,
)

#: The array the magnetic models read their moments from
#: (``mace/tools/default_keys.py:18``), and therefore the array the fixtures
#: write them to. Not ase's initial magnetic moments: no forward on this tree
#: reads those, and the harness refuses a structure that carries them there.
MAGMOM_KEY = "REF_magmom"

#: What the eval CLI prefixes its results with by default.
INFO_PREFIX = "MACE_"


def load_anchor(model_path: Path = MODEL_PATH) -> torch.nn.Module:
    """The committed magnetic anchor, in float64. Needs the `magnetic` extra."""
    return torch.load(model_path, weights_only=False, map_location="cpu").to(
        torch.float64
    )


def load_scf_anchor(**overrides: Any) -> MagneticSCFMACE:
    """The same anchor wrapped for a self-consistent relaxation of the moments.

    A **fresh** wrapper, every time, and callers should keep it that way.
    ``MagneticSCFMACE`` stores the converged moments in ``cache_magmom`` and
    uses them as the starting point of the next call that arrives without any
    (``mace/modules/extensions.py:2018-2026``), so a wrapper reused across
    fixtures carries state between them. Our fixtures all carry their own
    moments, so the cache never gets to speak -- but a golden whose value
    depends on the order its fixtures happened to run in is not a golden, and
    the cheapest way not to have that property is not to share the object.
    """
    settings = dict(SCF_CONFIG)
    settings.update(overrides)
    return MagneticSCFMACE(model=load_anchor(), **settings)


def magnetic_fixtures() -> Dict[str, Any]:
    """The fixture subset this family can evaluate: iron, carrying moments.

    Both filters are load-bearing and neither is redundant. ``elements``
    excludes the H/C/O anchor set, which is not in this model's z-table;
    ``tags`` excludes anything that has iron but no moments, since ``magmom``
    is a required input -- the forward calls ``data["magmom"].requires_grad_``
    before it does anything else -- and a structure without one is a KeyError,
    not an empty channel.
    """
    return harness.load_fixtures(tags=["magmom"], elements=ATOMIC_NUMBERS)


#: The subset the SCF reference is taken on, and it is a subset for a measured
#: reason rather than for economy.
#:
#: ``MagneticSCFMACE`` relaxes the moments with LBFGS and returns wherever the
#: optimiser stopped. On these three, that point is a smooth function of where
#: the relaxation started: perturbing the initial moments by 1e-9 moves the
#: answer by 1e-9. On the other two -- ``mag_fe3_canted`` and
#: ``mag_feo_cluster`` -- the same 1e-9 perturbation moves it by 2e-5 to 7e-5,
#: and perturbing by 1e-6 moves it by about the same amount again, which is
#: the signature of a terminal point that is not a function of the input at
#: all. Pinning one of those at the fp64 row would be pinning the arithmetic
#: of one machine: the cross-machine noise a golden has to tolerate is many
#: orders of magnitude smaller than 1e-9 and would be amplified the same way.
#:
#: So the reference covers the three, and the exclusion of the other two is
#: itself asserted by a named test, which re-measures the amplification. If
#: the wrapper ever grows a real convergence criterion, that test fails and
#: the reference can be widened.
SCF_REFERENCE_FIXTURES = ("mag_fe_atom", "mag_fe_dimer_fm", "mag_fe_dimer_afm")


def scf_fixtures():
    """The moment-carrying fixtures whose relaxed state is reproducible."""
    return harness.load_fixtures(
        list(SCF_REFERENCE_FIXTURES), tags=["magmom"], elements=ATOMIC_NUMBERS
    )


def build_batch(model: torch.nn.Module, atoms) -> Dict[str, torch.Tensor]:
    """One structure as the graph batch the model consumes, in float64.

    The float64 scope has to cover ``AtomicData.from_config`` and not only the
    config: that is where the tensors are built and it reads the process-wide
    default dtype, which is float32 under pytest. The same trap is documented
    at length in ``test_tiny_anchors._batch`` -- a graph built in float32 and
    cast up afterwards reproduces the calculator to about 2e-8, which is
    inside the fp64 row and so reads as agreement while making a bit-exact
    comparison impossible.
    """
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    keyspec = mace_data.KeySpecification(
        info_keys={}, arrays_keys={"magmom": MAGMOM_KEY}
    )
    with torch_tools.default_dtype("float64"):
        config = mace_data.config_from_atoms(atoms, key_specification=keyspec)
        atomic_data = mace_data.AtomicData.from_config(
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


class MagneticForward:
    """The model surface: the forward's own dict, which is where dE/dm lives.

    ``magmom_key`` is not decoration either. The harness asks the object it
    was handed where it reads its inputs from (``register_input_probe`` in
    ``calculator_keys.py``), and without it the recorded moments would come
    from a default rather than from what this evaluation actually read.
    """

    golden_surface = harness.SURFACE_MODEL
    magmom_key = MAGMOM_KEY

    def __init__(self, model: Optional[torch.nn.Module] = None, **forward_kwargs):
        self.model = load_anchor() if model is None else model
        self.forward_kwargs = {
            "training": False,
            "compute_force": True,
            "compute_magforces": True,
        }
        self.forward_kwargs.update(forward_kwargs)

    def golden_outputs(self, atoms) -> Dict[str, Any]:
        with torch_tools.default_dtype("float64"):
            out = self.model(build_batch(self.model, atoms), **self.forward_kwargs)
        recorded: Dict[str, Any] = {
            # Graph channels are declared per graph, so the single graph is
            # indexed out here rather than squeezed inside the schema.
            "energy": float(out["energy"][0].detach()),
            "node_energy": out["node_energy"].detach().numpy(),
            "forces": out["forces"].detach().numpy(),
        }
        if out["magforces"] is not None:
            recorded["magforces"] = out["magforces"].detach().numpy()
        return recorded


class MagneticSCFForward:
    """The model surface again, with the moments relaxed to their fixed point.

    A fresh wrapper per structure, for the reason in :func:`load_scf_anchor`.
    ``scf_steps`` and ``scf_energy_history`` come back too and are declared
    metadata in the schema, so the harness records them next to the numbers
    instead of asserting them: they describe how LBFGS got to the fixed point,
    not where it is, and a build that converges in one step fewer would fail a
    reference that pinned them without a single physical number having moved.
    """

    golden_surface = harness.SURFACE_MODEL
    magmom_key = MAGMOM_KEY

    def __init__(self, **overrides: Any):
        self.overrides = overrides
        self.last: Optional[MagneticSCFMACE] = None

    def golden_outputs(self, atoms) -> Dict[str, Any]:
        wrapper = load_scf_anchor(**self.overrides)
        self.last = wrapper
        with torch_tools.default_dtype("float64"):
            out = wrapper(build_batch(wrapper.magmom_mace, atoms), compute_force=True)
        return {
            "energy": float(out["energy"][0].detach()),
            "forces": out["forces"].detach().numpy(),
            "equilibrated_magmom": out["equilibrated_magmom"].detach().numpy(),
            "scf_steps": out["scf_steps"],
            "scf_energy_history": out["scf_energy_history"].numpy(),
        }


class MagneticEvalCLI:
    """The evaluation command line, which returns nothing and writes files.

    ``mace_eval_configs`` puts its results back onto the structures under a
    prefix and writes extxyz, so this drives the real ``run()`` over a
    one-structure file, reads the file back and hands the harness what the
    prefix collected. Reading the file rather than the in-memory objects is
    deliberate: the written artefact is what a user compares against, and
    extxyz's ``%16.8f`` on the per-atom columns is part of that surface.

    ``run()`` calls ``torch_tools.set_default_dtype`` on the *process*
    (``mace/cli/eval_configs.py:188``), so the call is wrapped in a scope that
    puts it back -- otherwise this leaks float64 into every test that runs
    after it in the same worker.
    """

    golden_surface = harness.SURFACE_EVAL
    magmom_key = MAGMOM_KEY

    def __init__(self, workdir: Path, model_path: Path = MODEL_PATH):
        self.workdir = Path(workdir)
        self.model_path = Path(model_path)
        self.calls = 0

    def golden_outputs(self, atoms) -> Dict[str, Any]:
        self.calls += 1
        stem = atoms.info.get("golden_name", f"structure_{self.calls}")
        configs = self.workdir / f"{stem}.in.xyz"
        output = self.workdir / f"{stem}.out.xyz"
        probe = atoms.copy()
        probe.info.pop("golden_name", None)
        ase.io.write(configs, probe, format="extxyz")
        args = argparse.Namespace(
            model=str(self.model_path),
            configs=str(configs),
            output=str(output),
            device="cpu",
            default_dtype="float64",
            batch_size=1,
            compute_stress=False,
            compute_bec=False,
            enable_cueq=False,
            return_contributions=False,
            return_descriptors=False,
            return_node_energies=True,
            return_magforces=True,
            info_prefix=INFO_PREFIX,
            head=None,
            magmom_key=MAGMOM_KEY,
        )
        from mace.cli.eval_configs import run  # noqa: PLC0415

        with torch_tools.default_dtype("float64"):
            run(args)
        written = ase.io.read(output, index=0, format="extxyz")
        collected = harness.collect_prefixed_outputs(written, INFO_PREFIX)
        # The CLI writes one energy per structure as a scalar in `info`; every
        # other quantity is a per-atom column. Nothing is dropped here -- the
        # schema decides what it knows, and an unknown key raises.
        return {key: np.asarray(value) for key, value in collected.items()}
