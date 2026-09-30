"""The published foundation models, converted, against the frozen tree.

Every artifact on the roster goes through the production converter and is
held to three things:

* **The model it describes**, in process. The frozen tree's model is rebuilt
  from the artifact's configuration at float64 and given the artifact's
  weights and constants, and the conversion has to compute what it computes to
  ten digits, on every structure and every head.
* **The model as published.** The same comparison against the pickle loaded
  as the calculator loads it. For an artifact built in float64 that is the
  same bound. An artifact built in float32 keeps its coupling bases, and in
  one case its distance transform, at float32 even after it is carried to
  float64, and that rounding is part of what the pickle computes: the float32
  part of a basis lies off the span the reduced basis spans, so no reduced
  model reproduces it. For those the bound is the harness's float32 row.
* **The committed goldens**, where there are any, at the harness row that
  applies by the same rule.

The artifacts are read from the frozen tree's cache and downloaded only when
the network is allowed.
"""

from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from mace_core.data.configuration import Configuration
from mace_core.observables import load_default_catalogue
from mace_torch.data.batch import collate_training
from mace_torch.data.graphs import graph_from_configuration
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.deploy.reference import verify_against_reference
from mace_torch.train.data_stage import graph_inputs_of

from tests.golden import foundation_artifacts as fa
from tests.golden.harness import load_fixtures, tolerance
from tests.parity.test_fm_convert_legacy import EXTRACTOR, FIXTURES, extract_here

CATALOGUE = load_default_catalogue()

#: The in-process bound: ten digits, relative to the largest component of
#: the quantity on that structure.
IN_PROCESS = 1e-10

FLOAT32 = tolerance("fp32")
REFERENCE = tolerance("fp64_cpu_reference")

OMOL_URL = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/"
    "MACE-omol-0-extra-large-1024.model"
)

#: ``name -> (family, key)``: the frozen tree's URL tables, read rather than
#: copied, so a re-pointed alias is the artifact the test converts.
ROSTER = {
    **{f"mp-{key}": ("mace_mp_urls", key) for key in (
        "small", "medium", "large", "small-0b", "medium-0b", "small-0b2",
        "medium-0b2", "large-0b2", "medium-0b3", "medium-mpa-0",
        "small-omat-0", "medium-omat-0", "mace-matpes-pbe-0",
        "mace-matpes-r2scan-0", "mh-0", "mh-1",
    )},
    **{f"off-{key}": ("mace_off_urls", key) for key in ("small", "medium", "large")},
    "omol": ("omol", "extra_large"),
    **{key: ("polar_model_urls", key) for key in ("polar-1-s", "polar-1-m", "polar-1-l")},
}

#: Roster artifacts the converter does not carry yet, and what it stops at.
NOT_YET: dict[str, str] = {}


def _url(name: str) -> str:
    from mace.calculators import foundations_models

    table, key = ROSTER[name]
    if table == "omol":
        return OMOL_URL
    return getattr(foundations_models, table)[key]


def artifact_path(name: str) -> Path:
    """The cached file, downloaded first when the network is allowed."""
    from mace.calculators import foundations_models

    table, key = ROSTER[name]
    if table == "mace_mp_urls" and key == "medium-mpa-0":
        return fa.REPO_ROOT / fa.TRACKED_MPA0
    base = os.path.basename(_url(name))
    if table in ("mace_mp_urls", "polar_model_urls"):
        base = "".join(c for c in base if c.isalnum() or c == "_")
    path = Path(foundations_models.get_cache_dir()) / base.split("?")[0]
    if path.is_file():
        return path
    if os.environ.get("MACE_CI_ALLOW_NETWORK") != "1":
        pytest.skip(f"{name} is not cached, and downloads are opt-in")
    if table == "mace_mp_urls":
        return Path(foundations_models.download_mace_mp_checkpoint(key))
    if table == "polar_model_urls":
        return Path(foundations_models.download_mace_polar_checkpoint(key))
    loader = foundations_models.mace_off if table == "mace_off_urls" else (
        foundations_models.mace_omol
    )
    loader(model=key, return_raw_model=True, device="cpu")
    return path


def _param(name: str):
    marks = [] if name == "mp-medium-mpa-0" else [pytest.mark.network]
    if name.startswith("polar-"):
        # The frozen tree unpickles these through graph_longrange.
        marks.append(pytest.mark.polar)
    if name in NOT_YET:
        marks.append(
            pytest.mark.xfail(
                raises=EXTRACTOR["ExtractionError"], strict=True, reason=NOT_YET[name]
            )
        )
    return pytest.param(name, marks=marks, id=name)


PARAMS = [_param(name) for name in ROSTER]


def legacy_model(path: Path) -> torch.nn.Module:
    return torch.load(path, map_location="cpu", weights_only=False)


@functools.lru_cache(maxsize=2)
def converted(path: Path):
    with tempfile.TemporaryDirectory() as directory:
        sidecar = extract_here(path, Path(directory) / "artifact")
        return import_neutral(sidecar, CATALOGUE), _rebuilt_bases(sidecar)


def _rebuilt_bases(sidecar: Path) -> bool:
    from mace_core.weights.neutral_format import read_neutral

    return any(
        spec.descriptor.get("basis_rebuilt_at_float64")
        for spec in read_neutral(sidecar).sidecar.ops.values()
        if spec.op_kind == "symmetric_contraction"
    )


def float32_constants(path: Path) -> bool:
    """Whether the pickle computes with constants held at float32."""
    legacy = legacy_model(path)
    stored = any(
        value.dtype == torch.float32
        for value in legacy.state_dict().values()
        if value.is_floating_point()
    )
    return stored or converted(path)[1]


def exact_rebuild(path: Path) -> torch.nn.Module:
    """The frozen tree's model rebuilt at float64 from the artifact's
    configuration, with every tensor the converter carries copied in and
    everything it derives built fresh.

    A coupling basis is taken fresh by the extractor's own rule: when the
    stored one is the float64 construction held at float32. A basis of
    another construction, which some published models hold, is copied.
    """
    from mace.tools.scripts_utils import extract_config_mace_model

    tolerance = EXTRACTOR["FLOAT32_BASIS_TOLERANCE"]
    trained = legacy_model(path)
    walk = EXTRACTOR["walk"](trained, EXTRACTOR["classify"](trained))
    fresh = type(trained)(**extract_config_mace_model(trained)).to(torch.float64)
    state = fresh.state_dict()
    for key, value in trained.state_dict().items():
        if key in walk.derived or key not in state:
            continue
        if ".U_matrix_" in key:
            stored = value.double()
            scale = float(state[key].abs().max())
            if float((state[key] - stored).abs().max()) <= tolerance * scale:
                continue
        state[key] = value.to(state[key].dtype) if value.is_floating_point() else value
    if "kspace_cutoff" in state:
        # The cutoff the forward sums to, carried rather than derived; see
        # the converter.
        state["kspace_cutoff"] = trained.kspace_cutoff.to(state["kspace_cutoff"].dtype)
    fresh.load_state_dict(state)
    return fresh


def v1_outputs(imported, atoms, head: int) -> dict[str, np.ndarray]:
    inputs = graph_inputs_of(imported.config.model)
    configuration = Configuration(
        atomic_numbers=np.asarray(atoms.get_atomic_numbers()),
        positions=np.asarray(atoms.get_positions(), dtype=np.float64),
        cell=np.asarray(atoms.get_cell().array, dtype=np.float64),
        pbc=tuple(bool(axis) for axis in atoms.get_pbc()),
        properties={name: atoms.info[name] for name in inputs if name in atoms.info},
    )
    graph = graph_from_configuration(
        configuration,
        cutoff=float(imported.config.model.r_max),
        z_table=imported.z_table,
        head=head,
        graph_inputs=inputs,
    )
    batch = collate_training([(graph, {}, {})], z_table=imported.z_table)
    output = imported.engine(dict(batch.graph), compute=("forces", "stress"))
    values = {
        "energy": output.total_energy.detach().reshape(()),
        "forces": output.forces.detach(),
    }
    if any(atoms.get_pbc()):
        values["stress"] = output.stress.detach().reshape(3, 3)
    return {name: value.double().numpy() for name, value in values.items()}


def charge_aware(path: Path) -> bool:
    return type(legacy_model(path)).__name__ == "PolarMACE"


def legacy_calculator(path: Path, head: str, model=None):
    """The frozen tree's calculator, on the file or on a model already built."""
    from mace.calculators import MACECalculator

    kind = "PolarMACE" if charge_aware(path) else "MACE"
    if model is None:
        return MACECalculator(
            model_paths=str(path),
            device="cpu",
            default_dtype="float64",
            head=head,
            model_type=kind,
        )
    return MACECalculator(
        models=[model], device="cpu", default_dtype="float64", head=head, model_type=kind
    )


def legacy_outputs(calculator, atoms) -> dict[str, np.ndarray]:
    atoms = atoms.copy()
    atoms.calc = calculator
    values = {
        "energy": np.asarray(atoms.get_potential_energy()),
        "forces": atoms.get_forces(),
    }
    if any(atoms.get_pbc()):
        values["stress"] = atoms.get_stress(voigt=False)
    return values


def deviations(path: Path, calculator_for) -> list[tuple[str, str, str, float, float]]:
    """``(head, fixture, quantity, largest difference, largest component)``
    for every head of the artifact on every structure it can evaluate."""
    imported, _ = converted(path)
    heads = list(getattr(legacy_model(path), "heads", ["Default"]))
    found = []
    for index, head in enumerate(heads):
        calculator = calculator_for(head)
        for fixture, atoms in load_fixtures().items():
            if not set(atoms.get_atomic_numbers().tolist()) <= set(
                imported.z_table.zs
            ):
                continue
            got = v1_outputs(imported, atoms, index)
            for quantity, value in legacy_outputs(calculator, atoms).items():
                if quantity == "stress" and charge_aware(path):
                    # The frozen tree's charge-aware stress leaves out the
                    # long-range energy's cell dependence; v1's is checked
                    # against a finite difference instead.
                    continue
                difference = float(np.abs(got[quantity] - value).max())
                found.append(
                    (head, fixture, quantity, difference, float(np.abs(value).max()))
                )
    return found


@pytest.mark.parametrize("name", PARAMS)
def test_the_conversion_is_the_model_the_artifact_describes(name, fp64):
    path = artifact_path(name)
    model = exact_rebuild(path)
    exceeded = [
        entry
        for entry in deviations(path, lambda head: legacy_calculator(path, head, model))
        if entry[3] > IN_PROCESS * max(entry[4], 1.0)
    ]
    assert not exceeded, exceeded


@pytest.mark.parametrize("name", PARAMS)
def test_the_conversion_is_the_model_as_published(name, fp64):
    """At ten digits for an artifact built in float64, and at the float32 row
    for one whose pickle computes with float32 constants."""
    path = artifact_path(name)
    rounded = float32_constants(path)
    found = deviations(path, lambda head: legacy_calculator(path, head))

    def allowed(largest: float) -> float:
        if rounded:
            return FLOAT32.atol + FLOAT32.rtol * largest
        return IN_PROCESS * max(largest, 1.0)

    exceeded = [entry for entry in found if entry[3] > allowed(entry[4])]
    assert not exceeded, exceeded


#: The committed goldens, by the roster name of the artifact they were taken
#: from.
GOLDENS = {
    "mp_small": "mp-small",
    "mpa0_medium": "mp-medium-mpa-0",
    "off_small": "off-small",
    "mh_0": "mp-mh-0",
}


@pytest.mark.parametrize(
    "golden",
    [
        pytest.param(
            golden,
            marks=[pytest.mark.network] if fa.ARTIFACTS[golden].network else [],
            id=golden,
        )
        for golden in sorted(GOLDENS)
    ],
)
def test_the_conversion_reproduces_the_committed_golden(golden, fp64):
    spec = fa.ARTIFACTS[golden]
    path = artifact_path(GOLDENS[golden])
    imported, _ = converted(path)
    heads = list(getattr(legacy_model(path), "heads", ["Default"]))
    head = spec.loader_kwargs.get("head", heads[0])
    row = FLOAT32 if float32_constants(path) else REFERENCE
    report = verify_against_reference(
        imported.engine,
        fa.REPO_ROOT / "tests" / "golden" / "references" / spec.reference,
        FIXTURES,
        z_table=imported.z_table,
        cutoff=float(imported.config.model.r_max),
        head=heads.index(head),
        atol=row.atol,
        rtol=row.rtol,
        graph_inputs=graph_inputs_of(imported.config.model),
    )
    assert report.passed, report.describe()


@pytest.mark.parametrize(
    ("name", "removed"),
    [
        pytest.param("mp-small", 341_760, marks=pytest.mark.network, id="mp-small"),
        pytest.param("mp-medium-mpa-0", 820_224, id="mp-medium-mpa-0"),
        pytest.param("off-medium", 92_160, marks=pytest.mark.network, id="off-medium"),
    ],
)
def test_the_projection_removes_exactly_the_gauge_parameters(name, removed, fp64):
    path = artifact_path(name)
    legacy = sum(p.numel() for p in legacy_model(path).parameters())
    v1 = sum(p.numel() for p in converted(path)[0].engine.parameters())
    assert legacy - v1 == removed


@pytest.mark.network
def test_mace_mp_0b_medium_computes_its_pair_radii_in_float32(fp64):
    """Its parameters are float64 and its Agnesi transform's buffers float32,
    so the calculator, seeing a float64 model, converts nothing and sums each
    pair's covalent radii in float32. The conversion computes at the model's
    precision, which is the model the artifact describes."""
    from mace.calculators import MACECalculator

    path = artifact_path("mp-medium-0b")
    legacy = legacy_model(path)
    assert {str(p.dtype) for p in legacy.parameters()} == {"torch.float64"}
    assert sorted(
        key for key, value in legacy.state_dict().items() if value.dtype == torch.float32
    ) == [f"radial_embedding.distance_transform.{name}" for name in (
        "a", "covalent_radii", "p", "q"
    )]
    atoms = load_fixtures()["triclinic_bulk"]
    published = legacy_outputs(
        MACECalculator(model_paths=str(path), device="cpu", default_dtype="float64"),
        atoms,
    )["energy"]
    uniform = legacy_outputs(
        MACECalculator(models=[legacy.double()], device="cpu", default_dtype="float64"),
        atoms,
    )["energy"]
    assert abs(float(published - uniform)) > 1e-7


@pytest.mark.network
@pytest.mark.polar
def test_the_charge_aware_conversion_reproduces_its_committed_golden(fp64, tmp_path):
    """Through the calculator, on a checkpoint written from the conversion,
    so the whole surface the golden pins is compared: the charges, spins,
    densities, Fukui functions and the three energy terms beside the energy
    and forces. At the float32 row, which is where the float32 artifact's own
    constants put it. The stress is left out: the frozen tree's leaves out
    the long-range energy's cell dependence."""
    from mace_torch.calculators.ase_calculator import MACECalculator
    from mace_torch.train import write_model

    from tests.golden import harness
    from tests.golden.targets.foundation_references import POLAR_FIXTURES

    imported, _ = converted(artifact_path("polar-1-s"))
    checkpoint = write_model(tmp_path / "polar", imported.engine, imported.metadata)
    calculator = MACECalculator(model_paths=str(checkpoint), device="cpu")
    snapshot = harness.snapshot_outputs(
        calculator,
        harness.load_fixtures(names=list(POLAR_FIXTURES)),
        dtype="float64",
        device="cpu",
        backend="reference",
    )
    reference = harness.load_reference(
        harness.REFERENCES_DIR / "polar_foundation_cpu_fp64.json"
    )
    channels = sorted(
        {
            channel
            for entry in reference["fixtures"].values()
            for channel in entry["outputs"]
            if channel != "stress"
        }
    )
    harness.compare_to_reference(
        snapshot, reference, row=FLOAT32.name, channels=channels
    )
