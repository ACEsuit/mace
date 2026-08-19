"""The published foundation models reproduce their committed references.

Three artifacts, two tiers. The tracked MACE-MPA-0 medium checkpoint is in
this repository, so `mace_mp()` with no model argument short-circuits to it
and touches no network: that golden carries no marker and runs on every pull
request, on all four python versions, in a fresh offline clone. MACE-MP-0
small and MACE-OFF23 small are downloads, so they are `network`-marked and
run in the nightly `foundations` job, which exports MACE_REQUIRE_CAPS with
`network` -- there a broken download host or a re-uploaded artifact fails the
job instead of skipping it green.

What is being pinned is not "some model called MPA-0" but a specific file.
Every test here goes through `tests/golden/foundation_artifacts.py`, which
carries the loader call the reference was generated with and the sha256 of
the bytes it was generated from, and which observes `torch.load` so the
digest is taken from the file the calculator actually opened rather than from
a path this test reconstructed.
"""

import contextlib
import pathlib

import numpy as np
import pytest
import torch

from tests.golden import foundation_artifacts as fa
from tests.golden import harness

#: pytest params: the two downloaded tiers carry the network marker, the
#: tracked one carries nothing at all.
ARTIFACT_PARAMS = [
    pytest.param(
        name,
        marks=[pytest.mark.network] if spec.network else [],
        id=name,
    )
    for name, spec in sorted(fa.ARTIFACTS.items())
]


def _fixtures_for(spec):
    return harness.load_fixtures(
        names=list(spec.fixture_names), tags=spec.fixture_tags or None
    )


def _load(spec):
    """Load the calculator, refusing any download for the tracked tier.

    The tracked tier's promise is that it needs no network, and a machine
    that has already downloaded everything cannot tell that promise from a
    warm cache. Under `no_network()` a download attempt raises, so the
    per-PR golden is exercised the way a fresh offline clone would run it.
    """
    guard = contextlib.nullcontext() if spec.network else fa.no_network()
    with guard:
        return fa.load_calculator(spec)


@pytest.mark.parametrize("name", ARTIFACT_PARAMS)
def test_foundation_model_reproduces_its_reference(name):
    spec = fa.ARTIFACTS[name]
    calc, _ = _load(spec)
    snapshot = harness.snapshot_outputs(
        calc,
        _fixtures_for(spec),
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    reference = harness.load_reference(harness.REFERENCES_DIR / spec.reference)
    harness.compare_to_reference(
        snapshot, reference, row=harness.FP64_CPU_REFERENCE.name
    )


@pytest.mark.parametrize("name", ARTIFACT_PARAMS)
def test_the_snapshot_came_from_the_pinned_artifact(name):
    """Identity, measured on the bytes the loader read.

    An alias is a name, and names get re-pointed upstream. Without this, a
    reference generated from one release and a test run against its
    replacement agree on the file name, disagree on the numbers, and the
    failure reads as a physics regression in MACE.
    """
    spec = fa.ARTIFACTS[name]
    _, checkpoint = _load(spec)
    digest = fa.sha256_of(checkpoint)
    reference = harness.load_reference(harness.REFERENCES_DIR / spec.reference)
    assert digest == spec.sha256, (
        f"{name}: the checkpoint at {checkpoint} is not the artifact this "
        f"golden pins (sha256 {digest}, expected {spec.sha256}). Either the "
        f"upstream artifact was replaced or a stale download is cached; "
        f"neither is fixed by regenerating the reference."
    )
    assert reference["provenance"]["sha256"] == spec.sha256
    assert reference["provenance"]["source"] == spec.origin


@pytest.mark.parametrize("name", ARTIFACT_PARAMS)
def test_the_registry_url_is_still_the_package_url(name):
    """A download tier pins the URL `foundations_models.py` actually uses."""
    spec = fa.ARTIFACTS[name]
    if not spec.network:
        from mace.calculators import foundations_models  # noqa: PLC0415

        # Compared by suffix, not by absolute path: CI installs the wheel, so
        # the package's idea of this path is inside site-packages while the
        # registry names it relative to the repository.
        installed = foundations_models.local_model_path.replace("\\", "/")
        assert installed.endswith(spec.origin), (
            f"the tracked tier names {spec.origin}, the package short-circuits "
            f"to {installed}"
        )
        return
    assert spec.origin == fa.expected_origin_url(spec), (
        f"{name}: {spec.url_table[0]}[{spec.url_table[1]!r}] no longer points "
        f"at the artifact this reference was generated from"
    )


def test_an_unqualified_mace_mp_is_mpa0_medium_and_reads_no_url():
    """The alias trap this ticket exists around.

    `mace_mp()` with no argument used to mean the 2023-12-03 L1 model and now
    means MPA-0 medium, and both files are tracked in the same directory
    (`2023-12-03-mace-mp.model` is still there). A golden that assumed the
    old meaning would pin a model nobody loads.
    """
    from mace.calculators import foundations_models  # noqa: PLC0415

    with fa.tracked_checkpoint_in_place(), fa.no_network():
        resolved = foundations_models.download_mace_mp_checkpoint(None)
        assert resolved == foundations_models.local_model_path
    assert resolved.endswith("mace-mpa-0-medium.model")
    spec = fa.ARTIFACTS["mpa0_medium"]
    assert fa.sha256_of(resolved) == spec.sha256
    # and it is a different file from the pre-3.10 default, which also ships
    older = fa.REPO_ROOT / "mace/calculators/foundations_models/2023-12-03-mace-mp.model"
    if older.exists():
        assert fa.sha256_of(older) != spec.sha256


def test_the_tracked_checkpoint_resolves_to_a_real_file_in_either_install():
    """The packaging fact the per-PR tier stands on, pinned rather than assumed.

    "It is in the tree" is true of the repository and false of the installed
    package: setup.cfg declares no package_data, so the wheel is ~300 KB and
    carries `foundations_models.py` without the directory of the same name --
    and CI installs the wheel. Unpatched, `mace_mp()` in the per-PR job would
    resolve `local_model_path` to a missing file and download the artifact
    from the release, which is precisely what a per-PR golden must not do.

    `tracked_checkpoint_in_place()` closes that gap by pointing the package at
    the copy in the checkout, and this is the assertion that the gap is really
    closed in whichever way this environment is installed. It also fails, with
    a digest mismatch, if the checkout's copy is not the pinned artifact.
    """
    from mace.calculators import foundations_models  # noqa: PLC0415

    spec = fa.ARTIFACTS["mpa0_medium"]
    with fa.tracked_checkpoint_in_place():
        resolved = pathlib.Path(foundations_models.local_model_path)
        assert resolved.is_file(), (
            f"{resolved} does not exist, so an unqualified mace_mp() would "
            f"download the checkpoint instead of reading it"
        )
        assert fa.sha256_of(resolved) == spec.sha256


def test_a_wheel_shaped_install_would_download_without_the_redirect(
    monkeypatch, tmp_path
):
    """The measurement behind the paragraph above, so it cannot rot into prose.

    With `local_model_path` pointing where a wheel install puts it -- at
    nothing -- and an empty download cache, `download_mace_mp_checkpoint(None)`
    goes to the release. That is the failure the per-PR tier would hit in CI,
    and it is silent on any machine whose cache is already warm, which is
    every machine a developer would test this on.
    """
    from mace.calculators import foundations_models  # noqa: PLC0415

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        foundations_models,
        "local_model_path",
        str(tmp_path / "site-packages/mace/calculators/foundations_models/"
            "mace-mpa-0-medium.model"),
    )
    with fa.no_network():
        with pytest.raises(fa.NetworkAccessRefused):
            foundations_models.download_mace_mp_checkpoint(None)
        # and with the redirect in place it resolves offline, to the checkout
        with fa.tracked_checkpoint_in_place():
            resolved = foundations_models.download_mace_mp_checkpoint(None)
    assert pathlib.Path(resolved) == fa.REPO_ROOT / fa.TRACKED_MPA0


@pytest.mark.parametrize("name", ARTIFACT_PARAMS)
def test_every_evaluated_fixture_is_within_the_model_element_table(name):
    """Asserted, not left to whatever the model does with an unknown Z.

    A structure containing an element the checkpoint was not fitted for is
    either an index error or -- worse -- a number, and a golden would pin the
    number. The MP tiers cover 89 elements; MACE-OFF is organic chemistry
    only, which is why its fixture selection is the molecular tag.
    """
    spec = fa.ARTIFACTS[name]
    calc, _ = _load(spec)
    supported = {int(z) for z in calc.models[0].atomic_numbers}
    for fixture, atoms in _fixtures_for(spec).items():
        present = {int(z) for z in atoms.get_atomic_numbers()}
        assert present <= supported, (
            f"{name}/{fixture}: element(s) {sorted(present - supported)} are "
            f"outside the checkpoint's table {sorted(supported)}"
        )


@pytest.mark.parametrize("name", ARTIFACT_PARAMS)
def test_the_loader_call_forces_cpu_float64(name):
    """Dtype discipline, checked on the registry and on the loaded weights.

    `mace_mp` defaults to float32 and to CUDA when one is present, so a
    reference taken without both arguments is a reference to whatever the
    generating machine happened to be. The registry entry is what the test
    replays, so the requirement is stated there and enforced here.
    """
    spec = fa.ARTIFACTS[name]
    assert spec.loader_kwargs["default_dtype"] == "float64"
    assert spec.loader_kwargs["device"] == "cpu"
    calc, _ = _load(spec)
    parameter = next(calc.models[0].parameters())
    assert parameter.dtype == torch.float64
    assert parameter.device.type == "cpu"


@pytest.mark.parametrize("name", ARTIFACT_PARAMS)
def test_reference_carries_dtype_units_and_provenance(name):
    spec = fa.ARTIFACTS[name]
    reference = harness.load_reference(harness.REFERENCES_DIR / spec.reference)
    assert reference["dtype"] == "float64"
    assert reference["device"] == "cpu"
    assert reference["backend"] == "e3nn"
    assert reference["units"] == {"energy": "eV", "length": "Ang"}
    provenance = reference["provenance"]
    assert provenance["tolerance_row"] == harness.FP64_CPU_REFERENCE.name
    if spec.release_url:
        assert provenance["release_url"] == spec.release_url
    assert spec.loader in provenance["recipe"]
    assert "default_dtype='float64'" in provenance["recipe"]
    pinned = set(reference["fixtures"])
    assert pinned == set(_fixtures_for(spec)), (
        f"{name}: the reference pins {sorted(pinned)} but the model's fixture "
        f"selection is {sorted(_fixtures_for(spec))}"
    )
    for fixture, entry in reference["fixtures"].items():
        outputs = entry["outputs"]
        assert {"energy", "forces"} <= set(outputs), fixture
        assert ("stress" in outputs) == entry["periodic"], fixture
        for channel in outputs.values():
            assert channel["unit"]
            assert channel["kind"] in harness.KINDS
            assert np.isfinite(np.asarray(channel["value"], dtype=float)).all()


def test_the_tracked_anicc_checkpoint_cannot_be_loaded_on_a_cpu_only_host():
    """Why there is no `mace_anicc` golden, as a measurement rather than a note.

    `ani500k_large_CC.model` is tracked in this repository exactly like the
    MPA-0 checkpoint, so it looks like a second no-download golden. It is
    not: its e3nn submodules were serialised as TorchScript archives on a
    CUDA host, and e3nn's `CodeGenMixin.__setstate__` calls
    `torch.jit.load(buffer)` with no `map_location`, so the archive is
    restored to the device it was saved on and `torch.load(...,
    map_location="cpu")` still dies reaching for that device. Nothing in
    MACE can override that from the outside.

    Which error arrives is a property of the local torch wheel, not of the
    checkpoint, so both shapes are accepted here. A build without CUDA at
    all (the macOS wheel) has no kernels registered and raises
    `NotImplementedError` over the missing `CUDA` dispatch key; a CUDA build
    on a host with no driver (the linux wheel every CI job installs) gets as
    far as initialising the context and raises `RuntimeError: Found no
    NVIDIA driver`. `NotImplementedError` is a subclass of `RuntimeError`,
    so one `raises` covers both.

    If this ever starts failing, the artifact was re-exported and a
    `mace_anicc` golden becomes possible -- add it to
    `foundation_artifacts.ARTIFACTS` rather than deleting this test.
    """
    path = fa.REPO_ROOT / fa.ANICC_TRACKED_PATH
    if not path.exists():  # pragma: no cover - the file is tracked
        pytest.skip(f"{fa.ANICC_TRACKED_PATH} is not present")
    if torch.cuda.is_available():
        pytest.skip("the refusal is a property of a host without CUDA")
    with pytest.raises(RuntimeError, match="CUDA|NVIDIA"):
        torch.load(path, map_location="cpu", weights_only=False)
