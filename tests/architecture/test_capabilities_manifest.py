"""The capabilities manifest, and the CI configuration derived from it.

`capabilities.toml` records where every capability of MACE lives during the
rewrite. Three things are generated from it -- the paths-filter entries of the
extension jobs, the per-file coverage floors, and the xfail markers of the
per-debt fitness tests -- and the tests here hold the generation direction:
the manifest is edited, the generated regions are not.

The direction is the point. A hand-maintained copy of either list only ever
goes stale in the direction nothing checks. A capability that ports to
`packages/` keeps an extension job watching paths its code has left, and keeps
a coverage floor on a legacy file that `coverage report --include` then matches
nothing against, which exits 1 on "No data to report" -- a message that reads
as a broken coverage run rather than as a floor whose subject moved.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tests.architecture import capabilities, generate_ci

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = REPO_ROOT / "tests" / "golden" / "feature_inventory.md"
CI_EXTENSIONS = REPO_ROOT / ".github" / "workflows" / "ci-extensions.yaml"

#: The inventory's capability rows pin their own `CAPABILITY_PROBES` entry, and
#: that is the shape this reads: `tests/conftest.py::CAPABILITY_PROBES[gpu]`.
#: Keying on the pin rather than on the row id is deliberate -- the inventory
#: has fourteen `marker.*` rows and only eleven of them are capabilities.
INVENTORY_PROBE_PIN = re.compile(r"CAPABILITY_PROBES\[(?P<name>[a-z_]+)\]")

#: The three registered markers that are costs or infrastructure, not
#: capabilities. `marker.timeout` in particular says so in as many words: it
#: has no `CAPABILITY_PROBES` entry and must not be absorbed into this
#: manifest. They are named here so that absorbing one fails a test.
NOT_CAPABILITIES = ("slow", "benchmark", "timeout")


# ---------------------------------------------------------------------------
# The manifest itself
# ---------------------------------------------------------------------------


def test_the_manifest_is_valid():
    """The same checks `check_capabilities_manifest.py` runs."""
    problems = capabilities.problems()
    assert not problems, "\n".join(f"- {problem}" for problem in problems)


def test_the_manifest_holds_every_capability_probe():
    """Asserted against `CAPABILITY_PROBES` itself, never a list retyped here.

    Two of the eleven are routinely dropped from hand-written enumerations --
    `magnetic`, which carries the whole MagneticMACE family and two default
    property keys, and `wandb`, which the six `--wandb` flags need -- and both
    have their own job in `ci-extensions.yaml`. A manifest short by one would
    be a capability with no recorded migration state and nothing to say so.
    """
    from tests.conftest import CAPABILITY_PROBES  # noqa: PLC0415

    declared = set(capabilities.probes())
    assert declared == set(CAPABILITY_PROBES), (
        f"the manifest declares {sorted(declared)} as probes and "
        f"tests/conftest.py declares {sorted(CAPABILITY_PROBES)}; "
        f"missing here: {sorted(set(CAPABILITY_PROBES) - declared)}, "
        f"invented here: {sorted(declared - set(CAPABILITY_PROBES))}"
    )
    assert len(declared) == 11, (
        f"there are eleven capability probes; this manifest has {len(declared)}"
    )


@pytest.mark.parametrize("marker", NOT_CAPABILITIES)
def test_a_cost_marker_is_not_a_capability(marker):
    """`slow`, `benchmark` and `timeout` are registered markers with no probe.

    They select how expensive a test is, or make collection work when a plugin
    is absent. None of them has a migration state, and putting one here would
    give the generator a filter and a floor for something that is not a
    feature.
    """
    assert marker not in capabilities.load()


def test_every_probe_is_carried_by_an_axis():
    """A probe with no axis is an optional dependency no ticket ever retires.

    This is where a dependency would go missing: the probes are the visible
    list, the axes are what the RET-* tickets delete, and a probe pointing at
    no axis is a capability that survives the migration by not being noticed.
    """
    orphans = [
        name for name, cap in capabilities.probes().items() if not cap.owned_by
    ]
    assert not orphans, f"probes owned by no axis: {orphans}"


def test_every_axis_maps_onto_a_retirement_ticket():
    """One axis, one RET ticket. The axes exist to be deleted.

    A second axis sharing a burn step would mean one deletion-only pull
    request retiring two capabilities, which is precisely the shape the RET-*
    tickets are split to avoid.
    """
    steps = {}
    for name, cap in capabilities.axes().items():
        assert cap.burn_step.startswith(("RET-", "DEP-")), (
            f"{name}: burn_step {cap.burn_step!r} is not a retirement ticket"
        )
        assert cap.burn_step not in steps, (
            f"{name} and {steps[cap.burn_step]} share burn step "
            f"{cap.burn_step}; one deletion-only pull request cannot retire two"
        )
        steps[cap.burn_step] = name


def test_every_axis_names_a_legacy_surface_that_exists():
    """An axis whose legacy files are gone is `retired`, not `legacy`.

    The surface is what the axis's RET ticket deletes, so a path that does not
    resolve means either the deletion happened without a state change, or the
    manifest names the wrong file.
    """
    missing = {}
    for name, cap in capabilities.axes().items():
        assert cap.legacy_surface, f"{name}: no legacy_surface recorded"
        for pattern in cap.legacy_surface:
            if not list(REPO_ROOT.glob(pattern)):
                missing.setdefault(name, []).append(pattern)
    assert not missing, (
        f"axes naming a legacy surface that does not resolve: {missing}. If it "
        f"was deleted, move the capability's state on instead of editing the "
        f"path away"
    )


def test_every_state_in_the_manifest_is_a_declared_state():
    for name, cap in capabilities.load().items():
        assert cap.state in capabilities.STATES, f"{name}: {cap.state!r}"


# ---------------------------------------------------------------------------
# Against the feature inventory (P0-0)
# ---------------------------------------------------------------------------


def _inventory_probes() -> set:
    text = INVENTORY.read_text(encoding="utf-8")
    return set(INVENTORY_PROBE_PIN.findall(text))


@pytest.mark.skipif(
    not INVENTORY.is_file(), reason="the feature inventory is not part of an install"
)
def test_every_inventory_capability_appears_exactly_once():
    """The acceptance criterion, read off the inventory rather than a list.

    The inventory pins each capability row to its own `CAPABILITY_PROBES`
    entry, so its capability set is derivable rather than transcribed. "Exactly
    once" is structural in TOML -- a duplicate table is a parse error -- so
    what this really asserts is the two-way match: nothing in the inventory is
    missing here, and nothing here is invented.
    """
    inventory = _inventory_probes()
    assert inventory, (
        "no CAPABILITY_PROBES pins found in the inventory; the pin format "
        "changed and this test is now reading nothing"
    )
    declared = set(capabilities.probes())
    assert inventory == declared, (
        f"in the inventory but not the manifest: {sorted(inventory - declared)}; "
        f"in the manifest but not the inventory: {sorted(declared - inventory)}"
    )


# ---------------------------------------------------------------------------
# The generated CI configuration
# ---------------------------------------------------------------------------


def test_the_generated_ci_matches_the_manifest():
    """The gate on the generation direction.

    A pull request that edits a filter entry or a floor by hand fails here,
    with the command that puts it back.
    """
    stale = [str(path.relative_to(REPO_ROOT)) for path in generate_ci.drift()]
    assert not stale, (
        f"these files' generated regions disagree with capabilities.toml: "
        f"{stale}. Edit the manifest, then run\n"
        f"    python3 tests/architecture/generate_ci.py --write"
    )


def test_the_generator_covers_more_than_one_file():
    """Both coverage shapes, and the filters. A generator that emits into one
    place would leave the other hand-maintained, which is the state this
    replaces."""
    files = {path.name for path in generate_ci.files()}
    assert files == {"ci-extensions.yaml", "nightly.yaml", "ci-core.yaml"}, files


def _extensions() -> dict:
    return yaml.safe_load(CI_EXTENSIONS.read_text(encoding="utf-8"))


def test_every_generated_filter_output_is_read_by_a_job():
    """A filter nothing consumes is a capability CI silently stopped running."""
    workflow = _extensions()
    outputs = set(workflow["jobs"]["changes"]["outputs"])
    consumed = set()
    for name, job in workflow["jobs"].items():
        if name == "changes":
            continue
        condition = str(job.get("if", ""))
        for output in outputs:
            if f"needs.changes.outputs.{output}" in condition:
                consumed.add(output)
    assert outputs == consumed, (
        f"filter outputs no job reads: {sorted(outputs - consumed)}; "
        f"jobs reading an output that is not generated: "
        f"{sorted(consumed - outputs)}"
    )


def test_every_filter_output_belongs_to_a_capability():
    workflow = _extensions()
    outputs = set(workflow["jobs"]["changes"]["outputs"])
    declared = {entry.name for entry in capabilities.ci_filters()}
    assert outputs == declared, (
        f"the workflow's filter outputs are {sorted(outputs)} and the manifest "
        f"declares {sorted(declared)}"
    )


def test_every_filter_starts_from_the_shared_set():
    """A change to the model, the tools or the data layer can break any of them.

    The shared anchor is also how the manifest itself gets into every filter:
    editing `capabilities.toml` changes what the jobs watch, so a pull request
    that touches it has to be able to start them.
    """
    workflow = _extensions()
    step = [
        candidate
        for candidate in workflow["jobs"]["changes"]["steps"]
        if "paths-filter" in str(candidate.get("uses", ""))
    ][0]
    filters = yaml.safe_load(step["with"]["filters"])
    shared = filters["shared"]
    assert "tests/architecture/capabilities.toml" in shared, (
        "the manifest is not in the shared filter set, so a change to it "
        "generates new filters and starts none of the jobs they describe"
    )
    for name, paths in filters.items():
        if name == "shared":
            continue
        assert paths[0] == shared, f"filter {name} does not start from the shared set"


def test_the_coverage_floors_are_declared_by_a_capability():
    """Every floor has an owner, and the owner is what moves it.

    A floor protects a behaviour rather than a legacy file. Recording which
    capability owns each one is what turns "move the floor when the module
    moves" from something to remember into a manifest edit the generator
    propagates.
    """
    floors = capabilities.coverage_floors()
    assert floors, "no coverage floors declared in the manifest"
    for floor in floors:
        owner = capabilities.load()[floor.owner]
        assert owner.is_axis, (
            f"{floor.path} is owned by {floor.owner}, which is a probe. A "
            f"floor moves when a functional axis ports, so its owner is an axis"
        )
