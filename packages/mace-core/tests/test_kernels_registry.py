"""The kernel contract: descriptors, capabilities, and finding a backend.

All of it torch-free, which is the point rather than a side effect: this is the
half of the dispatch layer that both the torch and the jax implementations read,
and a checkpoint records.
"""

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import pytest
from mace_core.kernels import (
    CANONICAL_LAYOUT,
    DISPATCHED_OPS,
    ENTRY_POINT_GROUPS,
    KERNEL_SPEC_VERSION,
    REFERENCE_ONLY_OPS,
    BackendCapabilities,
    BackendNotAvailableError,
    ChannelwiseTPConvDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
    UnsupportedDescriptorError,
    available_backends,
    canonical_weight_shape,
    get_backend,
)
from mace_core.kernels import registry as registry_module

# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def test_a_descriptor_is_hashable_so_it_can_key_a_cache():
    """Resolution happens once per distinct op, so the descriptor has to be
    usable as a key. A mutable descriptor would also mean an op could be built
    against one shape and asked for another."""
    first = LinearDescriptor(irreps_in="16x0e", irreps_out="8x0e")
    second = LinearDescriptor(irreps_in="16x0e", irreps_out="8x0e")
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    with pytest.raises(AttributeError):
        # Through `setattr`, because the assignment is the thing being tested
        # and a checker is right to reject it written out.
        setattr(first, "irreps_in", "1x0e")  # noqa: B010


def test_the_linear_weight_count_is_the_matching_multiplicities():
    """An equivariant linear map connects a term only to a term of the same
    irrep, so 1o never mixes into 0e however wide either is."""
    assert LinearDescriptor(irreps_in="16x0e", irreps_out="8x0e").weight_numel == 128
    assert LinearDescriptor(irreps_in="16x0e", irreps_out="8x1o").weight_numel == 0
    mixed = LinearDescriptor(irreps_in="16x0e+16x1o", irreps_out="32x0e+8x1o")
    assert mixed.weight_numel == 16 * 32 + 16 * 8


def test_a_bias_is_counted_and_only_on_scalars():
    """A bias on anything but 0e would break equivariance, so it is not merely
    unsupported, it is not counted."""
    plain = LinearDescriptor(irreps_in="16x0e", irreps_out="4x0e+2x1o")
    biased = LinearDescriptor(irreps_in="16x0e", irreps_out="4x0e+2x1o", has_bias=True)
    assert biased.weight_numel - plain.weight_numel == 4


def test_the_symmetric_contraction_weight_count_is_the_canonical_shape():
    descriptor = SymmetricContractionDescriptor(
        irreps_in="0e+1o+2e+3o",
        irreps_out="0e+1o",
        correlation=3,
        num_elements=2,
        num_features=16,
    )
    assert descriptor.path_count == 29
    assert descriptor.weight_numel == 2 * 29 * 16
    assert canonical_weight_shape(2, 29, 16) == (2, 29, 16)


def test_the_basis_is_a_recorded_field_and_changes_the_count():
    """The defect this contract removes: on the frozen tree this number changes
    with what happens to be installed. Here it changes only with the field."""
    common: dict[str, Any] = {
        "irreps_in": "0e+1o+2e+3o",
        "irreps_out": "0e+1o",
        "correlation": 3,
        "num_elements": 1,
        "num_features": 1,
    }
    assert SymmetricContractionDescriptor(**common, basis="reduced").path_count == 29
    assert SymmetricContractionDescriptor(**common, basis="full").path_count == 86


def test_an_op_whose_weights_come_from_outside_owns_none():
    assert ChannelwiseTPConvDescriptor(num_radial=8).weight_numel == 0


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_the_coarse_filter_rejects_a_precision_that_was_not_declared():
    capabilities = BackendCapabilities(dtypes=frozenset({"float32"}))
    assert not capabilities.supports(LinearDescriptor(precision="float64"))
    assert capabilities.supports(LinearDescriptor(precision="float32"))


def test_a_basis_the_backend_did_not_declare_is_refused_by_name():
    capabilities = BackendCapabilities(bases=frozenset({"reduced"}))
    with pytest.raises(UnsupportedDescriptorError, match="does not support"):
        capabilities.require(SymmetricContractionDescriptor(basis="full"), "pretend")


def test_an_lmax_beyond_the_declared_one_is_refused():
    capabilities = BackendCapabilities(max_lmax=2)
    assert capabilities.supports(SphericalHarmonicsDescriptor(lmax=2))
    assert not capabilities.supports(SphericalHarmonicsDescriptor(lmax=3))


def test_a_backend_may_declare_no_limit():
    assert BackendCapabilities(max_lmax=0).supports(
        SphericalHarmonicsDescriptor(lmax=11)
    )


def test_missing_double_backward_is_a_loud_rejection_naming_what_needed_it():
    """Never a warning. A first derivative taken through a backward that is not
    itself differentiable gives wrong forces rather than no forces, and wrong
    forces train a model that looks like it is working."""
    capabilities = BackendCapabilities(supports_double_backward=False)
    with pytest.raises(UnsupportedDescriptorError) as caught:
        capabilities.require_double_backward("pretend", "training on forces")
    message = str(caught.value)
    assert "training on forces" in message
    assert "pretend" in message
    assert "inference" in message


def test_double_backward_declared_passes_quietly():
    BackendCapabilities(supports_double_backward=True).require_double_backward(
        "pretend", "training on forces"
    )


def test_a_backend_can_override_the_exact_answer():
    """The coarse fields are a filter, not the authority. A backend whose limits
    depend on the shape says so here and nothing guesses for it."""

    @dataclass(frozen=True)
    class Fussy(BackendCapabilities):
        def supports(self, descriptor):
            if isinstance(descriptor, RadialBasisDescriptor):
                return descriptor.num_basis % 2 == 0
            return super().supports(descriptor)

    fussy = Fussy()
    assert fussy.supports(RadialBasisDescriptor(num_basis=8))
    assert not fussy.supports(RadialBasisDescriptor(num_basis=7))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@dataclass
class PretendBackend:
    name: str = "pretend"

    def capabilities(self):
        return BackendCapabilities(ops=DISPATCHED_OPS, supports_double_backward=True)


class PretendEntryPoint:
    def __init__(self, name, factory=None, failure=None):
        self.name = name
        self._factory = factory
        self._failure = failure

    def load(self):
        if self._failure is not None:
            raise self._failure
        return self._factory


@pytest.fixture
def registered(monkeypatch):
    def install(*entries):
        monkeypatch.setattr(
            registry_module, "entry_points", lambda group: list(entries)
        )

    return install


def test_a_backend_registered_through_an_entry_point_is_discovered(registered):
    registered(PretendEntryPoint("pretend", PretendBackend))
    found = available_backends("torch")
    assert [backend.name for backend in found] == ["pretend"]
    assert found[0].loaded
    assert get_backend("pretend", "torch").name == "pretend"


def test_a_backend_that_cannot_import_is_recorded_and_not_raised(registered):
    """A machine without a CUDA runtime is expected to carry entry points it
    cannot load. Listing what is available must work there, because that is
    exactly where somebody is trying to find out what went wrong."""
    registered(
        PretendEntryPoint("broken", failure=ImportError("libcuda.so.1 not found")),
        PretendEntryPoint("pretend", PretendBackend),
    )
    found = {backend.name: backend for backend in available_backends("torch")}
    assert not found["broken"].loaded
    assert "libcuda" in found["broken"].reason
    assert found["pretend"].loaded


def test_asking_for_a_broken_backend_raises_and_quotes_the_reason(registered):
    """Discovery records, resolution raises. Substituting another backend here
    would silently change the numbers a run produces."""
    registered(PretendEntryPoint("broken", failure=ImportError("libcuda missing")))
    with pytest.raises(BackendNotAvailableError) as caught:
        get_backend("broken", "torch")
    message = str(caught.value)
    assert "libcuda missing" in message
    assert "not being substituted" in message


def test_asking_for_a_name_nobody_registered_lists_what_there_is(registered):
    registered(PretendEntryPoint("pretend", PretendBackend))
    with pytest.raises(BackendNotAvailableError) as caught:
        get_backend("absent", "torch")
    message = str(caught.value)
    assert "['pretend']" in message
    assert ENTRY_POINT_GROUPS["torch"] in message


def test_an_unknown_framework_is_refused_rather_than_searched():
    with pytest.raises(ValueError, match="fortran"):
        available_backends("fortran")


def test_the_two_frameworks_have_separate_groups():
    assert ENTRY_POINT_GROUPS["torch"] != ENTRY_POINT_GROUPS["jax"]


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------


def test_the_op_set_is_closed_and_the_two_halves_do_not_overlap():
    assert not DISPATCHED_OPS & REFERENCE_ONLY_OPS
    assert len(DISPATCHED_OPS) == 5
    assert CANONICAL_LAYOUT == "mul_ir"
    assert KERNEL_SPEC_VERSION


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_the_contract_imports_no_framework(framework):
    """Run in a fresh interpreter: this suite imports numpy and pydantic
    already, and an in-process check would pass on whatever the session left
    behind."""
    probe = (
        "import sys, mace_core.kernels\n"
        f"assert {framework!r} not in sys.modules, "
        f"'importing mace_core.kernels pulled in {framework}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_the_protocol_is_checkable_without_a_framework():
    """The generic-over-the-tensor-type trick has to work with nothing
    installed, because the same file is read by the jax side."""
    assert importlib.util.find_spec("mace_core.kernels.protocol") is not None
