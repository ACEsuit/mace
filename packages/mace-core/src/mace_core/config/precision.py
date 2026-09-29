"""Which dtype each dispatched op computes and accumulates in.

**Precision is per dispatched op, never global.** A :class:`PrecisionConfig`
answers, for each op descriptor, a :class:`PrecisionPolicy`: what the op
computes in, the floor its reductions accumulate in, and what its parameters
are held in. The policy is stamped on the descriptor when the model is built,
so a backend reads it from there and nothing is decided in ``forward``. A
global autocast would be wrong for a model that trains in float64, and a fused
span has one math dtype, so the unit has to be the dispatch.

**``accumulate`` is a floor the device has to be able to satisfy.** A backend
meets it natively, cuEquivariance as its math dtype, or by computing in it, as
the plain-torch reference does. A device with no float64 degrades a float64
floor to the model's own dtype, which is what the frozen tree's ``safe_double``
does on MPS, and the degradation is reported when the model is built rather
than found later as a dtype error.

Dtypes are names, from :mod:`mace_core.kernels.precision`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from mace_core.kernels.precision import WIDTH, Precision, widest

__all__ = ["OP_KINDS", "PrecisionConfig", "PrecisionPolicy"]

#: The op kind of each descriptor class, by name. By name so this module need
#: not import the descriptors, which import it.
_KIND_OF = {
    "LinearDescriptor": "linear",
    "ChannelwiseTPConvDescriptor": "channelwise_tp_conv",
    "SymmetricContractionDescriptor": "symmetric_contraction",
    "FullyConnectedTPDescriptor": "fully_connected_tp",
    "SegmentReduceDescriptor": "segment_reduce",
    "SphericalHarmonicsDescriptor": "spherical_harmonics",
    "RadialBasisDescriptor": "radial_basis",
}

#: The op kinds a policy can be given for.
OP_KINDS: tuple[str, ...] = tuple(_KIND_OF.values())


@dataclass(frozen=True)
class PrecisionPolicy:
    """What one dispatched op, or one fused span, computes in.

    Attributes:
        compute_dtype: What the op's arithmetic runs in. Its inputs are cast
            to it at the op's boundary and its output is cast back.
        accumulate_dtype: The floor its reductions accumulate in. At least as
            wide as ``compute_dtype``; a backend without a separate
            accumulator meets it by computing in it.
        param_dtype: What the op's weights are held in.
    """

    compute_dtype: Precision
    accumulate_dtype: Precision
    param_dtype: Precision

    def __post_init__(self) -> None:
        if WIDTH[self.accumulate_dtype] < WIDTH[self.compute_dtype]:
            raise ValueError(
                f"an accumulation floor of {self.accumulate_dtype} is narrower "
                f"than the {self.compute_dtype} the op computes in; a floor "
                f"can only widen a reduction."
            )

    @classmethod
    def uniform(cls, precision: Precision) -> PrecisionPolicy:
        """Everything in one precision."""
        return cls(precision, precision, precision)


@dataclass(frozen=True)
class PrecisionConfig:
    """The model's precision: its interface, its reductions, and each op kind.

    Attributes:
        model: The interface dtype, that of every tensor crossing an op
            boundary and of the model's own parameters outside the ops.
        accumulate: The floor of the model-level reductions: per-atom and total
            energies. The reason it may be wider than ``model``: a sum over ten
            thousand site energies loses accuracy the model's own arithmetic
            does not.
        ops: Per op kind, the policy that replaces the default. The default is
            every op computing, accumulating and holding weights in ``model``.
    """

    model: Precision = "float64"
    accumulate: Precision = "float64"
    ops: tuple[tuple[str, PrecisionPolicy], ...] = field(default=())

    def __post_init__(self) -> None:
        unknown = sorted({kind for kind, _ in self.ops} - set(OP_KINDS))
        if unknown:
            raise ValueError(
                f"{unknown} are not op kinds; a policy can be given for "
                f"{list(OP_KINDS)}."
            )

    @property
    def interface_dtype(self) -> Precision:
        """The dtype of tensors crossing an op boundary."""
        return self.model

    def policy_for(self, descriptor: Any | Sequence[Any]) -> PrecisionPolicy:
        """The policy for this dispatched op, or for this fused span.

        Authoritative: a backend reads the policy stamped from here and never
        chooses one. For a span the answer is one policy, the widest of its
        members' in each field, so the span's single math dtype meets every
        member's floor.

        Args:
            descriptor: One op descriptor, or the descriptors of a span.
        """
        if isinstance(descriptor, Sequence) and not isinstance(descriptor, str):
            policies = [self.policy_for(member) for member in descriptor]
            if not policies:
                raise ValueError("a span has at least one op")
            return PrecisionPolicy(
                compute_dtype=widest(*(p.compute_dtype for p in policies)),
                accumulate_dtype=widest(*(p.accumulate_dtype for p in policies)),
                param_dtype=widest(*(p.param_dtype for p in policies)),
            )
        kind = _KIND_OF.get(type(descriptor).__name__)
        overrides = dict(self.ops)
        if kind is not None and kind in overrides:
            return overrides[kind]
        return PrecisionPolicy.uniform(self.model)

    def resolved_accumulate(self, supports_float64: bool) -> Precision:
        """The model-level accumulation type this device can actually give.

        Args:
            supports_float64: Whether the device has float64 at all.
        """
        if self.accumulate == "float64" and not supports_float64:
            return self.model
        return self.accumulate

    def for_device(self, supports_float64: bool) -> tuple[PrecisionConfig, list[str]]:
        """This configuration with every float64 floor a device cannot meet
        degraded to the model's dtype, and what was degraded.

        The report is what the build logs: a floor quietly lowered is a sum
        less accurate than the configuration asked for.
        """
        if supports_float64:
            return self, []
        degraded: list[str] = []

        def lower(name: Precision, what: str) -> Precision:
            if name == "float64" and self.model != "float64":
                degraded.append(what)
                return self.model
            return name

        ops = tuple(
            (
                kind,
                PrecisionPolicy(
                    lower(policy.compute_dtype, f"{kind} compute"),
                    lower(policy.accumulate_dtype, f"{kind} accumulate"),
                    lower(policy.param_dtype, f"{kind} parameters"),
                ),
            )
            for kind, policy in self.ops
        )
        accumulate = lower(self.accumulate, "the energy reduction")
        return replace(self, accumulate=accumulate, ops=ops), degraded

    @classmethod
    def float64(cls) -> PrecisionConfig:
        """Everything in float64."""
        return cls(model="float64", accumulate="float64")

    @classmethod
    def float32(cls) -> PrecisionConfig:
        """The ops in float32, and the energy reduction in float64, which is
        what the frozen tree's float32 run does."""
        return cls(model="float32", accumulate="float64")

    @classmethod
    def mixed(cls) -> PrecisionConfig:
        """float32 through the model, with the two places that lose accuracy
        first held wider: the radial basis, which feeds every weight of the
        convolution, and the symmetric contraction's reductions, which sum a
        product of up to ``correlation`` factors."""
        return cls(
            model="float32",
            accumulate="float64",
            ops=(
                ("radial_basis", PrecisionPolicy.uniform("float64")),
                (
                    "symmetric_contraction",
                    PrecisionPolicy("float32", "float64", "float32"),
                ),
            ),
        )
