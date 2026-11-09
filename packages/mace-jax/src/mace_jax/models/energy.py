"""Single-head energy output layer and composition with the JAX backbone."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.config.model import ModelConfig
from mace_core.kernels.canonical import SILU_SECOND_MOMENT_SCALE
from mace_core.kernels.protocol import KernelBackend
from mace_core.kernels.specs import LinearSpec
from mace_core.outputs import MACEOutput

from mace_jax.kernels.ops import CanonicalOp
from mace_jax.nn.backbone import MACEBackbone
from mace_jax.nn.radial import zbl


class Readout(nnx.Module):
    def __init__(self, backend, irreps, middle, nonlinear, dtype):
        self.nonlinear = nonlinear
        if nonlinear:
            self.first = backend.make_linear(
                LinearSpec(irreps_in=irreps, irreps_out=middle, precision=dtype)
            )
            self.second = backend.make_linear(
                LinearSpec(irreps_in=middle, irreps_out="0e", precision=dtype)
            )
        else:
            self.linear = backend.make_linear(
                LinearSpec(irreps_in=irreps, irreps_out="0e", precision=dtype)
            )

    def __call__(self, features):
        value = (
            self.second(SILU_SECOND_MOMENT_SCALE * jax.nn.silu(self.first(features)))
            if self.nonlinear
            else self.linear(features)
        )
        return value[:, 0]


class MACEOutputs(nnx.Module):
    """Per-layer readouts, stored E0s, scale/shift and separate energy reductions."""

    def __init__(self, backend, config, layer_irreps, elements, precision):
        self.precision = precision
        self.scale_shift = config.model == "scale_shift"
        middle = "+".join(
            f"{mul}x0e"
            for mul, ir in Irreps.parse(config.readout.mlp_irreps)
            if ir.degree == 0 and ir.parity == 1
        )
        self.readouts = nnx.List(
            [
                Readout(
                    backend,
                    irreps,
                    middle,
                    index > 0 and index == len(layer_irreps) - 1,
                    precision.model,
                )
                for index, irreps in enumerate(layer_irreps)
            ]
        )
        self.energy = CanonicalOp(
            {
                "e0_table": jnp.zeros((1, elements), dtype=jnp.float64),
                "scale": jnp.ones((1,), dtype=precision.model),
                "shift": jnp.zeros((1,), dtype=precision.model),
            }
        )

    def __call__(self, layers, graph, repulsion):
        sites = sum(
            readout(features)
            for readout, features in zip(self.readouts, layers, strict=True)
        )
        if self.scale_shift:
            sites = sites + repulsion
        sites = sites * self.energy.tensor("scale")[0] + self.energy.tensor("shift")[0]
        if not self.scale_shift:
            sites = sites + repulsion
        elements = jnp.argmax(graph["node_attrs"], axis=-1)
        e0 = self.energy.tensor("e0_table")[0, elements].astype(self.precision.model)
        count = graph["ptr"].shape[0] - 1
        total = jax.ops.segment_sum(
            sites.astype(self.precision.accumulate), graph["batch"], num_segments=count
        ).astype(self.precision.model)
        total += jax.ops.segment_sum(
            e0.astype(self.precision.accumulate), graph["batch"], num_segments=count
        ).astype(self.precision.model)
        node_dtype = (
            self.precision.accumulate if self.scale_shift else self.precision.model
        )
        return MACEOutput(
            total_energy=total,
            node_energies=sites.astype(node_dtype) + e0.astype(node_dtype),
        )

    def canonical_components(self):
        result = {"model.outputs.energy_readout": self.energy}
        for index, readout in enumerate(self.readouts):
            prefix = f"model.outputs.readouts.energy.readouts.{index}"
            if readout.nonlinear:
                result[prefix + ".first"] = readout.first
                result[prefix + ".second"] = readout.second
            else:
                result[prefix] = readout.linear
        return result


class MACEModel(nnx.Module):
    """Backbone + learnable outputs; derivatives are evaluated around this call."""

    def __init__(
        self,
        backend: KernelBackend,
        config: ModelConfig,
        atomic_numbers: tuple[int, ...],
        precision,
    ):
        import ase.data

        self.config = config
        elements = len(atomic_numbers)
        self.backbone = MACEBackbone(backend, config, elements, precision.model)
        self.outputs = MACEOutputs(
            backend, config, self.backbone.layer_irreps, elements, precision
        )
        self.repulsion = (
            CanonicalOp(
                {
                    "screening_coefficients": jnp.asarray(
                        [0.1818, 0.5099, 0.2802, 0.02817], dtype=precision.model
                    ),
                    "covalent_radii": jnp.asarray(
                        ase.data.covalent_radii, dtype=precision.model
                    ),
                    "screening_length_exponent": jnp.asarray(
                        0.3, dtype=precision.model
                    ),
                    "screening_length_prefactor": jnp.asarray(
                        0.4543, dtype=precision.model
                    ),
                }
            )
            if config.pair_repulsion
            else None
        )

    def __call__(self, graph):
        layers, lengths = self.backbone(graph)
        repulsion = (
            0
            if self.repulsion is None
            else zbl(
                lengths,
                graph["node_atomic_numbers"],
                graph["edge_index"],
                self.repulsion.tensor,
                self.config.cutoff.polynomial_order,
            )
        )
        return self.outputs(layers, graph, repulsion)

    def canonical_components(self):
        result = {
            **self.backbone.canonical_components(),
            **self.outputs.canonical_components(),
        }
        if self.repulsion is not None:
            result["model.repulsion"] = self.repulsion
        return result
