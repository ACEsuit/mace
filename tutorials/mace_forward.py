"""An inspectable, single-head ScaleShiftMACE energy forward in plain JAX.

Arrays encode real, component-normalised irreps in multiplicity/irrep order.
This teaching implementation deliberately uses dense CG tables and Python loops.
It does not train a model or implement forces, long-range physics or deployment.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

SILU_SECOND_MOMENT_SCALE = 1.679176792398942


def load_fixture():
    """Load stored weights and a molecule, without importing a MACE package."""
    return json.loads(Path(__file__).with_name("fixture.json").read_text())


def spherical_harmonics(vectors, max_ell):
    """Real component harmonics through ell=3, with y as the polar axis."""
    if not 0 <= max_ell <= 3:
        raise ValueError("The educational harmonics cover ell=0,...,3")
    unit = vectors / jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    x, y, z = unit.T
    blocks = [jnp.ones((len(vectors), 1))]
    if max_ell >= 1:
        blocks.append(jnp.sqrt(3.0) * unit)
    if max_ell >= 2:
        blocks.append(
            jnp.stack(
                [
                    jnp.sqrt(15.0) * x * z,
                    jnp.sqrt(15.0) * x * y,
                    jnp.sqrt(5.0) / 2 * (2 * y * y - x * x - z * z),
                    jnp.sqrt(15.0) * y * z,
                    jnp.sqrt(15.0) / 2 * (z * z - x * x),
                ],
                axis=-1,
            )
        )
    if max_ell >= 3:
        blocks.append(
            jnp.stack(
                [
                    jnp.sqrt(35 / 8) * x * (3 * z * z - x * x),
                    jnp.sqrt(105.0) * x * y * z,
                    jnp.sqrt(21 / 8) * x * (4 * y * y - x * x - z * z),
                    jnp.sqrt(7.0) / 2 * y * (2 * y * y - 3 * x * x - 3 * z * z),
                    jnp.sqrt(21 / 8) * z * (4 * y * y - x * x - z * z),
                    jnp.sqrt(105.0) / 2 * y * (z * z - x * x),
                    jnp.sqrt(35 / 8) * z * (z * z - 3 * x * x),
                ],
                axis=-1,
            )
        )
    return jnp.concatenate(blocks, axis=-1)


def radial_embedding(distances, settings):
    """Bessel functions multiplied by a smooth polynomial cutoff."""
    radius = settings["r_max"]
    order = settings["num_polynomial_cutoff"]
    scaled = distances / radius
    cutoff = (
        1
        - (order + 1) * (order + 2) / 2 * scaled**order
        + order * (order + 2) * scaled ** (order + 1)
        - order * (order + 1) / 2 * scaled ** (order + 2)
    )
    cutoff = jnp.where(scaled < 1, cutoff, 0)
    frequencies = jnp.arange(1, settings["num_bessel"] + 1) * jnp.pi / radius
    # sinc has the analytic limit at zero separation.
    bessel = (
        jnp.sqrt(2 / radius)
        * frequencies
        * jnp.sinc(distances[:, None] * frequencies / jnp.pi)
    )
    return bessel * cutoff[:, None]


def equivariant_linear(features, parameters, attributes=None):
    """Apply stored weights with their separate path normalisation.

    For a species-dependent map, the one-hot attributes select a weight row.
    Rows and columns enumerate the nonzero entries of an equivariant matrix;
    source ties the same weight across the components of each irrep.
    """
    weight = jnp.asarray(parameters["weight"]) * jnp.asarray(
        parameters["path_normalization"]
    )
    if attributes is not None:
        weight = attributes @ weight
    row = jnp.asarray(parameters["row"], dtype=int)
    column = jnp.asarray(parameters["column"], dtype=int)
    source = jnp.asarray(parameters["source"], dtype=int)
    contributions = features[:, column] * weight[..., source]
    return (
        jnp.zeros((len(features), parameters["dim_out"])).at[:, row].add(contributions)
    )


def channel_major(features, terms, channels):
    """Move channels inside each irrep block onto a common axis."""
    blocks, offset = [], 0
    for multiplicity, dimension in terms:
        width = multiplicity * dimension
        blocks.append(
            features[:, offset : offset + channels * width].reshape(
                len(features), channels, width
            )
        )
        offset += channels * width
    return jnp.concatenate(blocks, axis=-1)


def irrep_major(features, terms):
    blocks, offset = [], 0
    for multiplicity, dimension in terms:
        width = multiplicity * dimension
        blocks.append(features[..., offset : offset + width].reshape(len(features), -1))
        offset += width
    return jnp.concatenate(blocks, axis=-1)


def normalized_silu(values):
    return SILU_SECOND_MOMENT_SCALE * jax.nn.silu(values)


def radial_network(radial, parameters):
    values = radial
    for index, (weight, normalization) in enumerate(
        zip(
            parameters["radial_weights"],
            parameters["radial_normalization"],
            strict=True,
        )
    ):
        values = values @ (jnp.asarray(weight) * normalization)
        if index + 1 < len(parameters["radial_weights"]):
            values = normalized_silu(values)
    return values


def interaction(
    features, attributes, harmonics, radial, senders, receivers, parameters, channels
):
    """Couple sender features to edge harmonics and sum into receivers."""
    raised = equivariant_linear(features, parameters["linear_up"])
    raised = channel_major(raised, parameters["node_terms"], channels)
    coefficients = jnp.asarray(parameters["coefficients"])
    weights = radial_network(radial, parameters).reshape(
        len(senders), len(coefficients), channels
    )
    messages = jnp.zeros((len(features), channels, coefficients.shape[1]))
    for edge, (sender, receiver) in enumerate(zip(senders, receivers, strict=True)):
        # p: coupling path, c: channel, i/a/o: input/edge/output component.
        message = jnp.einsum(
            "ci,a,poia,pc->co",
            raised[sender],
            harmonics[edge],
            coefficients,
            weights[edge],
        )
        messages = messages.at[receiver].add(message)
    messages = (
        equivariant_linear(
            irrep_major(messages, parameters["path_terms"]), parameters["linear"]
        )
        / parameters["neighbors"]
    )
    if parameters["residual"]:
        return messages, equivariant_linear(features, parameters["skip"], attributes)
    return equivariant_linear(messages, parameters["skip"], attributes), None


def symmetric_contraction(features, attributes, parameters, channels):
    """Contract explicit tensor powers against the full symmetric CG basis.

    Each order couples that many neighbour-density factors; order two already
    contains correlations between pairs of neighbours. Dense powers keep the
    mathematics visible, at the cost of exponential memory in this tutorial.
    """
    values = channel_major(features, parameters["in_terms"], channels)
    outputs = []
    for orders in parameters["groups"]:
        result = None
        for item in orders:
            power = values
            for _ in range(1, item["order"]):
                power = jnp.einsum("nci,ncj->ncij", power, values).reshape(
                    len(values), channels, -1
                )
            basis = jnp.asarray(item["basis"]).reshape(item["shape"])
            basis = basis.reshape(basis.shape[0], basis.shape[1], power.shape[-1])
            paths = jnp.einsum("nci,poi->ncpo", power, basis)
            weight = jnp.einsum("ne,epc->npc", attributes, jnp.asarray(item["weight"]))
            contribution = jnp.einsum("ncpo,npc->nco", paths, weight)
            result = contribution if result is None else result + contribution
        outputs.append(result.reshape(len(values), -1))
    return jnp.concatenate(outputs, axis=-1)


def readout(features, parameters):
    if "linear" in parameters:
        return equivariant_linear(features, parameters["linear"])[:, 0]
    hidden = normalized_silu(equivariant_linear(features, parameters["first"]))
    return equivariant_linear(hidden, parameters["second"])[:, 0]


def forward(model, graph, *, return_intermediates=False):
    """Return one energy per structure; E0 is added outside scale and shift."""
    positions = jnp.asarray(graph["positions"])
    senders, receivers = graph["senders"], graph["receivers"]
    vectors = positions[jnp.asarray(receivers)] - positions[jnp.asarray(senders)]
    attributes = (
        jnp.asarray(graph["numbers"])[:, None] == jnp.asarray(model["species"])[None, :]
    )
    attributes = attributes.astype(positions.dtype)
    harmonics = spherical_harmonics(vectors, model["settings"]["max_ell"])
    radial = radial_embedding(jnp.linalg.norm(vectors, axis=-1), model["settings"])
    features = equivariant_linear(attributes, model["embedding"])
    intermediate = {"embedding": features, "harmonics": harmonics, "radial": radial}
    node_energy = jnp.zeros(len(positions))
    for index, layer in enumerate(model["layers"]):
        messages, residual = interaction(
            features,
            attributes,
            harmonics,
            radial,
            senders,
            receivers,
            layer,
            model["settings"]["num_channels"],
        )
        products = symmetric_contraction(
            messages,
            attributes,
            layer["contraction"],
            model["settings"]["num_channels"],
        )
        features = equivariant_linear(products, layer["product_linear"])
        if residual is not None:
            features = features + residual
        node_energy = node_energy + readout(features, layer["readout"])
        intermediate[f"layer_{index + 1}"] = features
    batch = jnp.asarray(graph["batch"])
    interaction_energy = (
        jnp.zeros(graph["num_graphs"])
        .at[batch]
        .add(model["scale"][0] * node_energy + model["shift"][0])
    )
    e0 = attributes @ jnp.asarray(model["e0"][0])
    energy = interaction_energy + jnp.zeros(graph["num_graphs"]).at[batch].add(e0)
    intermediate["node_readout_sum"] = node_energy
    return (energy, intermediate) if return_intermediates else energy


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    fixture = load_fixture()
    print(forward(fixture, fixture["graph"]))
