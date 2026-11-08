"""Extract a pickled legacy MACE checkpoint into the neutral weights format.

This is the one piece of the converter that runs against the legacy package,
and it runs nowhere else. It is a script, not a module: it sits in a directory
whose name is not an identifier, so nothing can import it, and it is executed
by an interpreter that has the legacy package installed, either the in-tree one
during development or a pinned environment built for the purpose. Only the two
files it writes cross back.

Loading a legacy checkpoint executes the classes it was pickled from, which is
why this has to run where they exist, and why nothing on the other side ever
unpickles anything.

What it writes is described in ``mace_core.weights.neutral_format``; this script
imports nothing from that package, so it states the same JSON by hand, and the
reader on the other side validates every field.

Every tensor the checkpoint holds is accounted for. It is carried under an op,
recorded in ``derived`` with the reason it need not be carried, or the
extraction stops and names it. A tensor dropped here would be invisible until
the converted model computed something else.

Usage::

    python extract_legacy.py SOURCE OUTPUT
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

CONVERTER_VERSION = "1.0"
FORMAT = "mace-neutral"
VERSION = "1.0"
SCHEMA_VERSION = "1.0"
SEPARATOR = "::"

#: The classes this carries across, and the model spelling each becomes. Checked
#: by identity against the legacy module, never by name or by the shape of the
#: weights: a subclass can hold tensors that look right and mean more.
SUPPORTED = {
    "MACE": "plain",
    "ScaleShiftMACE": "scale_shift",
    "AtomicDielectricMACE": "dielectric",
    "AtomicDipolesMACE": "dipole",
    "EnergyDipolesMACE": "energy_dipole",
    "MagneticScaleShiftMACE": "magnetic",
    "MagneticSCFMACE": "magnetic",
    "PolarMACE": "polar",
    "MACELES": "les",
}

#: The magnetic base implements no forward of its own.
REFUSED = {
    "MagneticMACE": (
        "the magnetic base class whose forward raises NotImplementedError; "
        "convert its executable MagneticScaleShiftMACE or MagneticSCFMACE instead"
    ),
}

#: Interaction blocks whose weights this knows how to walk.
INTERACTIONS = (
    "RealAgnosticInteractionBlock",
    "RealAgnosticResidualInteractionBlock",
    "RealAgnosticDensityInteractionBlock",
    "RealAgnosticDensityResidualInteractionBlock",
    "RealAgnosticResidualNonLinearInteractionBlock",
    "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
)

#: Readout blocks whose weights this knows how to walk.
READOUTS = (
    "LinearReadoutBlock",
    "NonLinearReadoutBlock",
    "NonLinearBiasReadoutBlock",
    "LinearDipolePolarReadoutBlock",
    "NonLinearDipolePolarReadoutBlock",
    "LinearDipoleReadoutBlock",
    "NonLinearDipoleReadoutBlock",
)


class ExtractionError(RuntimeError):
    """The checkpoint cannot be carried across faithfully."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(source: Path):
    """Unpickle the checkpoint with the legacy package in scope.

    The legacy package is imported first on purpose: its ``__init__`` enables
    full unpickling, and importing the equivariance library before it breaks
    even that library's own constants.
    """
    importlib.import_module("mace")
    torch = importlib.import_module("torch")
    return torch.load(str(source), map_location="cpu", weights_only=False)


def classify(model) -> str:
    """The model spelling, or a refusal naming the class and why."""
    candidates = {}
    for module_name in ("mace.modules.models", "mace.modules.extensions"):
        module = importlib.import_module(module_name)
        for name in (*SUPPORTED, *REFUSED):
            cls = getattr(module, name, None)
            if cls is not None:
                candidates[name] = cls
    for name, why in REFUSED.items():
        if name in candidates and type(model) is candidates[name]:
            raise ExtractionError(
                f"the checkpoint is {name}, {why}. This converter carries "
                f"{sorted(SUPPORTED)} across."
            )
    for name, spelling in SUPPORTED.items():
        if name in candidates and type(model) is candidates[name]:
            return spelling
    raise ExtractionError(
        f"the checkpoint is a {type(model).__module__}.{type(model).__name__}, "
        f"which is none of {sorted(SUPPORTED)}. A subclass is refused as well: "
        f"it can hold tensors that look convertible and mean something else."
    )


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


def legacy_config(model) -> dict[str, Any]:
    """What the legacy package itself says it takes to rebuild the model.

    Read through the legacy package's own function, which is the authoritative
    list. It refuses the plain energy and dipole classes by their names alone,
    although their common fields exist, so those classes use a view that
    reports the scale-shift class's name and forwards every attribute. Some
    releases of that function read the scale-shift block without checking it
    exists, so the view offers the identity one, which is what the plain class
    computes, and the two keys it produces are dropped afterwards: these
    classes have no scale and shift to record. Response readout details are
    read from the actual gate below.
    """
    # Typed loosely on purpose: the plain class is passed through a view that is
    # not a module, and the function reads only attributes.
    extract: Any = importlib.import_module(
        "mace.tools.scripts_utils"
    ).extract_config_mace_model
    target = model
    plain = type(model).__name__ in {"MACE", "AtomicDipolesMACE", "EnergyDipolesMACE"}
    if plain:
        torch = importlib.import_module("torch")
        identity = type(
            "IdentityScaleShift",
            (),
            {"scale": torch.ones(1), "shift": torch.zeros(1)},
        )()

        class ScaleShiftMACE:  # the name the function checks, nothing more
            def __getattr__(self, name):
                if name == "scale_shift" and not hasattr(model, name):
                    return identity
                return getattr(model, name)

        target = ScaleShiftMACE()
    config = extract(target)
    if "error" in config:
        raise ExtractionError(
            f"the legacy configuration reader refused: {config['error']}"
        )
    if plain:
        for key in ("atomic_inter_scale", "atomic_inter_shift"):
            config.pop(key, None)
    if type(model).__name__ in {
        "AtomicDielectricMACE",
        "AtomicDipolesMACE",
        "EnergyDipolesMACE",
    } and hasattr(model.readouts[-1], "equivariant_nonlin"):
        # The legacy reader hardcodes SiLU for this family. Read the actual
        # activation, including identity, from the trained readout instead.
        nonlinear = model.readouts[-1].equivariant_nonlin
        if hasattr(nonlinear, "_act_scalar"):
            config["gate"] = (
                nonlinear._act_scalar
                if len(nonlinear.irreps_scalars)
                else nonlinear._act_gate
            )
        else:
            activations = nonlinear.act_scalars.acts or nonlinear.act_gates.acts
            activation = activations[0]
            config["gate"] = getattr(activation, "f", activation)
        config["MLP_irreps"] = model.readouts[-1].hidden_irreps
    if type(model).__name__ == "MACELES":
        config["les_arguments"] = les_settings(model)
    if type(model).__name__ == "PolarMACE":
        # The cutoff the forward sums to, which is not always what the recorded
        # factor gives: the published models hold the cutoff of a factor of one
        # beside a recorded factor of 1.5.
        config["kspace_cutoff"] = float(model.kspace_cutoff)
    specs = config.get("embedding_specs")
    if specs is not None:
        # As ordered pairs: the frozen tree concatenates the embeddings in this
        # order, and the sidecar is written with its keys sorted.
        config["embedding_specs"] = [[name, spec] for name, spec in specs.items()]
    return {key: jsonable(value) for key, value in config.items()}


def les_settings(model) -> dict[str, Any]:
    """Recover what the readouts and external solver actually use, even without YAML."""
    settings = {
        "use_dipole": getattr(model, "use_dipoles", False),
        "use_quad": getattr(model, "use_quads", False),
        "use_induced_charge": getattr(model, "use_induced_charges", False),
        "use_induced_dipole": getattr(model, "use_induced_dipoles", False),
        **{
            name: getattr(model, "les_" + name, 1.0)
            for name in ("output_scale", "alpha_scale", "kappa_scale")
        },
    }
    for name, fallback in {
        "compute_bec": False,
        "bec_output_index": None,
        "use_anisotropic_polarizability": False,
        "alpha_irreps": "0e+1o+2e",
        "alpha_1o_nonlinear_readout": False,
        "alpha_1o_linear_w_pos": True,
        "make_kappa_positive": False,
        "make_alpha_positive": False,
    }.items():
        settings[name] = getattr(model, name, fallback)
    for name in (
        "use_atomwise",
        "sigma",
        "dl",
        "remove_self_interaction",
        "remove_mean",
        "epsilon_factor",
        "use_atomic_alpha",
        "use_epsilon_r_scaling",
        "use_fixed_atomic_charges",
        "fixed_atomic_charges_scaling_factor",
        "n_layers",
        "n_hidden",
        "add_linear_nn",
        "output_scaling_factor",
    ):
        settings[name] = getattr(model.les, name)
    return settings


def jsonable(value: Any) -> Any:
    """A configuration value as JSON: classes and functions by name, irreps as
    strings, tensors and arrays as lists, anything else by its repr."""
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    # Before the sequence case: an irreps declaration is a tuple underneath.
    if type(value).__name__ in {"Irreps", "Irrep"}:
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, type) or (callable(value) and hasattr(value, "__name__")):
        return value.__name__
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: jsonable(getattr(value, name)) for name in value.__dataclass_fields__
        }
    return repr(value)


# ---------------------------------------------------------------------------
# Canonical layouts
# ---------------------------------------------------------------------------


def linear_to_canonical(linear) -> tuple[Any, Any]:
    """An equivariant linear's weight and bias, output copies outermost, with
    the per-instruction normalization folded into the weight.

    e3nn walks its instructions input-term major and lays each out as a
    ``[multiplicity_in, multiplicity_out]`` block scaled in the forward by the
    instruction's path weight.
    """
    numpy = importlib.import_module("numpy")
    source = list(linear.irreps_in)
    target = list(linear.irreps_out)
    blocks = {}
    offset = 0
    flat = linear.weight.detach().cpu().numpy().astype(numpy.float64)
    # A biased linear lists each bias as an instruction with no input; the
    # biases are added unscaled and carried apart from the weights.
    for instruction in (ins for ins in linear.instructions if ins.i_in >= 0):
        rows, columns = instruction.path_shape
        span = rows * columns
        blocks[(instruction.i_in, instruction.i_out)] = (
            flat[offset : offset + span].reshape(rows, columns)
            * instruction.path_weight
        )
        offset += span
    if offset != flat.size:
        raise ExtractionError(
            f"a linear's instructions account for {offset} weights and it holds "
            f"{flat.size}, so its layout is not the one this reads."
        )
    ordered = []
    for out_index, (out_multiplicity, out_irrep) in enumerate(target):
        for out_copy in range(out_multiplicity):
            for in_index, (in_multiplicity, in_irrep) in enumerate(source):
                if in_irrep != out_irrep or (in_index, out_index) not in blocks:
                    continue
                block = blocks[(in_index, out_index)]
                ordered.extend(block[copy, out_copy] for copy in range(in_multiplicity))
    bias = linear.bias.detach().cpu().numpy().astype(numpy.float64)
    return numpy.asarray(ordered, dtype=numpy.float64), bias


def skip_to_canonical(tensor_product) -> Any:
    """The skip connection, a fully connected product against the element
    attributes, as a per-element linear map: ``[elements, plan]``, the plan
    running output copies outermost, unscaled.

    e3nn's per-instruction factor and the coupling of an irrep with a scalar
    carry ``sqrt(dim_out)`` between them in opposite directions, so what is
    folded in is the path weight over ``sqrt(dim_out)``.
    """
    numpy = importlib.import_module("numpy")
    source = list(tensor_product.irreps_in1)
    target = list(tensor_product.irreps_out)
    blocks = {}
    offset = 0
    flat = tensor_product.weight.detach().cpu().numpy().astype(numpy.float64)
    for instruction in tensor_product.instructions:
        first, second, out = instruction.path_shape
        span = first * second * out
        scale = instruction.path_weight / numpy.sqrt(target[instruction.i_out][1].dim)
        blocks[(instruction.i_in1, instruction.i_out)] = (
            flat[offset : offset + span].reshape(first, second, out) * scale
        )
        offset += span
    if offset != flat.size:
        raise ExtractionError(
            f"the skip connection's instructions account for {offset} weights "
            f"and it holds {flat.size}."
        )
    columns = []
    for out_index, (out_multiplicity, out_irrep) in enumerate(target):
        for out_copy in range(out_multiplicity):
            for in_index, (in_multiplicity, in_irrep) in enumerate(source):
                if in_irrep != out_irrep or (in_index, out_index) not in blocks:
                    continue
                block = blocks[(in_index, out_index)]
                columns.extend(
                    block[copy, :, out_copy] for copy in range(in_multiplicity)
                )
    return numpy.stack(columns, axis=1).astype(numpy.float64)


def basis_path_first(basis, order: int, target_dimension: int) -> Any:
    """A contraction's coupling basis with the path axis first and the output
    component axis always present.

    The legacy package stores the path axis last and drops the component axis
    for a scalar output.
    """
    numpy = importlib.import_module("numpy")
    values = numpy.moveaxis(basis.detach().cpu().numpy().astype(numpy.float64), -1, 0)
    if values.ndim == order + 1:
        values = values[:, None]
    if values.shape[1] != target_dimension:
        raise ExtractionError(
            f"a coupling basis of order {order} has {values.shape[1]} output "
            f"components where its target has {target_dimension}."
        )
    return numpy.ascontiguousarray(values)


#: How far a stored basis may be from the float64 construction, relative to
#: its largest entry, and still be that construction computed in float32. The
#: frozen tree builds the higher orders by float32 products, a few units of
#: float32 rounding apart; a basis of another construction is off by order one.
FLOAT32_BASIS_TOLERANCE = 1e-5


def float64_basis(contraction, target: str, order: int, stored, reduced: bool):
    """The coupling basis rebuilt at float64, when the stored one is that
    construction held at float32, and ``None`` otherwise, including when the
    two are equal.

    A model built in float32 holds its bases at float32, and keeps them so
    when it is carried to float64. Carried as they are, that error enters the
    conversion and the converted model computes neither the published
    function nor the exact one. The basis is rebuilt with the frozen tree's
    own construction, and used only when every stored entry is a float32
    number, the zeros sit in the same places and the two agree to float32
    precision: anything else is carried unchanged.
    """
    torch = importlib.import_module("torch")
    values = stored.detach().cpu().double()
    if not torch.equal(values, values.float().double()):
        return None
    construction = importlib.import_module("mace.tools.cg")
    exact = construction.U_matrix_real(
        irreps_in=contraction.coupling_irreps,
        irreps_out=target,
        correlation=order,
        use_cueq_cg=reduced,
        dtype=torch.float64,
    )[-1]
    if exact.shape != values.shape:
        return None
    if not torch.equal(exact == 0, values == 0):
        return None
    scale = float(exact.abs().max())
    difference = float((exact - values).abs().max())
    if difference == 0.0 or difference > FLOAT32_BASIS_TOLERANCE * scale:
        return None
    return exact


def channel_irreps(irreps) -> str:
    """One channel of a declaration: its irreps without multiplicities, in the
    order the declaration has them."""
    return "+".join(str(irrep) for _, irrep in irreps)


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


class Walk:
    """Collects ops and tensors, and remembers which source tensors it used."""

    def __init__(self, model):
        self.state = {key: value for key, value in model.state_dict().items()}
        self.used: set[str] = set()
        self.derived: dict[str, str] = {}
        self.ops: dict[str, dict[str, Any]] = {}
        self.tensors: dict[str, Any] = {}

    def use(self, *keys: str) -> None:
        for key in keys:
            if key not in self.state:
                raise ExtractionError(f"the checkpoint has no tensor {key!r}.")
            self.used.add(key)

    def derive(self, key: str, reason: str) -> None:
        self.derived[key] = reason

    def op(self, name: str, kind: str, tensors: dict[str, Any], **fields: Any) -> None:
        numpy = importlib.import_module("numpy")
        self.ops[name] = {
            "op_kind": kind,
            "schema_version": SCHEMA_VERSION,
            "tensors": sorted(tensors),
            **fields,
        }
        for tensor_name, value in tensors.items():
            self.tensors[f"{name}{SEPARATOR}{tensor_name}"] = numpy.ascontiguousarray(
                value
            )

    def linear(self, name: str, module, prefix: str) -> None:
        weight, bias = linear_to_canonical(module)
        self.use(f"{prefix}.weight")
        if f"{prefix}.bias" in self.state:
            self.use(f"{prefix}.bias")
        self.derive(
            f"{prefix}.output_mask",
            "which outputs a path reaches, rebuilt from the irreps",
        )
        self.op(
            name,
            "linear",
            {"weight": weight, "bias": bias},
            descriptor={
                "irreps_in": str(module.irreps_in),
                "irreps_out": str(module.irreps_out),
            },
        )

    def unaccounted(self) -> list[str]:
        return sorted(set(self.state) - self.used - set(self.derived))


def layer_norm_mlp(walker: Walk, name: str, module, prefix: str) -> None:
    """A linear, layer norm and SiLU stack, its parameters under their own
    names: ``<layer>.weight`` and ``<layer>.bias``, as the module holds them."""
    tensors = {}
    for key, value in module.net.named_parameters():
        walker.use(f"{prefix}.net.{key}")
        tensors[key] = value.detach().cpu().numpy()
    walker.op(name, "layer_norm_mlp", tensors, descriptor={"widths": list(module.hs)})


def walk_nonlinear(walker: Walk, block, prefix: str) -> None:
    """The nonlinear interaction: its linears, its two layer-norm networks and
    the two scalars of its learned normalization.

    The block's own op carries ``alpha`` and ``beta``: the message is divided
    by ``alpha + beta * density``.
    """
    walker.use(f"{prefix}.alpha", f"{prefix}.beta")
    walker.op(
        prefix,
        "interaction",
        {
            name: getattr(block, name).detach().cpu().numpy().reshape(())
            for name in ("alpha", "beta")
        },
        descriptor={
            "class": type(block).__name__,
            "avg_num_neighbors": float(block.avg_num_neighbors),
            "edge_irreps": str(block.edge_irreps),
        },
    )
    linears = {
        "linear_up": "linear_up",
        "linear_res": "linear_res",
        "source": "source_embedding",
        "target": "target_embedding",
        "linear_mid": "linear_1",
        "linear_out": "linear_2",
        "skip": "skip_tp",
    }
    for name, attribute in linears.items():
        walker.linear(
            f"{prefix}.{name}", getattr(block, attribute), f"{prefix}.{attribute}"
        )
    layer_norm_mlp(
        walker, f"{prefix}.radial", block.conv_tp_weights, f"{prefix}.conv_tp_weights"
    )
    layer_norm_mlp(
        walker, f"{prefix}.density", block.density_fn, f"{prefix}.density_fn"
    )
    for key in list(walker.state):
        if key.startswith(f"{prefix}.conv_tp."):
            walker.derive(
                key,
                "the edge tensor product holds no weights of its own; its "
                "coupling constants and masks are rebuilt from the irreps",
            )
        elif key.startswith(f"{prefix}.equivariant_nonlin."):
            # Earlier releases pickled e3nn's own gate, whose elementwise
            # product holds an empty weight and a mask of the paths it keeps.
            if key.endswith(".weight") and walker.state[key].numel():
                raise ExtractionError(
                    f"{key} holds {walker.state[key].numel()} weights, and the "
                    f"gate this converter carries holds none."
                )
            walker.derive(key, "the gate holds no weights; rebuilt from the irreps")


def walk(model, spelling: str) -> Walk:
    """Every op of the model, in canonical form."""
    numpy = importlib.import_module("numpy")
    walker = Walk(model)
    walker.use("atomic_numbers", "r_max", "num_interactions")

    walker.linear(
        "node_embedding", model.node_embedding.linear, "node_embedding.linear"
    )

    radial = model.radial_embedding
    basis = radial.bessel_fn
    if type(basis).__name__ != "BesselBasis":
        raise ExtractionError(
            f"the radial basis is a {type(basis).__name__}, and this converter "
            f"carries the Bessel basis only."
        )
    walker.use(
        "radial_embedding.bessel_fn.bessel_weights",
        "radial_embedding.bessel_fn.prefactor",
    )
    walker.derive(
        "radial_embedding.bessel_fn.r_max", "the cutoff, carried in the configuration"
    )
    walker.op(
        "radial_basis",
        "bessel_basis",
        {
            "weights": basis.bessel_weights.detach().cpu().numpy(),
            "prefactor": basis.prefactor.detach().cpu().numpy(),
        },
        descriptor={
            "r_max": float(basis.r_max),
            "num_basis": int(basis.bessel_weights.numel()),
        },
    )
    walker.use("radial_embedding.cutoff_fn.p")
    walker.derive(
        "radial_embedding.cutoff_fn.r_max", "the cutoff, carried in the configuration"
    )
    walker.op(
        "cutoff",
        "polynomial_cutoff",
        {},
        descriptor={
            "p": int(radial.cutoff_fn.p),
            "r_max": float(radial.cutoff_fn.r_max),
        },
    )
    if hasattr(radial, "distance_transform"):
        transform = type(radial.distance_transform).__name__
        carried = {
            "AgnesiTransform": ("agnesi_transform", ("a", "q", "p", "covalent_radii")),
            "SoftTransform": ("soft_transform", ("alpha", "covalent_radii")),
        }
        if transform not in carried:
            raise ExtractionError(
                f"the radial embedding applies a {transform} distance transform, "
                f"whose parameters this converter does not carry. It carries "
                f"{sorted(carried)}."
            )
        kind, names = carried[transform]
        walker.use(*(f"radial_embedding.distance_transform.{name}" for name in names))
        walker.op(
            "distance_transform",
            kind,
            {
                name: getattr(radial.distance_transform, name)
                .detach()
                .cpu()
                .numpy()
                .astype(numpy.float64)
                for name in names
            },
        )

    if hasattr(model, "pair_repulsion_fn"):
        zbl = model.pair_repulsion_fn
        names = ("c", "covalent_radii", "a_exp", "a_prefactor")
        walker.use(
            *(f"pair_repulsion_fn.{name}" for name in names), "pair_repulsion_fn.p"
        )
        for key in (
            "pair_repulsion_fn.r_max",
            "pair_repulsion_fn.cutoff.p",
            "pair_repulsion_fn.cutoff.r_max",
        ):
            if key in walker.state:
                walker.derive(
                    key,
                    "state of an earlier release of the repulsion, which the "
                    "frozen tree's forward never reads: its envelope ends at the "
                    "pair's covalent radii and has the order p",
                )
        walker.op(
            "pair_repulsion",
            "zbl",
            {name: getattr(zbl, name).detach().cpu().numpy() for name in names},
            descriptor={"p": int(zbl.p)},
        )

    # A dielectric model has no isolated-atom energies: it reads out no energy.
    if hasattr(model, "atomic_energies_fn"):
        walker.use("atomic_energies_fn.atomic_energies")
        energies = model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
        walker.op(
            "energy.atomic_energies",
            "atomic_energies",
            {"values": numpy.atleast_2d(energies).astype(numpy.float64)},
        )
    if spelling in ("scale_shift", "polar", "magnetic", "les"):
        walker.use("scale_shift.scale", "scale_shift.shift")
        walker.op(
            "energy.scale_shift",
            "scale_shift",
            {
                "scale": model.scale_shift.scale.detach().cpu().numpy().reshape(-1),
                "shift": model.scale_shift.shift.detach().cpu().numpy().reshape(-1),
            },
        )

    for index, block in enumerate(model.interactions):
        prefix = f"interactions.{index}"
        kind = type(block).__name__
        if kind not in INTERACTIONS:
            raise ExtractionError(
                f"interaction {index} is a {kind}, whose weights this converter "
                f"does not map. It maps {list(INTERACTIONS)}."
            )
        # A buffer in recent releases and a plain attribute in older ones,
        # which is how every published model was pickled.
        if f"{prefix}.avg_num_neighbors" in walker.state:
            walker.use(f"{prefix}.avg_num_neighbors")
        if kind == "RealAgnosticResidualNonLinearInteractionBlock":
            walk_nonlinear(walker, block, prefix)
            continue
        walker.op(
            prefix,
            "interaction",
            {},
            descriptor={
                "class": kind,
                "avg_num_neighbors": float(block.avg_num_neighbors),
            },
        )
        walker.linear(f"{prefix}.linear_up", block.linear_up, f"{prefix}.linear_up")
        if spelling == "magnetic":
            walk_magnetic_interaction(walker, block, prefix)
            continue
        walk_radial(
            walker,
            f"{prefix}.radial",
            block.conv_tp_weights,
            f"{prefix}.conv_tp_weights",
        )
        density = getattr(block, "density_fn", None)
        if density is not None:
            walk_radial(walker, f"{prefix}.density", density, f"{prefix}.density_fn")
        for key in list(walker.state):
            if key.startswith(f"{prefix}.conv_tp."):
                walker.derive(
                    key,
                    "the edge tensor product holds no weights of its own; its "
                    "coupling constants and masks are rebuilt from the irreps",
                )
        walker.linear(f"{prefix}.linear", block.linear, f"{prefix}.linear")
        walker.use(f"{prefix}.skip_tp.weight")
        walker.derive(f"{prefix}.skip_tp.output_mask", "rebuilt from the irreps")
        walker.op(
            f"{prefix}.skip",
            "element_linear",
            {"weight": skip_to_canonical(block.skip_tp)},
            descriptor={
                "irreps_in": str(block.skip_tp.irreps_in1),
                "irreps_out": str(block.skip_tp.irreps_out),
                "num_elements": int(block.skip_tp.irreps_in2.dim),
            },
        )

    use_reduced = bool(getattr(model, "use_reduced_cg", False))
    for index, product in enumerate(model.products):
        prefix = f"products.{index}"
        contractions = product.symmetric_contractions
        irreps_in = channel_irreps(contractions.irreps_in)
        contraction_linear = (
            product.linear_ori if spelling == "magnetic" else product.linear
        )
        targets = [str(irrep) for _, irrep in contraction_linear.irreps_in]
        for position, (target, contraction) in enumerate(
            zip(targets, contractions.contractions, strict=True)
        ):
            source = f"{prefix}.symmetric_contractions.contractions.{position}"
            correlation = contraction.correlation
            by_order = {correlation: contraction.weights_max}
            walker.use(f"{source}.weights_max")
            for offset, weight in enumerate(contraction.weights):
                by_order[correlation - 1 - offset] = weight
                walker.use(f"{source}.weights.{offset}")
            zeroed = {}
            for key in list(walker.state):
                if key.startswith(f"{source}.") and key.endswith("_zeroed"):
                    walker.use(key)
            for order in range(1, correlation + 1):
                slot = "max" if order == correlation else str(correlation - 1 - order)
                flag = walker.state.get(f"{source}.weights_{slot}_zeroed")
                zeroed[str(order)] = bool(flag) if flag is not None else False
            tensors = {}
            rebuilt = []
            dimension = 2 * int(target[:-1]) + 1
            for order in range(1, correlation + 1):
                walker.use(f"{source}.U_matrix_{order}")
                tensors[f"weights.{order}"] = by_order[order].detach().cpu().numpy()
                basis = getattr(contraction, f"U_matrix_{order}")
                exact = float64_basis(contraction, target, order, basis, use_reduced)
                if exact is not None:
                    basis = exact
                    rebuilt.append(str(order))
                tensors[f"basis.{order}"] = basis_path_first(basis, order, dimension)
            walker.op(
                f"{prefix}.contraction.{target}",
                "symmetric_contraction",
                tensors,
                clebsch_gordan_basis="reduced" if use_reduced else "full",
                parametrization="nested",
                descriptor={
                    "irreps_in": irreps_in,
                    "target": target,
                    "correlation": int(correlation),
                    "zeroed": zeroed,
                    "basis_rebuilt_at_float64": rebuilt,
                },
            )
        walker.linear(f"{prefix}.linear", product.linear, f"{prefix}.linear")
        if spelling == "magnetic":
            walker.linear(
                f"{prefix}.linear_ori", product.linear_ori, f"{prefix}.linear_ori"
            )
            walk_radial(
                walker,
                f"{prefix}.moment_weights",
                product.conv_tp_weights,
                f"{prefix}.conv_tp_weights",
            )
            derived_tensor_product(walker, f"{prefix}.conv_tp.")

    if hasattr(model, "joint_embedding"):
        tensors = {}
        for key, value in model.joint_embedding.named_parameters():
            walker.use(f"joint_embedding.{key}")
            tensors[key] = value.detach().cpu().numpy()
        walker.op("graph_features", "graph_feature_embedding", tensors)
    if hasattr(model, "embedding_readout"):
        walker.linear(
            "embedding_readout",
            model.embedding_readout.linear,
            "embedding_readout.linear",
        )

    for index, readout in enumerate(model.readouts):
        prefix = f"readouts.{index}"
        kind = type(readout).__name__
        if kind not in READOUTS:
            raise ExtractionError(
                f"readout {index} is a {kind}, whose weights this converter does "
                f"not map. It maps {list(READOUTS)}."
            )
        if kind in (
            "LinearReadoutBlock",
            "LinearDipolePolarReadoutBlock",
            "LinearDipoleReadoutBlock",
        ):
            walker.linear(prefix, readout.linear, f"{prefix}.linear")
            continue
        if kind in ("NonLinearDipolePolarReadoutBlock", "NonLinearDipoleReadoutBlock"):
            dipole_polar_readout(walker, readout, prefix)
            continue
        walker.linear(f"{prefix}.first", readout.linear_1, f"{prefix}.linear_1")
        if kind == "NonLinearBiasReadoutBlock":
            walker.linear(
                f"{prefix}.middle", readout.linear_mid, f"{prefix}.linear_mid"
            )
        walker.linear(f"{prefix}.second", readout.linear_2, f"{prefix}.linear_2")
        acts = getattr(readout.non_linearity, "acts", [])
        gate = [getattr(getattr(act, "f", act), "__name__", None) for act in acts]
        walker.ops[f"{prefix}.first"]["descriptor"]["gate"] = gate[0] if gate else None
    if spelling == "les":
        walk_les(walker, model)
    if spelling == "polar":
        walk_polar(walker, model)
    if spelling == "dielectric":
        walk_dielectric(walker, model)
    if spelling == "magnetic":
        torch = importlib.import_module("torch")
        for name in ("mag_radial_embedding", "one_body_cheb_basis_with_const"):
            if hasattr(model, name):
                basis = getattr(model, name)
                first = 0 if name == "one_body_cheb_basis_with_const" else 1
                expected = torch.arange(first, first + basis.num_basis).to(basis.n)
                if not torch.equal(basis.n.reshape(-1), expected):
                    raise ExtractionError(
                        f"{name}.n differs from the canonical Chebyshev orders"
                    )
                walker.derive(
                    f"{name}.n",
                    "Chebyshev orders, validated against the configured basis",
                )
        walker.use("m_max")
        walker.op(
            "moments",
            "magnetic_moments",
            {"saturation": model.m_max.detach().cpu().numpy()},
        )
        if model.use_magmom_one_body:
            names = {
                "coefficients": "onebody_magmombasis_coeffs",
                "offset": "one_body_magmom_const_correction",
            }
            walker.use(*names.values())
            walker.op(
                "one_body",
                "one_body_moment",
                {
                    name: getattr(model, source).detach().cpu().numpy()
                    for name, source in names.items()
                },
            )
    return walker


def walk_les(walker: Walk, model) -> None:
    """The LES-specific readouts. Equivariant maps use the usual canonical weights."""
    for source, role in {
        "les_readouts": "outputs.heads.latent_charges.readouts",
        "les_alpha_readouts": "outputs.heads.latent_alphas.readouts",
        "les_kappa_readouts": "outputs.heads.latent_kappas.readouts",
        "les_u_readouts": "multipoles.dipole",
        "les_quad_2e_readouts": "multipoles.quad_sh",
        "les_alpha_2e_readouts": "multipoles.alpha_sh",
        "les_quad_1o_readouts": "multipoles.quad_tensor",
        "les_alpha_1o_readouts": "multipoles.alpha_tensor",
    }.items():
        for index, block in enumerate(getattr(model, source, ())):
            prefix, target = f"{source}.{index}", f"{role}.{index}"
            if role.endswith("_tensor"):
                walker.linear(target + ".linear", block.linear, prefix + ".linear")
                mixer = {}
                for name, part in (
                    ("mlp", "1o"),
                    ("scalar_to_weight_1o", "1o"),
                    ("scalar_to_weight_1e", "1e"),
                ):
                    module = getattr(block, name, None)
                    if module is not None:
                        for key, value in module.named_parameters():
                            walker.use(f"{prefix}.{name}.{key}")
                            mixer[f"{part}.{key}"] = value.detach().numpy()
                walker.op(target, "les_tensor_mixing", mixer)
            elif hasattr(block, "linear"):
                walker.linear(target, block.linear, prefix + ".linear")
            else:
                walker.linear(target + ".first", block.linear_1, prefix + ".linear_1")
                walker.linear(target + ".second", block.linear_2, prefix + ".linear_2")
    for buffer in ("change_of_basis", "change_of_basis_quads"):
        if buffer in walker.state:
            walker.use(buffer)
            walker.op(
                buffer, "les_cartesian_basis", {"basis": walker.state[buffer].numpy()}
            )


def walk_radial(walker: Walk, name: str, module, source: str) -> None:
    """A legacy fully-connected radial network, with its recorded activation."""
    layers = {}
    activation = None
    for index in range(len(module.hs) - 1):
        layer = getattr(module, f"layer{index}")
        walker.use(f"{source}.layer{index}.weight")
        layers[f"layer_{index}"] = layer.weight.detach().cpu().numpy()
        act = getattr(layer, "act", None)
        if act is not None:
            activation = getattr(act, "f", act).__name__
    walker.op(
        name,
        "radial_mlp",
        layers,
        descriptor={"widths": list(module.hs), "activation": activation},
    )


def derived_tensor_product(walker: Walk, prefix: str) -> None:
    """Only weightless coupling buffers may be reconstructed from the irreps."""
    for key in walker.state:
        if key.startswith(prefix):
            relative = key.removeprefix(prefix)
            if relative not in {"weight", "output_mask"} and not relative.startswith(
                "_compiled_main_left_right._w3j_"
            ):
                raise ExtractionError(f"unaccounted tensor-product buffer {key}")
            if key.endswith(".weight") and walker.state[key].numel():
                raise ExtractionError(
                    f"{key} has weights in a supposedly weightless tensor product"
                )
            walker.derive(
                key, "weightless tensor product; coefficients rebuilt from the irreps"
            )


def walk_magnetic_interaction(walker: Walk, block, prefix: str) -> None:
    """The edge and moment networks and the element-conditioned skip."""
    for name, source in (
        ("edge_weights", "conv_tp_weights"),
        ("moment_weights", "conv_tp_weights_magmom"),
        ("density", "density_fn"),
    ):
        walk_radial(
            walker, f"{prefix}.{name}", getattr(block, source), f"{prefix}.{source}"
        )
    walker.linear(f"{prefix}.linear", block.magmom_linear, f"{prefix}.magmom_linear")
    source = "skip_tp" if hasattr(block, "skip_tp") else "magmom_skip_tp"
    skip = getattr(block, source)
    walker.use(f"{prefix}.{source}.weight")
    walker.derive(f"{prefix}.{source}.output_mask", "rebuilt from the irreps")
    walker.op(
        f"{prefix}.skip",
        "element_linear",
        {"weight": skip_to_canonical(skip)},
        descriptor={
            "irreps_in": str(skip.irreps_in1),
            "irreps_out": str(skip.irreps_out),
            "num_elements": int(skip.irreps_in2.dim),
        },
    )
    for name in ("conv_tp", "magmom_conv_tp"):
        derived_tensor_product(walker, f"{prefix}.{name}.")


def dipole_polar_readout(walker: Walk, readout, prefix: str) -> None:
    """The dielectric model's gated readout: its two linears, and the gate's
    sections, scalars, gates and gated, as the descriptor of the first."""
    walker.linear(f"{prefix}.first", readout.linear_1, f"{prefix}.linear_1")
    walker.linear(f"{prefix}.second", readout.linear_2, f"{prefix}.linear_2")
    gate = readout.equivariant_nonlin
    walker.ops[f"{prefix}.first"]["descriptor"]["gate_sections"] = {
        "scalars": str(gate.irreps_scalars),
        "gates": str(gate.irreps_gates),
        "gated": str(gate.irreps_gated),
    }
    # Pickled under a second name too, the same gate.
    for key in list(walker.state):
        if key.startswith(
            (f"{prefix}.equivariant_nonlin.", f"{prefix}.non_linearity.")
        ):
            if key.endswith(".weight") and walker.state[key].numel():
                raise ExtractionError(
                    f"{key} holds {walker.state[key].numel()} weights, and the "
                    f"gate this converter carries holds none."
                )
            walker.derive(key, "the gate holds no weights; rebuilt from the irreps")


def walk_dielectric(walker: Walk, model) -> None:
    """The constants beside the dielectric model's readouts.

    The means and deviations are written when the model is built and never
    read by its forward; the change of basis is the fixed map from spherical
    to Cartesian polarizabilities.
    """
    torch = importlib.import_module("torch")
    for name in (
        "dipole_mean",
        "dipole_std",
        "polarizability_mean",
        "polarizability_std",
    ):
        walker.derive(name, "recorded at construction and never read by the forward")
    torch_tools = importlib.import_module("mace.tools.torch_tools")
    # The extractor starts with float32 by default, independently of the
    # checkpoint. Construct at the buffer's precision before comparing;
    # promoting an already rounded float32 matrix cannot recover fp64 values.
    with torch_tools.default_dtype(model.change_of_basis.dtype):
        fixed = torch_tools.get_change_of_basis()
    if not torch.allclose(
        model.change_of_basis.double(), fixed.double(), atol=1e-12, rtol=0
    ):
        raise ExtractionError(
            "the model's change of basis is not the fixed spherical to Cartesian "
            "map, and v1 applies that map."
        )
    walker.derive("change_of_basis", "the fixed spherical to Cartesian map")


def sparse_product(walker: Walk, name: str, product, prefix: str) -> None:
    """A channel-pair product, one block per path with its normalization
    folded in, keyed by where its two inputs start: ``path.<first>.<second>``,
    each ``[multiplicity of the first, multiplicity of the second]``."""
    if not bool((product.output_mask == 1).all()):
        raise ExtractionError(
            f"{prefix} masks some of its outputs, and the product this carries "
            f"reaches every one."
        )
    walker.use(f"{prefix}.weight")
    walker.derive(f"{prefix}.output_mask", "every output is reached")
    flat = product.weight.detach().cpu().double()
    tensors = {}
    for meta in product._path_meta:
        first, second = int(meta[0]), int(meta[2])
        start, stop, weight = int(meta[6]), int(meta[7]), float(meta[8])
        rows, columns = int(meta[10]), int(meta[11])
        block = flat[start:stop].reshape(rows, columns) * weight
        tensors[f"path.{first}.{second}"] = block.numpy()
    walker.op(
        name,
        "sparse_product",
        tensors,
        descriptor={
            "irreps_in1": str(product.irreps_in1),
            "irreps_in2": str(product.irreps_in2),
            "irreps_out": str(product.irreps_out),
        },
    )


def bias_readout(walker: Walk, name: str, readout, prefix: str) -> None:
    """Linear, gate, biased linear, gate, biased linear, as three linears."""
    walker.linear(f"{name}.first", readout.linear_1, f"{prefix}.linear_1")
    walker.linear(f"{name}.middle", readout.linear_mid, f"{prefix}.linear_mid")
    walker.linear(f"{name}.last", readout.linear_2, f"{prefix}.linear_2")
    for key in list(walker.state):
        if key.startswith(f"{prefix}.equivariant_nonlin."):
            if key.endswith(".weight") and walker.state[key].numel():
                raise ExtractionError(
                    f"{key} holds {walker.state[key].numel()} weights, and the "
                    f"gate this converter carries holds none."
                )
            walker.derive(key, "the gate holds no weights; rebuilt from the irreps")


def walk_polar(walker: Walk, model) -> None:
    """The charge-aware half: the source maps, the layer mixer, the Fukui
    readout, every field update and the local electron energy.

    The long-range solver's own buffers are constants of its configuration,
    rebuilt from it, and so is the per-component feature normalization.
    """
    for index, source in enumerate(model.lr_source_maps):
        prefix = f"lr_source_maps.{index}"
        walker.linear(f"source_maps.{index}", source.linear, f"{prefix}.linear")
    for index, linear in enumerate(model.layer_feature_mixer.linears):
        prefix = f"layer_feature_mixer.linears.{index}"
        walker.linear(f"layer_mixer.{index}", linear, prefix)
    bias_readout(walker, "fukui_readout", model.fukui_source_map, "fukui_source_map")
    for index, update in enumerate(model.field_dependent_charges_maps):
        prefix = f"field_dependent_charges_maps.{index}"
        name = f"updates.{index}"
        if float(update.field_norm_factor) != 1.0:
            raise ExtractionError(
                f"{prefix} divides its field by {float(update.field_norm_factor)}, "
                f"and the update this converter carries divides by one."
            )
        walker.derive(f"{prefix}.field_norm_factor", "one, which divides by nothing")
        embedding = update.potential_embedding
        for target, source in (
            ("from_potential", "potential_linear"),
            ("from_features", "node_feats_linear"),
            ("from_density", "charge_embedding"),
        ):
            walker.linear(
                f"{name}.{target}",
                getattr(embedding, source),
                f"{prefix}.potential_embedding.{source}",
            )
        walker.linear(
            f"{name}.element_embedding",
            update.source_embedding,
            f"{prefix}.source_embedding",
        )
        sparse_product(
            walker, f"{name}.products", update.dot_products, f"{prefix}.dot_products"
        )
        layer_norm_mlp(
            walker, f"{name}.mlp", update.nonlinearity, f"{prefix}.nonlinearity"
        )
        sparse_product(walker, f"{name}.modulation", update.tp_out, f"{prefix}.tp_out")
        bias_readout(walker, f"{name}.readout", update.readout, f"{prefix}.readout")
    energy = model.local_electron_energy
    prefix = "local_electron_energy"
    walker.linear(
        "electron_energy.from_density", energy.linear_up_q, f"{prefix}.linear_up_q"
    )
    walker.linear(
        "electron_energy.from_potential", energy.linear_up_v, f"{prefix}.linear_up_v"
    )
    sparse_product(
        walker,
        "electron_energy.density_products",
        energy.dot_products_q,
        f"{prefix}.dot_products_q",
    )
    sparse_product(
        walker,
        "electron_energy.potential_products",
        energy.dot_products_v,
        f"{prefix}.dot_products_v",
    )
    layer_norm_mlp(walker, "electron_energy.mlp", energy.mlp, f"{prefix}.mlp")
    for key in list(walker.state):
        if key.startswith(
            (
                "electric_potential_descriptor.",
                "coulomb_energy.",
                "external_field_contribution.",
            )
        ):
            walker.derive(
                key,
                "a constant of the long-range solver, rebuilt from its configuration",
            )
    walker.derive("kspace_cutoff", "carried in the configuration as kspace_cutoff")
    walker.derive(
        "field_feature_norms",
        "the per-order normalization of the configuration, repeated per component",
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def extract(source: Path, output: Path) -> Path:
    """Read ``source``, write ``output.safetensors`` and ``output.json``.

    Returns:
        The sidecar's path.
    """
    model = load(source)
    source_class = type(model).__name__
    spelling = classify(model)
    fixed_point = None
    if source_class == "MagneticSCFMACE":
        wrapped = model.magmom_mace
        if (
            classify(wrapped) != "magnetic"
            or type(wrapped).__name__ != "MagneticScaleShiftMACE"
        ):
            raise ExtractionError("MagneticSCFMACE must wrap a MagneticScaleShiftMACE")
        unknown = set(model.state_dict()) - {
            f"magmom_mace.{key}" for key in wrapped.state_dict()
        }
        if unknown:
            raise ExtractionError(f"unaccounted wrapper tensors: {sorted(unknown)}")
        if model.cache_magmom is not None:
            raise ExtractionError(
                "clear MagneticSCFMACE.cache_magmom before conversion: runtime "
                "warm-start state is not model configuration"
            )
        fixed_point = {
            name: getattr(model, name)
            for name in (
                "n_scf_step",
                "scf_tol",
                "scf_step_size",
                "scf_logging",
                "use_scf",
                "use_collinear",
            )
        }
        model = wrapped
    config = legacy_config(model)
    if fixed_point is not None:
        config["fixed_point"] = fixed_point
    headless = not hasattr(model, "heads")
    heads = ["default"] if headless else [str(head) for head in model.heads]
    walker = walk(model, spelling)
    unaccounted = walker.unaccounted()
    if unaccounted:
        raise ExtractionError(
            f"the checkpoint holds tensors this converter neither carries nor "
            f"knows to be derived, and dropping them would change the model: "
            f"{unaccounted}."
        )
    dtypes = {str(value.dtype).removeprefix("torch.") for value in model.parameters()}
    if len(dtypes) != 1 or not dtypes <= {"float32", "float64"}:
        raise ExtractionError(
            f"the weights are in {sorted(dtypes)}, and one float type is expected."
        )
    mace = importlib.import_module("mace")
    sidecar = {
        "format": FORMAT,
        "version": VERSION,
        "provenance": {
            "source_file": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_class": source_class,
            "source_version": str(getattr(mace, "__version__", "unknown")),
            "converter_version": CONVERTER_VERSION,
            "headless": headless,
        },
        "family": spelling,
        "config": config,
        "heads": heads,
        "dtype": dtypes.pop(),
        "ops": walker.ops,
        "derived": walker.derived,
    }
    save_file = importlib.import_module("safetensors.numpy").save_file
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(walker.tensors, str(output.with_suffix(".safetensors")))
    document = output.with_suffix(".json")
    document.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the pickled legacy checkpoint")
    parser.add_argument(
        "output",
        type=Path,
        help="where to write, without a suffix; two files are written",
    )
    arguments = parser.parse_args(argv)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        written = extract(arguments.source, arguments.output)
    except ExtractionError as error:
        print(f"extraction refused: {error}", file=sys.stderr)
        return 2
    print(written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
