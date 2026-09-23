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

CONVERTER_VERSION = "1"
FORMAT = "mace-neutral"
VERSION = 1
SCHEMA_VERSION = "1.0"
SEPARATOR = "::"

#: The classes this carries across, and the model spelling each becomes. Checked
#: by identity against the legacy module, never by name or by the shape of the
#: weights: a subclass can hold tensors that look right and mean more.
SUPPORTED = {
    "MACE": "plain",
    "ScaleShiftMACE": "scale_shift",
    "AtomicDielectricMACE": "dielectric",
}

#: The classes refused by name, and what their conversion is waiting for. A
#: magnetic model's energy blocks look convertible right up to the point where
#: its magnetic-moment basis would be dropped.
REFUSED = {
    "AtomicDipolesMACE": (
        "a dipole-only model, which v1 reads out through observable heads "
        "rather than as a model class of its own"
    ),
    "EnergyDipolesMACE": (
        "an energy and dipole model, which v1 reads out through observable "
        "heads rather than as a model class of its own"
    ),
    "PolarMACE": (
        "an electrostatic model, whose conversion comes with the polar model "
        "on the v1 architecture"
    ),
    "MACELES": (
        "a latent Ewald summation model, which is not on the v1 architecture yet"
    ),
    "MagneticMACE": "a magnetic model, which is not on the v1 architecture yet",
    "MagneticScaleShiftMACE": (
        "a magnetic model, which is not on the v1 architecture yet"
    ),
    "MagneticSCFMACE": "a magnetic model, which is not on the v1 architecture yet",
}

#: Interaction blocks whose weights this knows how to walk.
INTERACTIONS = ("RealAgnosticInteractionBlock", "RealAgnosticResidualInteractionBlock")

#: Readout blocks whose weights this knows how to walk. The dielectric
#: model's dipole and polarizability readouts are not among them yet, so its
#: configuration is read in full and its weights are refused by the readout's
#: name.
READOUTS = ("LinearReadoutBlock", "NonLinearReadoutBlock")


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
    list. It refuses the plain energy class by its name alone, although every
    field it reads exists on it, so that class is read through a view that
    reports the scale-shift class's name and forwards every attribute. Some
    releases of that function read the scale-shift block without checking it
    exists, so the view offers the identity one, which is what the plain class
    computes, and the two keys it produces are dropped afterwards: the plain
    class has no scale and shift to record.
    """
    # Typed loosely on purpose: the plain class is passed through a view that is
    # not a module, and the function reads only attributes.
    extract: Any = importlib.import_module(
        "mace.tools.scripts_utils"
    ).extract_config_mace_model
    target = model
    plain = type(model).__name__ == "MACE"
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
    return {key: jsonable(value) for key, value in config.items()}


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
    for instruction in linear.instructions:
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
    walker.use("radial_embedding.bessel_fn.bessel_weights")
    walker.derive(
        "radial_embedding.bessel_fn.r_max", "the cutoff, carried in the configuration"
    )
    walker.derive(
        "radial_embedding.bessel_fn.prefactor",
        "sqrt(2 / r_max), derived from the cutoff",
    )
    walker.op(
        "radial_basis",
        "bessel_basis",
        {"weights": basis.bessel_weights.detach().cpu().numpy()},
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
        raise ExtractionError(
            f"the radial embedding applies a {transform} distance transform, "
            f"whose parameters this converter does not carry."
        )

    if hasattr(model, "pair_repulsion_fn"):
        zbl = model.pair_repulsion_fn
        names = ("c", "covalent_radii", "a_exp", "a_prefactor")
        walker.use(
            *(f"pair_repulsion_fn.{name}" for name in names), "pair_repulsion_fn.p"
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
    if spelling == "scale_shift":
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
        layers = {}
        widths = list(block.conv_tp_weights.hs)
        activation = None
        for layer in range(len(widths) - 1):
            module = getattr(block.conv_tp_weights, f"layer{layer}")
            walker.use(f"{prefix}.conv_tp_weights.layer{layer}.weight")
            layers[f"layer_{layer}"] = module.weight.detach().cpu().numpy()
            act = getattr(module, "act", None)
            if act is not None:
                activation = getattr(act, "f", act).__name__
        walker.op(
            f"{prefix}.radial",
            "radial_mlp",
            layers,
            descriptor={"widths": widths, "activation": activation},
        )
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
        targets = [str(irrep) for _, irrep in product.linear.irreps_in]
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
            dimension = 2 * int(target[:-1]) + 1
            for order in range(1, correlation + 1):
                walker.use(f"{source}.U_matrix_{order}")
                tensors[f"weights.{order}"] = by_order[order].detach().cpu().numpy()
                tensors[f"basis.{order}"] = basis_path_first(
                    getattr(contraction, f"U_matrix_{order}"), order, dimension
                )
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
                },
            )
        walker.linear(f"{prefix}.linear", product.linear, f"{prefix}.linear")

    for index, readout in enumerate(model.readouts):
        prefix = f"readouts.{index}"
        kind = type(readout).__name__
        if kind not in READOUTS:
            raise ExtractionError(
                f"readout {index} is a {kind}, whose weights this converter does "
                f"not map. It maps {list(READOUTS)}."
            )
        if kind == "LinearReadoutBlock":
            walker.linear(prefix, readout.linear, f"{prefix}.linear")
            continue
        walker.linear(f"{prefix}.first", readout.linear_1, f"{prefix}.linear_1")
        walker.linear(f"{prefix}.second", readout.linear_2, f"{prefix}.linear_2")
        acts = getattr(readout.non_linearity, "acts", [])
        gate = [getattr(act, "f", act).__name__ for act in acts]
        walker.ops[f"{prefix}.first"]["descriptor"]["gate"] = gate[0] if gate else None
    return walker


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def extract(source: Path, output: Path) -> Path:
    """Read ``source``, write ``output.safetensors`` and ``output.json``.

    Returns:
        The sidecar's path.
    """
    model = load(source)
    spelling = classify(model)
    config = legacy_config(model)
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
            "source_class": type(model).__name__,
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
