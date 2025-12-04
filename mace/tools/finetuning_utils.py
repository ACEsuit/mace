import torch

from mace.tools.utils import AtomicNumberTable


def _copy_if_same_shape(dst_module, src_module, name: str):
    """Copy parameter `name` from src_module to dst_module if shapes match."""
    if not hasattr(dst_module, name) or not hasattr(src_module, name):
        return
    dst_param = getattr(dst_module, name)
    src_param = getattr(src_module, name)
    if dst_param is None or src_param is None:
        return
    if dst_param.shape == src_param.shape:
        setattr(dst_module, name, torch.nn.Parameter(src_param.clone()))


def load_foundations_elements(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout: bool = False,
    use_shift: bool = True,
    use_scale: bool = True,
    max_L: int = 2,
    inherit_field_modules: bool = True,
):
    """
    Load the foundations of a model into a model for fine-tuning, restricting to a
    subset of elements.

    Supports both plain MACE foundations and MACEField foundations. When both
    models are MACEField and `inherit_field_modules=True`, the field-specific
    modules (field_feats, field_linear) are also copied when shape-compatible.

    Parameters
    ----------
    model
        Target model to be fine-tuned (e.g. MACEField with possibly multiple heads).
    model_foundations
        Foundation model (single-head) from which to transfer weights.
    table
        AtomicNumberTable of the target model (defines the element subset & ordering).
    load_readout
        If True, replicate and broadcast foundation readout weights across heads.
    use_shift, use_scale
        Whether to copy the foundation scale/shift into the multihead model.
    max_L
        Maximum L used for the product basis contraction slice logic (for head 0).
    inherit_field_modules
        If True and both models have field-specific modules (MACEField), copy
        those weights when shapes match.
    """
    assert model_foundations.r_max == model.r_max

    # Element tables
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    new_z_table = table

    model_heads = model.heads
    num_species_foundations = len(z_table.zs)
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_species = len(indices_weights)

    # Channels per species in foundation node embedding
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )

    # Radial basis dimension (for slicing conv_tp_weights layer0)
    num_radial = model.radial_embedding.out_dim

    # -------------------------------------------------------------------------
    # Node embedding: slice foundation by species and rescale
    # -------------------------------------------------------------------------
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / (num_species_foundations / num_species) ** 0.5
    )

    # -------------------------------------------------------------------------
    # Radial embedding / Bessel basis
    # -------------------------------------------------------------------------
    if model.radial_embedding.bessel_fn.__class__.__name__ == "BesselBasis":
        model.radial_embedding.bessel_fn.bessel_weights = torch.nn.Parameter(
            model_foundations.radial_embedding.bessel_fn.bessel_weights.clone()
        )

    # -------------------------------------------------------------------------
    # Interaction blocks
    # -------------------------------------------------------------------------
    for i in range(int(model.num_interactions)):
        # linear_up
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )

        # avg_num_neighbors
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors

        # conv_tp_weights: only layer0 depends on radial basis dim
        for j in range(4):  # assuming 4 layers in conv_tp_weights
            layer_name = f"layer{j}"
            src_layer = getattr(
                model_foundations.interactions[i].conv_tp_weights, layer_name
            )
            dst_layer = getattr(model.interactions[i].conv_tp_weights, layer_name)

            if j == 0:
                # layer0 depends on num_radial, so slice in the radial dimension
                dst_layer.weight = torch.nn.Parameter(
                    src_layer.weight[:num_radial, :].clone()
                )
            else:
                dst_layer.weight = torch.nn.Parameter(src_layer.weight.clone())

        # main linear in the interaction block
        model.interactions[i].linear.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear.weight.clone()
        )
        src_skip_w = model_foundations.interactions[i].skip_tp.weight
        dst_skip_w = model.interactions[i].skip_tp.weight
        if src_skip_w.shape == dst_skip_w.shape:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                src_skip_w.clone()
            )

        # Density networks (where present)
        if model.interactions[i].__class__.__name__ in [
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ]:
            getattr(model.interactions[i].density_fn, "layer0").weight = (
                torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].density_fn,
                        "layer0",
                    ).weight.clone()
                )
            )

    # -------------------------------------------------------------------------
    # Product basis blocks
    # -------------------------------------------------------------------------
    # Assumes two "product groups", with different L coverage.
    for i in range(2):  # two product module groups
        max_range = max_L + 1 if i == 0 else 1
        for j in range(max_range):
            # weights_max: slice by species
            model.products[i].symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights, :, :]
                    .clone()
                )
            )

            # weights: two weight tensors per contraction, also sliced by species
            for k in range(2):
                model.products[i].symmetric_contractions.contractions[j].weights[k] = (
                    torch.nn.Parameter(
                        model_foundations.products[i]
                        .symmetric_contractions.contractions[j]
                        .weights[k][indices_weights, :, :]
                        .clone()
                    )
                )

        # product linear
        model.products[i].linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

    # -------------------------------------------------------------------------
    # Readouts
    # -------------------------------------------------------------------------
    if load_readout:
        # We assume the foundation has a 2x0e output and replicate across heads.

        # Readout 0: LinearReadoutBlock for scalar-like quantities
        model_readouts_zero_linear_weight = (
            model_foundations.readouts[0]
            .linear.weight.view(num_channels_foundation, -1)
            .repeat(1, len(model_heads))
            .flatten()
            .clone()
        )
        model.readouts[0].linear.weight = torch.nn.Parameter(
            model_readouts_zero_linear_weight
        )

        # Readout 1: NonLinearReadoutBlock
        shape_input_1 = (
            model_foundations.readouts[1].linear_1.__dict__["irreps_out"].num_irreps
        )
        shape_output_1 = model.readouts[1].linear_1.__dict__["irreps_out"].num_irreps

        # linear_1: replicate across heads (like readout[0])
        model_readouts_one_linear_1_weight = (
            model_foundations.readouts[1]
            .linear_1.weight.view(num_channels_foundation, -1)
            .repeat(1, len(model_heads))
            .flatten()
            .clone()
        )
        model.readouts[1].linear_1.weight = torch.nn.Parameter(
            model_readouts_one_linear_1_weight
        )

        # linear_2: broadcast in both input and output head dimensions
        model_readouts_one_linear_2_weight = model_foundations.readouts[
            1
        ].linear_2.weight.view(shape_input_1, -1).repeat(
            len(model_heads), len(model_heads)
        ).flatten().clone() / (
            (shape_input_1 / shape_output_1) ** 0.5
        )
        model.readouts[1].linear_2.weight = torch.nn.Parameter(
            model_readouts_one_linear_2_weight
        )

    # -------------------------------------------------------------------------
    # Scale & shift
    # -------------------------------------------------------------------------
    if model_foundations.scale_shift is not None:
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.repeat(
                len(model_heads)
            ).clone()
        if use_shift:
            model.scale_shift.shift = model_foundations.scale_shift.shift.repeat(
                len(model_heads)
            ).clone()

    # -------------------------------------------------------------------------
    # MACEField-specific modules (field_feats, field_linear)
    # -------------------------------------------------------------------------
    if inherit_field_modules:
        # Only act if both models have these attributes (i.e. both are MACEField).
        if hasattr(model, "field_feats") and hasattr(model_foundations, "field_feats"):
            n = min(len(model.field_feats), len(model_foundations.field_feats))
            for i in range(n):
                # FullyConnectedTensorProduct typically has a 'weight' parameter
                _copy_if_same_shape(
                    model.field_feats[i], model_foundations.field_feats[i], "weight"
                )

        if hasattr(model, "field_linear") and hasattr(
            model_foundations, "field_linear"
        ):
            n = min(len(model.field_linear), len(model_foundations.field_linear))
            for i in range(n):
                # Copy weight and bias when shape-compatible
                _copy_if_same_shape(
                    model.field_linear[i], model_foundations.field_linear[i], "weight"
                )
                _copy_if_same_shape(
                    model.field_linear[i], model_foundations.field_linear[i], "bias"
                )

    return model


def load_foundations(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
):
    """
    Simple foundation loader: copy matching parameters from model_foundations
    into model, skipping anything in 'readouts'.
    """
    for name, param in model_foundations.named_parameters():
        if name in model.state_dict().keys():
            if "readouts" not in name:
                model.state_dict()[name].copy_(param)
    return model
