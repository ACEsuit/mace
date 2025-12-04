import torch

from mace.tools.utils import AtomicNumberTable


def load_foundations_elements(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout: bool = False,
    use_shift: bool = True,
    use_scale: bool = True,
    max_L: int = 2,  # kept for backwards-compatibility (currently unused)
    inherit_field_modules: bool = False,
):
    """
    Load the foundations of a model into a model for fine-tuning.

    Parameters
    ----------
    model
        Target model to be fine-tuned.
    model_foundations
        Pretrained foundation model to load parameters from.
    table
        AtomicNumberTable for the target model.
    load_readout
        If True, also copy the readout layers.
    use_shift
        If True, copy the output shift (E0) from the foundation model.
    use_scale
        If True, copy the output scaling from the foundation model.
    max_L
        Kept for backward compatibility; not used in the new skip_tp logic.
    inherit_field_modules
        If True, also copy field-related modules (e.g. MACEField's
        field_feats and field_linear) when their shapes match.
    """
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    model_heads = model.heads
    new_z_table = table
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_radial = model.radial_embedding.out_dim
    num_species = len(indices_weights)

    # -------------------------------------------------------------------------
    # Node embedding
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
    # Optional auxiliary embeddings
    # -------------------------------------------------------------------------
    if hasattr(model, "joint_embedding"):
        for (_, param_1), (_, param_2) in zip(
            model.joint_embedding.named_parameters(),
            model_foundations.joint_embedding.named_parameters(),
        ):
            param_1.data.copy_(param_2.data)

    if hasattr(model, "embedding_readout"):
        for (_, param_1), (_, param_2) in zip(
            model.embedding_readout.named_parameters(),
            model_foundations.embedding_readout.named_parameters(),
        ):
            param_1.data.copy_(
                param_2.data.reshape(-1, 1)
                .repeat(1, len(model_heads))
                .flatten()
                .clone()
            )

    # -------------------------------------------------------------------------
    # Radial embedding
    # -------------------------------------------------------------------------
    if model.radial_embedding.bessel_fn.__class__.__name__ == "BesselBasis":
        model.radial_embedding.bessel_fn.bessel_weights = torch.nn.Parameter(
            model_foundations.radial_embedding.bessel_fn.bessel_weights.clone()
        )

    # -------------------------------------------------------------------------
    # Interaction blocks
    # -------------------------------------------------------------------------
    for i in range(int(model.num_interactions)):
        # linear_up and avg_num_neighbors
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors

        # conv_tp_weights
        for (_, param_1), (_, param_2) in zip(
            model.interactions[i].conv_tp_weights.named_parameters(),
            model_foundations.interactions[i].conv_tp_weights.named_parameters(),
        ):
            if param_1.shape == param_2.shape:
                param_1.data.copy_(param_2.data)
            else:
                param_1.data.copy_(param_2.data[: (num_radial + 2 * num_species), ...])

        # optional linears
        if hasattr(model.interactions[i], "linear"):
            model.interactions[i].linear.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_1"):
            model.interactions[i].linear_1.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_1.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_2"):
            model.interactions[i].linear_2.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_2.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_res"):
            model.interactions[i].linear_res.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_res.weight.clone()
            )

        # species-dependent embeddings (if present)
        if hasattr(model.interactions[i], "source_embedding"):
            model.interactions[i].source_embedding.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .source_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        if hasattr(model.interactions[i], "target_embedding"):
            model.interactions[i].target_embedding.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .target_embedding.weight.view(num_species_foundations, -1)[
                    indices_weights, :
                ]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )

        # alpha / beta (if present)
        if hasattr(model.interactions[i], "alpha"):
            model.interactions[i].alpha = torch.nn.Parameter(
                model_foundations.interactions[i].alpha.clone()
            )
        if hasattr(model.interactions[i], "beta"):
            model.interactions[i].beta = torch.nn.Parameter(
                model_foundations.interactions[i].beta.clone()
            )
        if hasattr(model.interactions[i], "skip_tp") and hasattr(
            model_foundations.interactions[i], "skip_tp"
        ):
            src_skip = model_foundations.interactions[i].skip_tp.weight
            dst_skip = model.interactions[i].skip_tp.weight
            if src_skip.shape == dst_skip.shape:
                model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                    src_skip.clone()
                )

        # density_fn (if present)
        if hasattr(model.interactions[i], "density_fn"):
            for (_, param_1), (_, param_2) in zip(
                model.interactions[i].density_fn.named_parameters(),
                model_foundations.interactions[i].density_fn.named_parameters(),
            ):
                param_1.data.copy_(param_2.data)

    # -------------------------------------------------------------------------
    # Products
    # -------------------------------------------------------------------------
    for i, product in enumerate(model.products):
        indices_weights_prod = indices_weights
        if hasattr(product, "use_agnostic_product"):
            if product.use_agnostic_product:
                indices_weights_prod = [0]
        max_range = max_L + 1 if i < len(model.products) - 1 else 1
        for j in range(max_range):
            product.symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights_prod, :, :]
                    .clone()
                )
            )

            target_weights = product.symmetric_contractions.contractions[j].weights
            source_weights = (
                model_foundations.products[i]
                .symmetric_contractions.contractions[j]
                .weights
            )
            for k, _ in enumerate(target_weights):
                target_weights[k] = torch.nn.Parameter(
                    source_weights[k][indices_weights_prod, :, :].clone()
                )

        product.linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

    # -------------------------------------------------------------------------
    # Optional readouts
    # -------------------------------------------------------------------------
    if load_readout:
        for i, readout in enumerate(model.readouts):
            if readout.__class__.__name__ == "LinearReadoutBlock":
                model_readouts_zero_linear_weight = (
                    model_foundations.readouts[i]
                    .linear.weight.view(num_channels_foundation, -1)
                    .repeat(1, len(model_heads))
                    .flatten()
                    .clone()
                )
                readout.linear.weight = torch.nn.Parameter(
                    model_readouts_zero_linear_weight
                )

            if readout.__class__.__name__ in [
                "NonLinearBiasReadoutBlock",
                "NonLinearReadoutBlock",
            ]:
                assert hasattr(readout, "linear_1") or hasattr(
                    readout, "linear_mid"
                ), "Readout block must have linear_1 or linear_mid"

                # Determine shapes from foundation
                if hasattr(readout, "linear_1"):
                    shape_input_1 = (
                        model_foundations.readouts[i]
                        .linear_1.__dict__["irreps_out"]
                        .num_irreps
                    )
                    shape_output_1 = readout.linear_1.__dict__["irreps_out"].num_irreps
                else:
                    raise ValueError("Readout block must have linear_1")

                # linear_1
                if hasattr(readout, "linear_1"):
                    model_readouts_one_linear_1_weight = (
                        model_foundations.readouts[i]
                        .linear_1.weight.view(num_channels_foundation, -1)
                        .repeat(1, len(model_heads))
                        .flatten()
                        .clone()
                    )
                    readout.linear_1.weight = torch.nn.Parameter(
                        model_readouts_one_linear_1_weight
                    )

                    if readout.linear_1.bias is not None:
                        model_readouts_one_linear_1_bias = (
                            model_foundations.readouts[i]
                            .linear_1.bias.view(-1)
                            .repeat(len(model_heads))
                            .clone()
                        )
                        readout.linear_1.bias = torch.nn.Parameter(
                            model_readouts_one_linear_1_bias
                        )

                # linear_mid
                if hasattr(readout, "linear_mid"):
                    readout.linear_mid.weight = torch.nn.Parameter(
                        model_foundations.readouts[i]
                        .linear_mid.weight.view(
                            shape_input_1,
                            shape_input_1,
                        )
                        .repeat(len(model_heads), len(model_heads))
                        .flatten()
                        .clone()
                        / ((shape_input_1) / (shape_output_1)) ** 0.5
                    )
                    if readout.linear_mid.bias is not None:
                        readout.linear_mid.bias = torch.nn.Parameter(
                            model_foundations.readouts[i]
                            .linear_mid.bias.repeat(len(model_heads))
                            .clone()
                        )

                # linear_2
                if hasattr(readout, "linear_2"):
                    model_readouts_one_linear_2_weight = (
                        model_foundations.readouts[i]
                        .linear_2.weight.view(shape_input_1, -1)
                        .repeat(len(model_heads), len(model_heads))
                        .flatten()
                        .clone()
                        / ((shape_input_1) / (shape_output_1)) ** 0.5
                    )
                    readout.linear_2.weight = torch.nn.Parameter(
                        model_readouts_one_linear_2_weight
                    )

                    if readout.linear_2.bias is not None:
                        model_readouts_one_linear_2_bias = (
                            model_foundations.readouts[i]
                            .linear_2.bias.view(-1)
                            .repeat(len(model_heads))
                            .flatten()
                            .clone()
                        )
                        readout.linear_2.bias = torch.nn.Parameter(
                            model_readouts_one_linear_2_bias
                        )

    # -------------------------------------------------------------------------
    # Optional MACEField-specific modules (field_feats, field_linear)
    # -------------------------------------------------------------------------
    if inherit_field_modules:
        # field_feats: typically list of modules with a .weight parameter
        if hasattr(model, "field_feats") and hasattr(model_foundations, "field_feats"):
            n = min(len(model.field_feats), len(model_foundations.field_feats))
            for i in range(n):
                src = model_foundations.field_feats[i]
                dst = model.field_feats[i]
                if hasattr(src, "weight") and hasattr(dst, "weight"):
                    if src.weight.shape == dst.weight.shape:
                        dst.weight = torch.nn.Parameter(src.weight.clone())

        # field_linear: typically list of linear maps with weight & bias
        if hasattr(model, "field_linear") and hasattr(
            model_foundations, "field_linear"
        ):
            n = min(len(model.field_linear), len(model_foundations.field_linear))
            for i in range(n):
                src = model_foundations.field_linear[i]
                dst = model.field_linear[i]

                if hasattr(src, "weight") and hasattr(dst, "weight"):
                    if src.weight.shape == dst.weight.shape:
                        dst.weight = torch.nn.Parameter(src.weight.clone())

                if (
                    hasattr(src, "bias")
                    and hasattr(dst, "bias")
                    and src.bias is not None
                    and dst.bias is not None
                    and src.bias.shape == dst.bias.shape
                ):
                    dst.bias = torch.nn.Parameter(src.bias.clone())

    # -------------------------------------------------------------------------
    # Scale / shift
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

    return model


def load_foundations(
    model,
    model_foundations,
):
    for name, param in model_foundations.named_parameters():
        if name in model.state_dict().keys():
            if "readouts" not in name:
                model.state_dict()[name].copy_(param)
    return model
