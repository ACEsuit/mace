import torch

from mace.tools.utils import AtomicNumberTable


def load_foundations_elements(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=True,
    use_scale=True,
    max_L=2,
):
    """
    Load the foundations of a model into a model for fine-tuning.
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
    max_ell = model.spherical_harmonics._lmax  # pylint: disable=protected-access
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / (num_species_foundations / num_species) ** 0.5
    )
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
    if model.radial_embedding.bessel_fn.__class__.__name__ == "BesselBasis":
        model.radial_embedding.bessel_fn.bessel_weights = torch.nn.Parameter(
            model_foundations.radial_embedding.bessel_fn.bessel_weights.clone()
        )
    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors

        for (_, param_1), (_, param_2) in zip(
            model.interactions[i].conv_tp_weights.named_parameters(),
            model_foundations.interactions[i].conv_tp_weights.named_parameters(),
        ):
            if param_1.shape == param_2.shape:
                param_1.data.copy_(param_2.data)
            else:
                param_1.data.copy_(param_2.data[: (num_radial + 2 * num_species), ...])
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
        if hasattr(model.interactions[i], "alpha"):
            model.interactions[i].alpha = torch.nn.Parameter(
                model_foundations.interactions[i].alpha.clone()
            )
        if hasattr(model.interactions[i], "beta"):
            model.interactions[i].beta = torch.nn.Parameter(
                model_foundations.interactions[i].beta.clone()
            )
        if model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ]:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    num_species_foundations,
                    num_channels_foundation,
                )[:, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        elif model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualNonLinearInteractionBlock",
        ]:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i].skip_tp.weight
            )
        else:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    (max_ell + 1),
                    num_species_foundations,
                    num_channels_foundation,
                )[:, :, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        if hasattr(model.interactions[i], "density_fn"):
            for (_, param_1), (_, param_2) in zip(
                model.interactions[i].density_fn.named_parameters(),
                model_foundations.interactions[i].density_fn.named_parameters(),
            ):
                param_1.data.copy_(param_2.data)

    # Transferring products
    for i, product in enumerate(model.products):
        indices_weights_prod = indices_weights
        if hasattr(product, "use_agnostic_product"):
            if product.use_agnostic_product:
                indices_weights_prod = [0]
        max_range = max_L + 1 if i < len(model.products) - 1 else 1
        for j in range(max_range):  # Assuming 3 contractions in symmetric_contractions
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

    if load_readout:
        # Transferring readouts
        for i, readout in enumerate(model.readouts):
            if readout.__class__.__name__ == "LinearReadoutBlock":
                model_readouts_zero_linear_weight = readout.linear.weight.clone()
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
                # Determine shapes once to avoid uninitialized use
                if hasattr(readout, "linear_1"):
                    shape_input_1 = (
                        model_foundations.readouts[i]
                        .linear_1.__dict__["irreps_out"]
                        .num_irreps
                    )
                    shape_output_1 = readout.linear_1.__dict__["irreps_out"].num_irreps
                else:
                    raise ValueError("Readout block must have linear_1")
                if hasattr(readout, "linear_1"):
                    model_readouts_one_linear_1_weight = readout.linear_1.weight.clone()
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
                        model_readouts_one_linear_1_bias = readout.linear_1.bias.clone()
                        model_readouts_one_linear_1_bias = (
                            model_foundations.readouts[i]
                            .linear_1.bias.view(-1)
                            .repeat(len(model_heads))
                            .clone()
                        )
                        readout.linear_1.bias = torch.nn.Parameter(
                            model_readouts_one_linear_1_bias
                        )
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
                    # if it has biases transfer them too
                    if readout.linear_mid.bias is not None:
                        readout.linear_mid.bias = torch.nn.Parameter(
                            model_foundations.readouts[i]
                            .linear_mid.bias.repeat(len(model_heads))
                            .clone()
                        )
                if hasattr(readout, "linear_2"):
                    model_readouts_one_linear_2_weight = readout.linear_2.weight.clone()
                    model_readouts_one_linear_2_weight = model_foundations.readouts[
                        i
                    ].linear_2.weight.view(shape_input_1, -1).repeat(
                        len(model_heads), len(model_heads)
                    ).flatten().clone() / (
                        ((shape_input_1) / (shape_output_1)) ** 0.5
                    )
                    readout.linear_2.weight = torch.nn.Parameter(
                        model_readouts_one_linear_2_weight
                    )
                    if readout.linear_2.bias is not None:
                        model_readouts_one_linear_2_bias = readout.linear_2.bias.clone()
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
