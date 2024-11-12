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
        for j in range(4):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        )
                        .weight[:num_radial, :]
                        .clone()
                    )
                )
            else:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        ).weight.clone()
                    )
                )

        model.interactions[i].linear.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear.weight.clone()
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
        if model.interactions[i].__class__.__name__ in [
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ]:
            # Assuming only 1 layer in density_fn
            getattr(model.interactions[i].density_fn, "layer0").weight = (
                torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].density_fn,
                        "layer0",
                    ).weight.clone()
                )
            )
    # Transferring products
    for i in range(2):  # Assuming 2 products modules
        max_range = max_L + 1 if i == 0 else 1
        for j in range(max_range):  # Assuming 3 contractions in symmetric_contractions
            model.products[i].symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights, :, :]
                    .clone()
                )
            )

            for k in range(2):  # Assuming 2 weights in each contraction
                model.products[i].symmetric_contractions.contractions[j].weights[k] = (
                    torch.nn.Parameter(
                        model_foundations.products[i]
                        .symmetric_contractions.contractions[j]
                        .weights[k][indices_weights, :, :]
                        .clone()
                    )
                )

        model.products[i].linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

    if load_readout:
        # Transferring readouts
        model_readouts_zero_linear_weight = model.readouts[0].linear.weight.clone()
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

        shape_input_1 = (
            model_foundations.readouts[1].linear_1.__dict__["irreps_out"].num_irreps
        )
        shape_output_1 = model.readouts[1].linear_1.__dict__["irreps_out"].num_irreps
        model_readouts_one_linear_1_weight = model.readouts[1].linear_1.weight.clone()
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
        model_readouts_one_linear_2_weight = model.readouts[1].linear_2.weight.clone()
        model_readouts_one_linear_2_weight = model_foundations.readouts[
            1
        ].linear_2.weight.view(shape_input_1, -1).repeat(
            len(model_heads), len(model_heads)
        ).flatten().clone() / (
            ((shape_input_1) / (shape_output_1)) ** 0.5
        )
        model.readouts[1].linear_2.weight = torch.nn.Parameter(
            model_readouts_one_linear_2_weight
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
