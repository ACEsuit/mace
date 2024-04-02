from typing import Any, Dict

import torch
from e3nn import o3

from mace.tools.utils import AtomicNumberTable


def extract_config_mace_model(model: torch.nn.Module) -> Dict[str, Any]:
    def radial_to_name(radial_type):
        if radial_type == "BesselBasis":
            return "bessel"
        elif radial_type == "GaussianBasis":
            return "gaussian"
        elif radial_type == "ChebychevBasis":
            return "chebyshev"

    def radial_to_transform(radial):
        if not hasattr(radial, "distance_transform"):
            return None
        elif radial.distance_transform.__class__.__name__ == "AgnesiTransform":
            return "Agnesi"
        elif radial.distance_transform.__class__.__name__ == "SoftTransform":
            return "Soft"

    config = {
        "r_max": model.r_max.item(),
        "num_bessel": len(model.radial_embedding.bessel_fn.bessel_weights),
        "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
        "max_ell": model.spherical_harmonics._lmax,
        "interaction_cls": model.interactions[-1].__class__,
        "interaction_cls_first": model.interactions[0].__class__,
        "num_interactions": model.num_interactions.item(),
        "num_elements": len(model.atomic_numbers),
        "hidden_irreps": o3.Irreps(str(model.products[0].linear.irreps_out)),
        "MLP_irreps": o3.Irreps(str(model.readouts[-1].hidden_irreps)),
        "gate": model.readouts[-1].non_linearity._modules["acts"][0].f,
        "atomic_energies": model.atomic_energies_fn.atomic_energies.cpu().numpy(),
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": model.atomic_numbers,
        "correlation": len(
            model.products[0].symmetric_contractions.contractions[0].weights
        )
        + 1,
        "radial_type": radial_to_name(
            model.radial_embedding.bessel_fn.__class__.__name__
        ),
        "radial_MLP": model.interactions[0].conv_tp_weights.hs[1:-1],
        "pair_repulsion": hasattr(model, "pair_repulsion_fn"),
        "distance_transform": radial_to_transform(model.radial_embedding),
        "atomic_inter_scale": model.scale_shift.scale.item(),
        "atomic_inter_shift": model.scale_shift.shift.item(),
    }
    return config


def extract_load(f: str, map_location: str = "cpu") -> torch.nn.Module:
    model = torch.load(f=f, map_location=map_location)
    model_copy = model.__class__(**extract_config_mace_model(model))
    model_copy.load_state_dict(model.state_dict())
    return model_copy.to(map_location)


def load_foundations(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=False,
    use_scale=True,
    max_L=2,
):
    """
    Load the foundations of a model into a model for fine-tuning.
    """
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    new_z_table = table
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_radial = model.radial_embedding.out_dim
    num_species = len(indices_weights)
    max_ell = model.spherical_harmonics._lmax
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
        if (
            model.interactions[i].__class__.__name__
            == "RealAgnosticResidualInteractionBlock"
        ):
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
        model.readouts[0].linear.weight = torch.nn.Parameter(
            model_foundations.readouts[0].linear.weight.clone()
        )

        model.readouts[1].linear_1.weight = torch.nn.Parameter(
            model_foundations.readouts[1].linear_1.weight.clone()
        )

        model.readouts[1].linear_2.weight = torch.nn.Parameter(
            model_foundations.readouts[1].linear_2.weight.clone()
        )
    if model_foundations.scale_shift is not None:
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.clone()
        if use_shift:
            model.scale_shift.shift = model_foundations.scale_shift.shift.clone()
    return model
