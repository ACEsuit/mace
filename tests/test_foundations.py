from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional
from ase.build import molecule
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.calculators import mace_mp, mace_off, mace_omol
from mace.calculators.foundations_models import mace_polar, polar_model_paths
from mace.tools import torch_geometric
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.scripts_utils import extract_config_mace_model, remove_pt_head
from mace.tools.utils import AtomicNumberTable

try:
    import graph_longrange  # noqa: F401

    GRAPH_LONGRANGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRAPH_LONGRANGE_AVAILABLE = False

MODEL_PATH = (
    Path(__file__).parent.parent
    / "mace"
    / "calculators"
    / "foundations_models"
    / "2023-12-03-mace-mp.model"
)

torch.set_default_dtype(torch.float64)


# @pytest.skip("Problem with the float type", allow_module_level=True)
def test_foundations():
    # Create MACE model
    config = data.Configuration(
        atomic_numbers=molecule("H2COH").numbers,
        positions=molecule("H2COH").positions,
        properties={
            "forces": molecule("H2COH").positions,
            "energy": -1.5,
            "charges": molecule("H2COH").numbers,
            "dipole": np.array([-1.5, 1.5, 2.0]),
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
            "charges": 1.0,
            "dipole": 1.0,
        },
    )

    # Created the rotated environment
    rot = R.from_euler("z", 60, degrees=True).as_matrix()
    positions_rotated = np.array(rot @ config.positions.T).T
    config_rotated = data.Configuration(
        atomic_numbers=molecule("H2COH").numbers,
        positions=positions_rotated,
        properties={
            "forces": molecule("H2COH").positions,
            "energy": -1.5,
            "charges": molecule("H2COH").numbers,
            "dipole": np.array([-1.5, 1.5, 2.0]),
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
            "charges": 1.0,
            "dipole": 1.0,
        },
    )
    table = tools.AtomicNumberTable([1, 6, 8])
    atomic_energies = np.array([0.0, 0.0, 0.0], dtype=float)
    model_config = dict(
        r_max=6,
        num_bessel=10,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=3,
        hidden_irreps=o3.Irreps("128x0e + 128x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=0.1,
        atomic_inter_shift=0.0,
        use_reduced_cg=False,
    )
    model = modules.ScaleShiftMACE(**model_config)
    calc_foundation = mace_mp(model="medium", device="cpu", default_dtype="float64")
    model_loaded = load_foundations_elements(
        model,
        calc_foundation.models[0],
        table=table,
        load_readout=True,
        use_shift=False,
        max_L=1,
    )
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=6.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=6.0
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    forces_loaded = model_loaded(batch.to_dict())["forces"]
    forces = model(batch.to_dict())["forces"]
    assert torch.allclose(forces, forces_loaded)


def test_multi_reference():
    config_multi = data.Configuration(
        atomic_numbers=molecule("H2COH").numbers,
        positions=molecule("H2COH").positions,
        properties={
            "forces": molecule("H2COH").positions,
            "energy": -1.5,
            "charges": molecule("H2COH").numbers,
            "dipole": np.array([-1.5, 1.5, 2.0]),
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
            "charges": 1.0,
            "dipole": 1.0,
        },
        head="MP2",
    )
    table_multi = tools.AtomicNumberTable([1, 6, 8])
    atomic_energies_multi = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    table = tools.AtomicNumberTable([1, 6, 8])

    # Create MACE model
    model_config = dict(
        r_max=6,
        num_bessel=10,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=3,
        hidden_irreps=o3.Irreps("128x0e + 128x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies_multi,
        avg_num_neighbors=61,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=[1.0, 1.0],
        atomic_inter_shift=[0.0, 0.0],
        heads=["MP2", "DFT"],
        use_reduced_cg=False,
    )
    model = modules.ScaleShiftMACE(**model_config)
    calc_foundation = mace_mp(model="medium", device="cpu", default_dtype="float64")
    model_loaded = load_foundations_elements(
        model,
        calc_foundation.models[0],
        table=table,
        load_readout=True,
        use_shift=False,
        max_L=1,
    )
    atomic_data = data.AtomicData.from_config(
        config_multi, z_table=table_multi, cutoff=6.0, heads=["MP2", "DFT"]
    )
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    forces_loaded = model_loaded(batch.to_dict())["forces"]
    calc_foundation = mace_mp(model="medium", device="cpu", default_dtype="float64")
    atoms = molecule("H2COH")
    atoms.info["head"] = "MP2"
    atoms.calc = calc_foundation
    forces = atoms.get_forces()
    assert np.allclose(
        forces, forces_loaded.detach().numpy()[:5, :], atol=1e-5, rtol=1e-5
    )


def test_mace_omol_elements_subset_reproduces_energy_forces():
    """
    Extract MACE-OMOL config, create a same-config model with fewer elements,
    load weights with load_foundations_elements, and verify it reproduces
    energies and forces for a simple molecule.
    """

    calc_foundation = mace_omol(device="cpu", default_dtype="float64")

    foundation_model = calc_foundation.models[0]

    # Build a small test molecule using a subset of elements (H, C, O)
    atoms = molecule("H2O")
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05
    atoms.calc = calc_foundation
    energy_ref = atoms.get_potential_energy()
    forces_ref = atoms.get_forces()

    # Extract foundation config and adapt to a smaller element set
    full_cfg = extract_config_mace_model(foundation_model)
    subset_table = AtomicNumberTable([1, 6, 8])

    # Subset atomic energies to match our reduced element set and preserve heads
    # atomic_energies is [n_heads, n_elements] or [n_elements] -> make 2D, then subset columns
    ae_full = foundation_model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    ae_full_2d = ae_full if ae_full.ndim == 2 else ae_full[None, :]
    z_table_full = AtomicNumberTable([int(z) for z in foundation_model.atomic_numbers])
    col_idx = [z_table_full.z_to_index(z) for z in subset_table.zs]
    ae_subset = ae_full_2d[:, col_idx]

    # Prepare model config for the reduced element set
    model_cfg = dict(full_cfg)
    model_cfg.update(
        {
            "num_elements": len(subset_table),
            "atomic_numbers": subset_table.zs,
            "atomic_energies": ae_subset,
        }
    )

    # Create target model and load the subset of foundation weights
    model_subset = modules.ScaleShiftMACE(**model_cfg)
    model_loaded = load_foundations_elements(
        model_subset,
        foundation_model,
        table=subset_table,
        load_readout=True,
        use_shift=True,
        use_scale=True,
        max_L=2,
    )

    # Build a single-config batch for the same atoms using the reduced table
    config = data.Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        properties={
            "forces": np.zeros_like(atoms.positions),
            "energy": 0.0,
        },
        property_weights={"forces": 1.0, "energy": 1.0, "spin": 1.0, "charges": 0.0},
        head="omol",
    )
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=subset_table,
        cutoff=float(foundation_model.r_max),
        heads=["omol"],
    )
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(data_loader))
    outputs = model_loaded(batch.to_dict())

    # Compare energies (graph-level) and forces (node-level)
    energy_loaded = outputs["energy"].detach().cpu().numpy().item()
    forces_loaded = outputs["forces"].detach().cpu().numpy()

    assert np.allclose(energy_ref, energy_loaded, atol=1e-5, rtol=1e-5)
    assert np.allclose(forces_ref, forces_loaded, atol=1e-5, rtol=1e-5)


def test_mace_mh_1_elements_subset_reproduces_energy_forces():
    """
    Extract MACE-MH-1 config, create a same-config model with fewer elements,
    load weights with load_foundations_elements, and verify it reproduces
    energies and forces for a simple molecule.
    """

    calc_foundation = mace_mp(
        model="mh-1", device="cpu", default_dtype="float64", head="omat_pbe"
    )

    foundation_model = remove_pt_head(
        calc_foundation.models[0], head_to_keep="omat_pbe"
    )
    print("foundation_model", foundation_model)

    # Build a small test molecule using a subset of elements (H, C, O)
    atoms = molecule("H2O")
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05
    atoms.calc = calc_foundation
    energy_ref = atoms.get_potential_energy()
    forces_ref = atoms.get_forces()

    # Extract foundation config and adapt to a smaller element set
    full_cfg = extract_config_mace_model(foundation_model)
    subset_table = AtomicNumberTable([1, 6, 8])

    # Subset atomic energies to match our reduced element set and preserve heads
    # atomic_energies is [n_heads, n_elements] or [n_elements] -> make 2D, then subset columns
    ae_full = foundation_model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    ae_full_2d = ae_full if ae_full.ndim == 2 else ae_full[None, :]
    z_table_full = AtomicNumberTable([int(z) for z in foundation_model.atomic_numbers])
    col_idx = [z_table_full.z_to_index(z) for z in subset_table.zs]
    ae_subset = ae_full_2d[:, col_idx]

    # Prepare model config for the reduced element set
    model_cfg = dict(full_cfg)
    model_cfg.update(
        {
            "num_elements": len(subset_table),
            "atomic_numbers": subset_table.zs,
            "atomic_energies": ae_subset,
        }
    )

    # Create target model and load the subset of foundation weights
    model_subset = modules.ScaleShiftMACE(**model_cfg)
    model_loaded = load_foundations_elements(
        model_subset,
        foundation_model,
        table=subset_table,
        load_readout=True,
        use_shift=True,
        use_scale=True,
        max_L=1,
    )

    # Build a single-config batch for the same atoms using the reduced table
    config = data.Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        properties={
            "forces": np.zeros_like(atoms.positions),
            "energy": 0.0,
        },
        property_weights={"forces": 1.0, "energy": 1.0, "spin": 1.0, "charges": 0.0},
        head="omat_pbe",
    )
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=subset_table,
        cutoff=float(foundation_model.r_max),
        heads=["omat_pbe"],
    )
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(data_loader))
    outputs = model_loaded(batch.to_dict())

    # Compare energies (graph-level) and forces (node-level)
    energy_loaded = outputs["energy"].detach().cpu().numpy().item()
    forces_loaded = outputs["forces"].detach().cpu().numpy()

    assert np.allclose(energy_ref, energy_loaded, atol=1e-5, rtol=1e-5)
    assert np.allclose(forces_ref, forces_loaded, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "calc",
    [
        mace_mp(device="cpu", default_dtype="float64"),
        mace_mp(model="small", device="cpu", default_dtype="float64"),
        mace_mp(model="medium", device="cpu", default_dtype="float64"),
        mace_mp(model="large", device="cpu", default_dtype="float64"),
        mace_mp(model=MODEL_PATH, device="cpu", default_dtype="float64"),
        mace_off(model="small", device="cpu", default_dtype="float64"),
        mace_off(model="medium", device="cpu", default_dtype="float64"),
        mace_off(model="large", device="cpu", default_dtype="float64"),
        mace_off(model=MODEL_PATH, device="cpu", default_dtype="float64"),
    ],
)
def test_compile_foundation(calc):
    model = calc.models[0]
    atoms = molecule("CH4")
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
    batch = calc._atoms_to_batch(atoms)  # pylint: disable=protected-access
    output_1 = model(batch.to_dict())
    model_compiled = jit.compile(model)
    output = model_compiled(batch.to_dict())
    for key in output_1.keys():
        if isinstance(output_1[key], torch.Tensor):
            assert torch.allclose(output_1[key], output[key], atol=1e-5)


@pytest.mark.parametrize(
    "model",
    [
        mace_mp(model="small", device="cpu", default_dtype="float64").models[0],
        mace_mp(model="medium", device="cpu", default_dtype="float64").models[0],
        mace_mp(model="large", device="cpu", default_dtype="float64").models[0],
        mace_mp(model=MODEL_PATH, device="cpu", default_dtype="float64").models[0],
        mace_off(model="small", device="cpu", default_dtype="float64").models[0],
        mace_off(model="medium", device="cpu", default_dtype="float64").models[0],
        mace_off(model="large", device="cpu", default_dtype="float64").models[0],
        mace_off(model=MODEL_PATH, device="cpu", default_dtype="float64").models[0],
    ],
)
def test_extract_config(model):
    assert isinstance(model, modules.ScaleShiftMACE)
    config = data.Configuration(
        atomic_numbers=molecule("H2COH").numbers,
        positions=molecule("H2COH").positions,
        properties={
            "forces": molecule("H2COH").positions,
            "energy": -1.5,
            "charges": molecule("H2COH").numbers,
            "dipole": np.array([-1.5, 1.5, 2.0]),
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
            "charges": 1.0,
            "dipole": 1.0,
        },
    )
    model_copy = modules.ScaleShiftMACE(**extract_config_mace_model(model))
    model_copy.load_state_dict(model.state_dict(), strict=False)
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=6.0)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    output = model(batch.to_dict())
    output_copy = model_copy(batch.to_dict())
    # assert all items of the output dicts are equal
    for key in output.keys():
        if isinstance(output[key], torch.Tensor):
            assert torch.allclose(output[key], output_copy[key], atol=1e-5)


def test_remove_pt_head():
    # Set up test data
    torch.manual_seed(42)
    atomic_energies_pt_head = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    z_table = AtomicNumberTable([1, 8])  # H and O

    # Create multihead model
    model_config = {
        "r_max": 5.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": 2,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 2,
        "num_elements": len(z_table),
        "hidden_irreps": o3.Irreps("32x0e + 32x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": atomic_energies_pt_head,
        "avg_num_neighbors": 8,
        "atomic_numbers": z_table.zs,
        "correlation": 3,
        "heads": ["pt_head", "DFT"],
        "atomic_inter_scale": [1.0, 1.0],
        "atomic_inter_shift": [0.0, 0.1],
    }

    model = modules.ScaleShiftMACE(**model_config)

    # Create test molecule
    mol = molecule("H2O")
    config_pt_head = data.Configuration(
        atomic_numbers=mol.numbers,
        positions=mol.positions,
        properties={"energy": 1.0, "forces": np.random.randn(len(mol), 3)},
        property_weights={"forces": 1.0, "energy": 1.0},
        head="DFT",
    )
    atomic_data = data.AtomicData.from_config(
        config_pt_head, z_table=z_table, cutoff=5.0, heads=["pt_head", "DFT"]
    )
    dataloader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False
    )
    batch = next(iter(dataloader))
    # Test original mode
    output_orig = model(batch.to_dict())

    # Convert to single head model
    new_model = remove_pt_head(model, head_to_keep="DFT")

    # Basic structure tests
    assert len(new_model.heads) == 1
    assert new_model.heads[0] == "DFT"
    assert new_model.atomic_energies_fn.atomic_energies.shape[0] == 1
    assert len(torch.atleast_1d(new_model.scale_shift.scale)) == 1
    assert len(torch.atleast_1d(new_model.scale_shift.shift)) == 1

    # Test output consistency
    atomic_data = data.AtomicData.from_config(
        config_pt_head, z_table=z_table, cutoff=5.0, heads=["DFT"]
    )
    dataloader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False
    )
    batch = next(iter(dataloader))
    output_new = new_model(batch.to_dict())
    torch.testing.assert_close(
        output_orig["energy"], output_new["energy"], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        output_orig["forces"], output_new["forces"], rtol=1e-5, atol=1e-5
    )


def test_remove_pt_head_multihead():
    # Set up test data
    torch.manual_seed(42)
    atomic_energies_pt_head = np.array(
        [
            [1.0, 2.0],  # H energies for each head
            [3.0, 4.0],  # O energies for each head
        ]
        * 2
    )
    print("atomic_energies_pt_head", atomic_energies_pt_head.shape)
    z_table = AtomicNumberTable([1, 8])  # H and O

    # Create multihead model
    model_config = {
        "r_max": 5.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": 2,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 2,
        "num_elements": len(z_table),
        "hidden_irreps": o3.Irreps("32x0e + 32x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": atomic_energies_pt_head,
        "avg_num_neighbors": 8,
        "atomic_numbers": z_table.zs,
        "correlation": 3,
        "heads": ["pt_head", "DFT", "MP2", "CCSD"],
        "atomic_inter_scale": [1.0, 1.0, 1.0, 1.0],
        "atomic_inter_shift": [0.0, 0.1, 0.2, 0.3],
    }

    model = modules.ScaleShiftMACE(**model_config)

    # Create test configurations for each head
    mol = molecule("H2O")
    configs = {}
    atomic_datas = {}
    dataloaders = {}
    original_outputs = {}

    # First get outputs from original model for each head
    for head in model.heads:
        config_pt_head = data.Configuration(
            atomic_numbers=mol.numbers,
            positions=mol.positions,
            properties={"energy": 1.0, "forces": np.random.randn(len(mol), 3)},
            property_weights={"forces": 1.0, "energy": 1.0},
            head=head,
        )
        configs[head] = config_pt_head

        atomic_data = data.AtomicData.from_config(
            config_pt_head, z_table=z_table, cutoff=5.0, heads=model.heads
        )
        atomic_datas[head] = atomic_data

        dataloader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data], batch_size=1, shuffle=False
        )
        dataloaders[head] = dataloader

        batch = next(iter(dataloader))
        output = model(batch.to_dict())
        original_outputs[head] = output

    # Now test each head separately
    for i, head in enumerate(model.heads):
        # Convert to single head model
        new_model = remove_pt_head(model, head_to_keep=head)

        # Basic structure tests
        assert len(new_model.heads) == 1, f"Failed for head {head}"
        assert new_model.heads[0] == head, f"Failed for head {head}"
        assert (
            new_model.atomic_energies_fn.atomic_energies.shape[0] == 1
        ), f"Failed for head {head}"
        assert (
            len(torch.atleast_1d(new_model.scale_shift.scale)) == 1
        ), f"Failed for head {head}"
        assert (
            len(torch.atleast_1d(new_model.scale_shift.shift)) == 1
        ), f"Failed for head {head}"

        # Verify scale and shift values
        assert torch.allclose(
            new_model.scale_shift.scale, model.scale_shift.scale[i : i + 1]
        ), f"Failed for head {head}"
        assert torch.allclose(
            new_model.scale_shift.shift, model.scale_shift.shift[i : i + 1]
        ), f"Failed for head {head}"

        # Test output consistency
        single_head_data = data.AtomicData.from_config(
            configs[head], z_table=z_table, cutoff=5.0, heads=[head]
        )
        single_head_loader = torch_geometric.dataloader.DataLoader(
            dataset=[single_head_data], batch_size=1, shuffle=False
        )
        batch = next(iter(single_head_loader))
        new_output = new_model(batch.to_dict())

        # Compare outputs
        print(
            original_outputs[head]["energy"],
            new_output["energy"],
        )
        torch.testing.assert_close(
            original_outputs[head]["energy"],
            new_output["energy"],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Energy mismatch for head {head}",
        )
        torch.testing.assert_close(
            original_outputs[head]["forces"],
            new_output["forces"],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Forces mismatch for head {head}",
        )

    # Test error cases
    with pytest.raises(ValueError, match="Head non_existent not found in model"):
        remove_pt_head(model, head_to_keep="non_existent")

    # Test default behavior (first non-PT head)
    default_model = remove_pt_head(model)
    assert default_model.heads[0] == "DFT"

    # Additional test: check if each model's computation graph is independent
    models = {head: remove_pt_head(model, head_to_keep=head) for head in model.heads}
    results = {}

    for head, head_model in models.items():
        single_head_data = data.AtomicData.from_config(
            configs[head], z_table=z_table, cutoff=5.0, heads=[head]
        )
        single_head_loader = torch_geometric.dataloader.DataLoader(
            dataset=[single_head_data], batch_size=1, shuffle=False
        )
        batch = next(iter(single_head_loader))
        results[head] = head_model(batch.to_dict())

    # Verify each model produces different outputs
    energies = torch.stack([results[head]["energy"] for head in model.heads])
    assert not torch.allclose(
        energies[0], energies[1], rtol=1e-3
    ), "Different heads should produce different outputs"


def test_remove_pt_head_omol_multihead():
    # Set up test data
    calc = mace_omol(device="cpu", default_dtype="float64")
    model_config = extract_config_mace_model(calc.models[0])
    model_config["heads"] = ["pt_head", "DFT", "MP2", "CCSD"]
    model_config["atomic_inter_scale"] = [1.0] * len(model_config["heads"])
    model_config["atomic_inter_shift"] = [0.0] * len(model_config["heads"])
    # repeat atomic energies for each head from [n_elements] to [n_heads, n_elements]
    model_config["atomic_energies"] = model_config["atomic_energies"][
        None, 0, :
    ].repeat(len(model_config["heads"]), axis=0)
    model = modules.ScaleShiftMACE(**model_config)
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])

    # Create test configurations for each head
    mol = molecule("H2O")
    configs = {}
    atomic_datas = {}
    dataloaders = {}
    original_outputs = {}

    # First get outputs from original model for each head
    for head in model.heads:
        config_pt_head = data.Configuration(
            atomic_numbers=mol.numbers,
            positions=mol.positions,
            properties={"energy": 1.0, "forces": np.random.randn(len(mol), 3)},
            property_weights={"forces": 1.0, "energy": 1.0},
            head=head,
        )
        configs[head] = config_pt_head

        atomic_data = data.AtomicData.from_config(
            config_pt_head, z_table=z_table, cutoff=5.0, heads=model.heads
        )
        atomic_datas[head] = atomic_data

        dataloader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data], batch_size=1, shuffle=False
        )
        dataloaders[head] = dataloader

        batch = next(iter(dataloader))
        output = model(batch.to_dict())
        original_outputs[head] = output

    # Now test each head separately
    for i, head in enumerate(model.heads):
        # Convert to single head model
        new_model = remove_pt_head(model, head_to_keep=head)

        # Basic structure tests
        assert len(new_model.heads) == 1, f"Failed for head {head}"
        assert new_model.heads[0] == head, f"Failed for head {head}"
        assert (
            new_model.atomic_energies_fn.atomic_energies.shape[0] == 1
        ), f"Failed for head {head}"
        assert (
            len(torch.atleast_1d(new_model.scale_shift.scale)) == 1
        ), f"Failed for head {head}"
        assert (
            len(torch.atleast_1d(new_model.scale_shift.shift)) == 1
        ), f"Failed for head {head}"

        # Verify scale and shift values
        assert torch.allclose(
            new_model.scale_shift.scale, model.scale_shift.scale[i : i + 1]
        ), f"Failed for head {head}"
        assert torch.allclose(
            new_model.scale_shift.shift, model.scale_shift.shift[i : i + 1]
        ), f"Failed for head {head}"

        # Test output consistency
        single_head_data = data.AtomicData.from_config(
            configs[head], z_table=z_table, cutoff=5.0, heads=[head]
        )
        single_head_loader = torch_geometric.dataloader.DataLoader(
            dataset=[single_head_data], batch_size=1, shuffle=False
        )
        batch = next(iter(single_head_loader))
        new_output = new_model(batch.to_dict())

        # Compare outputs
        print(
            original_outputs[head]["energy"],
            new_output["energy"],
        )
        torch.testing.assert_close(
            original_outputs[head]["energy"],
            new_output["energy"],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Energy mismatch for head {head}",
        )
        torch.testing.assert_close(
            original_outputs[head]["forces"],
            new_output["forces"],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Forces mismatch for head {head}",
        )

    # Test error cases
    with pytest.raises(ValueError, match="Head non_existent not found in model"):
        remove_pt_head(model, head_to_keep="non_existent")

    # Test default behavior (first non-PT head)
    default_model = remove_pt_head(model)
    assert default_model.heads[0] == "DFT"

    # Additional test: check if each model's computation graph is independent
    models = {head: remove_pt_head(model, head_to_keep=head) for head in model.heads}
    results = {}

    for head, head_model in models.items():
        single_head_data = data.AtomicData.from_config(
            configs[head], z_table=z_table, cutoff=5.0, heads=[head]
        )
        single_head_loader = torch_geometric.dataloader.DataLoader(
            dataset=[single_head_data], batch_size=1, shuffle=False
        )
        batch = next(iter(single_head_loader))
        results[head] = head_model(batch.to_dict())

    # Verify each model produces different outputs
    energies = torch.stack([results[head]["energy"] for head in model.heads])
    assert not torch.allclose(
        energies[0], energies[1], rtol=1e-4
    ), "Different heads should produce different outputs"


def test_load_foundations_elements_omol_multihead_pt_matches():
    """
    Build a 2-head multihead model from the MACE-OMOL foundation config,
    load weights with load_foundations_elements, and verify that the
    pt_head reproduces the energies and forces of the mace_omol model.
    """

    # Foundation (single-head omol) calculator and reference outputs
    calc = mace_omol(device="cpu", default_dtype="float64")
    foundation_model = calc.models[0]

    atoms = molecule("H2O")
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.02
    atoms.calc = calc
    energy_ref = atoms.get_potential_energy()
    forces_ref = atoms.get_forces()

    # Create a 2-head multihead model using the foundation config
    model_config = extract_config_mace_model(foundation_model)
    model_config["heads"] = ["pt_head", "DFT"]  # use 2 heads instead of 4
    model_config["atomic_inter_scale"] = [1.0] * len(model_config["heads"])
    model_config["atomic_inter_shift"] = [0.0] * len(model_config["heads"])
    # repeat atomic energies for each head from [n_elements] to [n_heads, n_elements]
    model_config["atomic_energies"] = model_config["atomic_energies"][
        None, 0, :
    ].repeat(len(model_config["heads"]), axis=0)

    model_mh = modules.ScaleShiftMACE(**model_config)
    z_table = AtomicNumberTable([int(z) for z in model_mh.atomic_numbers])

    # Load foundation weights into the multihead model (multihead finetuning logic)
    model_loaded = load_foundations_elements(
        model_mh,
        foundation_model,
        table=z_table,
        load_readout=True,
        use_shift=True,
        use_scale=True,
        max_L=2,
    )

    # Prepare a batch selecting the pt_head
    config_pt_head = data.Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        properties={"energy": 0.0, "forces": np.zeros_like(atoms.positions)},
        property_weights={"forces": 1.0, "energy": 1.0},
        head="pt_head",
    )
    atomic_data = data.AtomicData.from_config(
        config_pt_head,
        z_table=z_table,
        cutoff=float(foundation_model.r_max),
        heads=model_config["heads"],
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False
    )
    batch = next(iter(loader))
    outputs = model_loaded(batch.to_dict())

    energy_loaded = outputs["energy"].detach().cpu().numpy().item()
    forces_loaded = outputs["forces"].detach().cpu().numpy()

    assert np.allclose(energy_ref, energy_loaded, atol=1e-5, rtol=1e-5)
    assert np.allclose(forces_ref, forces_loaded, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Polar foundation model tests
# ---------------------------------------------------------------------------

_polar_available = (
    GRAPH_LONGRANGE_AVAILABLE
    and polar_model_paths.get("polar-1-m", Path("__missing__")).exists()
    if GRAPH_LONGRANGE_AVAILABLE
    else False
)

_skip_polar = pytest.mark.skipif(
    not _polar_available,
    reason="graph_longrange not installed or polar-1-m model not available",
)


def _water_atoms():
    """A simple water molecule in a periodic box."""
    from ase import Atoms

    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2390, 0.9270, 0.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    return atoms


def _polar_batch(model, atoms, heads=None, head_name="Default"):
    """Build a PolarMACE-compatible batch from atoms, like the calculator does."""
    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    keyspec = data.KeySpecification(
        info_keys={
            "total_spin": "spin",
            "total_charge": "charge",
            "external_field": "external_field",
        },
        arrays_keys={"Qs": "charges"},
    )
    config = data.config_from_atoms(
        atoms, key_specification=keyspec, head_name=head_name
    )
    if heads is None:
        heads = model.heads if hasattr(model, "heads") else ["Default"]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=float(model.r_max),
                heads=heads,
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader)).to("cpu").to_dict()
    model_dtype = next(model.parameters()).dtype
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            batch[key] = value.to(model_dtype)
    return batch


@_skip_polar
def test_polar_extract_config_roundtrip():
    """
    Extract config from a PolarMACE foundation model, rebuild the model,
    load state dict, and verify outputs are identical.
    """
    foundation = mace_polar(model="polar-1-m", device="cpu", return_raw_model=True)
    config = extract_config_mace_model(foundation)
    assert "kspace_cutoff_factor" in config  # PolarMACE-specific field

    model_copy = modules.PolarMACE(**config)
    model_copy.load_state_dict(foundation.state_dict(), strict=True)
    # Ensure consistent dtype (buffers created during __init__ may differ)
    target_dtype = next(foundation.parameters()).dtype
    model_copy.to(target_dtype)

    atoms = _water_atoms()
    batch = _polar_batch(foundation, atoms)
    out_orig = foundation(batch, training=False)
    out_copy = model_copy(batch, training=False)

    for key in ("energy", "forces"):
        assert torch.allclose(
            out_orig[key], out_copy[key], atol=1e-10
        ), f"{key} mismatch after config roundtrip"


@_skip_polar
def test_polar_elements_subset_reproduces_energy_forces():
    """
    Extract PolarMACE config, create a model with fewer elements (H, O),
    load weights with load_foundations_elements, and verify it reproduces
    the foundation model's energies and forces on a water molecule.

    This mirrors the finetuning code path:
      1. extract_config_mace_model()
      2. override num_elements / atomic_numbers / atomic_energies
      3. PolarMACE(**config)
      4. load_foundations_elements()
    """
    foundation = mace_polar(model="polar-1-m", device="cpu", return_raw_model=True)

    # Reference outputs from the full 83-element foundation model
    atoms = _water_atoms()
    calc_ref = mace_polar(model="polar-1-m", device="cpu", default_dtype="float64")
    atoms.calc = calc_ref
    energy_ref = atoms.get_potential_energy()
    forces_ref = atoms.get_forces()

    # Extract config and adapt to a 2-element subset (H=1, O=8)
    full_cfg = extract_config_mace_model(foundation)
    subset_table = AtomicNumberTable([1, 8])
    z_table_full = AtomicNumberTable([int(z) for z in foundation.atomic_numbers])
    col_idx = [z_table_full.z_to_index(z) for z in subset_table.zs]

    ae_full = foundation.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    ae_subset = ae_full[col_idx]

    model_cfg = dict(full_cfg)
    model_cfg.update(
        num_elements=len(subset_table),
        atomic_numbers=subset_table.zs,
        atomic_energies=ae_subset,
    )

    model_subset = modules.PolarMACE(**model_cfg)
    model_loaded = load_foundations_elements(
        model_subset,
        foundation,
        table=subset_table,
        load_readout=True,
        use_shift=True,
        use_scale=True,
        max_L=full_cfg["hidden_irreps"].lmax,
    )

    # Evaluate the element-subset model in float64
    model_loaded = model_loaded.double()
    batch = _polar_batch(model_loaded, atoms)
    out = model_loaded(batch, training=False)

    energy_loaded = out["energy"].detach().cpu().numpy().item()
    forces_loaded = out["forces"].detach().cpu().numpy()

    assert np.allclose(
        energy_ref, energy_loaded, atol=1e-4, rtol=1e-5
    ), f"Energy mismatch: ref={energy_ref}, loaded={energy_loaded}"
    assert np.allclose(
        forces_ref, forces_loaded, atol=1e-4, rtol=1e-5
    ), "Forces mismatch"


@_skip_polar
def test_polar_elements_subset_no_readout_finite():
    """
    Same as above but without loading readouts (the default finetuning case
    for a new single-head model). Verify energies/forces are finite and
    reasonably close to the foundation (within the readout re-init error).
    """
    foundation = mace_polar(model="polar-1-m", device="cpu", return_raw_model=True)

    full_cfg = extract_config_mace_model(foundation)
    subset_table = AtomicNumberTable([1, 8])
    z_table_full = AtomicNumberTable([int(z) for z in foundation.atomic_numbers])
    col_idx = [z_table_full.z_to_index(z) for z in subset_table.zs]

    ae_full = foundation.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    ae_subset = ae_full[col_idx]

    model_cfg = dict(full_cfg)
    model_cfg.update(
        num_elements=len(subset_table),
        atomic_numbers=subset_table.zs,
        atomic_energies=ae_subset,
    )

    model_subset = modules.PolarMACE(**model_cfg)
    model_loaded = load_foundations_elements(
        model_subset,
        foundation,
        table=subset_table,
        load_readout=False,
        max_L=full_cfg["hidden_irreps"].lmax,
    )

    atoms = _water_atoms()
    batch = _polar_batch(model_loaded, atoms)
    out = model_loaded(batch, training=False)

    energy = out["energy"].detach().cpu().item()
    forces = out["forces"].detach().cpu().numpy()

    assert np.isfinite(energy), f"Energy is not finite: {energy}"
    assert np.all(np.isfinite(forces)), "Forces contain non-finite values"
    # Without readouts the energy should still be in the right ballpark
    # (atomic energies dominate), not astronomically large
    assert abs(energy) < 1e6, f"Energy suspiciously large: {energy}"


@_skip_polar
def test_polar_elements_subset_dtype_float64():
    """
    Verify that load_foundations_elements correctly casts a float32 foundation
    model into a float64 finetuning model (the --default_dtype=float64 case).
    """
    foundation = mace_polar(model="polar-1-m", device="cpu", return_raw_model=True)
    assert next(foundation.parameters()).dtype == torch.float32

    full_cfg = extract_config_mace_model(foundation)
    subset_table = AtomicNumberTable([1, 8])
    z_table_full = AtomicNumberTable([int(z) for z in foundation.atomic_numbers])
    col_idx = [z_table_full.z_to_index(z) for z in subset_table.zs]
    ae_subset = (
        foundation.atomic_energies_fn.atomic_energies.detach().cpu().numpy()[col_idx]
    )

    model_cfg = dict(full_cfg)
    model_cfg.update(
        num_elements=len(subset_table),
        atomic_numbers=subset_table.zs,
        atomic_energies=ae_subset,
    )

    # Build the new model in float64
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model_f64 = modules.PolarMACE(**model_cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    assert next(model_f64.parameters()).dtype == torch.float64

    model_loaded = load_foundations_elements(
        model_f64,
        foundation,
        table=subset_table,
        load_readout=False,
        max_L=full_cfg["hidden_irreps"].lmax,
    )

    # Verify all params are float64 after loading
    for name, param in model_loaded.named_parameters():
        assert (
            param.dtype == torch.float64
        ), f"Parameter {name} has dtype {param.dtype}, expected float64"

    atoms = _water_atoms()
    batch = _polar_batch(model_loaded, atoms)
    out = model_loaded(batch, training=False)
    energy = out["energy"].detach().cpu().item()
    forces = out["forces"].detach().cpu().numpy()
    assert np.isfinite(energy), f"Energy is not finite: {energy}"
    assert np.all(np.isfinite(forces)), "Forces contain non-finite values"


@_skip_polar
def test_polar_multihead_finetuning_loads_correctly():
    """
    Simulate multihead finetuning: build a 2-head PolarMACE from the
    foundation config, load weights, and verify the pt_head reproduces
    the foundation model's outputs.
    """
    foundation = mace_polar(model="polar-1-m", device="cpu", return_raw_model=True)

    # Reference from full foundation
    atoms = _water_atoms()
    batch_ref = _polar_batch(foundation, atoms)
    out_ref = foundation(batch_ref, training=False)
    energy_ref = out_ref["energy"].detach().cpu().item()
    forces_ref = out_ref["forces"].detach().cpu().numpy()

    # Build a 2-head model with element subsetting
    full_cfg = extract_config_mace_model(foundation)
    subset_table = AtomicNumberTable([1, 8])
    z_table_full = AtomicNumberTable([int(z) for z in foundation.atomic_numbers])
    col_idx = [z_table_full.z_to_index(z) for z in subset_table.zs]
    ae_full = foundation.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    ae_subset = ae_full[col_idx]

    heads = ["pt_head", "DFT"]
    model_cfg = dict(full_cfg)
    model_cfg.update(
        num_elements=len(subset_table),
        atomic_numbers=subset_table.zs,
        atomic_energies=np.stack([ae_subset, ae_subset]),
        heads=heads,
        atomic_inter_shift=[0.0, 0.0],
        atomic_inter_scale=[1.0, 1.0],
    )

    model_mh = modules.PolarMACE(**model_cfg)
    model_loaded = load_foundations_elements(
        model_mh,
        foundation,
        table=subset_table,
        load_readout=True,
        use_shift=True,
        use_scale=True,
        max_L=full_cfg["hidden_irreps"].lmax,
    )

    # Evaluate the pt_head — should reproduce the foundation
    batch_pt = _polar_batch(model_loaded, atoms, heads=heads, head_name="pt_head")
    out_pt = model_loaded(batch_pt, training=False)
    energy_pt = out_pt["energy"].detach().cpu().item()
    forces_pt = out_pt["forces"].detach().cpu().numpy()

    assert np.isfinite(energy_pt), f"pt_head energy not finite: {energy_pt}"
    assert np.allclose(
        energy_ref, energy_pt, atol=1e-4, rtol=1e-5
    ), f"pt_head energy mismatch: ref={energy_ref}, got={energy_pt}"
    assert np.allclose(
        forces_ref, forces_pt, atol=1e-4, rtol=1e-5
    ), "pt_head forces mismatch"

    # Evaluate the DFT head — should also be finite
    batch_dft = _polar_batch(model_loaded, atoms, heads=heads, head_name="DFT")
    out_dft = model_loaded(batch_dft, training=False)
    assert np.isfinite(out_dft["energy"].detach().cpu().item())
