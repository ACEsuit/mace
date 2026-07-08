import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator

from mace.data.lmdb_dataset import LMDBDataset
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.fairchem_dataset.lmdb_dataset_tools import LMDBDatabase
from mace.tools.run_train_utils import load_dataset_for_path


def test_lmdb_dataset():
    """Test the LMDBDataset by creating a fake database and verifying batch creation."""
    # Set default dtype to match typical MACE usage
    torch.set_default_dtype(torch.float64)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create temporary directories for the databases
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 folders for databases
        db_paths = []
        for i in range(3):
            folder_path = os.path.join(tmpdir, f"folder_{i}")
            os.makedirs(folder_path, exist_ok=True)

            # Create LMDB database files in each folder (2 per folder)
            for j in range(2):
                db_path = os.path.join(folder_path, f"data_{j}.aselmdb")
                db = LMDBDatabase(db_path, readonly=False)

                # Add 2 configurations to each database
                for _ in range(2):
                    # Create a water molecule using ASE's build functionality
                    atoms = molecule("H2O")

                    # Apply small random displacements to the positions
                    displacement = np.random.rand(*atoms.positions.shape) * 0.1
                    atoms.positions += displacement

                    # Set cell and PBC
                    atoms.set_cell(np.eye(3) * 5.0)
                    atoms.set_pbc(True)

                    # Add random energy, forces, and stress
                    energy = np.random.uniform(
                        -15.0, -5.0
                    )  # Random energy between -15 and -5 eV
                    forces = (
                        np.random.randn(*atoms.positions.shape) * 0.5
                    )  # Random forces
                    stress = np.random.randn(6) * 0.2  # Random stress in Voigt notation

                    # Add calculator to atoms with results
                    calc = SinglePointCalculator(
                        atoms, energy=energy, forces=forces, stress=stress
                    )
                    atoms.calc = calc

                    # Store in database
                    db.write(atoms)

                db.close()

            # Add folder path to our list
            db_paths.append(folder_path)

        # Create the dataset using paths joined with colons
        paths_str = ":".join(db_paths)
        z_table = AtomicNumberTable([1, 8])  # H and O
        dataset = LMDBDataset(file_path=paths_str, r_max=5.0, z_table=z_table)

        # Check dataset size (3 folders * 2 files * 2 configs = 12 entries)
        assert len(dataset) == 12

        # Test retrieving a single item
        item = dataset[0]
        print(item)
        assert item.positions.shape == (3, 3)  # 3 atoms, 3 coordinates
        assert hasattr(item, "energy")
        assert hasattr(item, "forces")
        assert hasattr(item, "stress")

        # Create a dataloader
        dataloader = torch_geometric.dataloader.DataLoader(
            dataset=dataset, batch_size=4, shuffle=False, drop_last=False
        )

        # Get a batch and validate it
        batch = next(iter(dataloader))

        # Verify batch properties - should have 12 atoms (4 configs * 3 atoms per water)
        assert batch.positions.shape == (12, 3)  # 12 atoms, 3 coordinates
        assert batch.energy.shape[0] == 4  # 4 energies (one per config)
        assert batch.forces.shape == (12, 3)  # Forces for each atom
        print(batch.stress.shape)
        assert batch.stress.shape == (4, 3, 3)  # Stress for each config

        # Check batch has required attributes for MACE model processing
        assert hasattr(batch, "batch")  # Batch indices
        assert batch.batch.shape[0] == 12  # One index per atom
        assert hasattr(batch, "ptr")  # Pointer for batch processing
        assert batch.ptr.shape[0] == 5  # One pointer per config + 1

        # Check that batch indices are correctly assigned
        # First 3 atoms should be from config 0, next 3 from config 1, etc.
        expected_batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        assert torch.all(batch.batch == expected_batch)

        # Check ptr correctly points to start of each configuration
        assert batch.ptr.tolist() == [0, 3, 6, 9, 12]

        # Create a batch dictionary that can be passed to a MACE model
        batch_dict = batch.to_dict()
        assert "positions" in batch_dict
        assert "energy" in batch_dict
        assert "forces" in batch_dict
        assert "stress" in batch_dict
        assert "batch" in batch_dict
        assert "ptr" in batch_dict

        # Verify additional properties required by MACE
        assert hasattr(batch, "edge_index")  # Connectivity information
        assert hasattr(batch, "shifts")  # For periodic boundary conditions
        assert hasattr(batch, "cell")  # Unit cell information

        # Test that a full batch can be processed (without errors)
        all_batches = list(dataloader)
        assert (
            len(all_batches) == 3
        )  # Should have 3 batches (12 configs with batch size 4)


def test_lmdb_dataset_honors_key_specification():
    """A CLI key specification must reach the loaded data.

    OMol-style data stores charge/spin under the row ``data`` dict (surfaced as
    ``atoms.info["charge"]`` / ``["spin"]``). A key specification built from
    ``--total_charge_key=charge --total_spin_key=spin`` must be honored; without
    one the defaults apply and ``total_charge`` / ``total_spin`` silently fall
    back to 0 / 1.
    """
    torch.set_default_dtype(torch.float64)

    charge, spin = -1, 3
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "sample.aselmdb")
        db = LMDBDatabase(db_path, readonly=False)
        db.write(molecule("H2O"), data={"charge": charge, "spin": spin})
        db.close()

        z_table = AtomicNumberTable([1, 8])  # H and O

        # Key spec as built by --total_charge_key=charge --total_spin_key=spin
        key_spec = KeySpecification.from_defaults()
        update_keyspec_from_kwargs(
            key_spec, {"total_charge_key": "charge", "total_spin_key": "spin"}
        )
        item = LMDBDataset(
            file_path=db_path,
            r_max=5.0,
            z_table=z_table,
            key_specification=key_spec,
        )[0]
        assert float(item.total_charge) == charge
        assert float(item.total_spin) == spin

        # Without a key spec the defaults apply (backward-compatible fallback)
        default_item = LMDBDataset(file_path=db_path, r_max=5.0, z_table=z_table)[0]
        assert float(default_item.total_charge) == 0.0
        assert float(default_item.total_spin) == 1.0


def test_load_dataset_for_path_forwards_key_specification():
    """`load_dataset_for_path` must forward the head's key specification to
    `LMDBDataset` for an ``.aselmdb`` input (the dispatch half of the fix)."""
    torch.set_default_dtype(torch.float64)

    charge, spin = -1, 3
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "sample.aselmdb")
        db = LMDBDatabase(db_path, readonly=False)
        db.write(molecule("H2O"), data={"charge": charge, "spin": spin})
        db.close()

        key_spec = KeySpecification.from_defaults()
        update_keyspec_from_kwargs(
            key_spec, {"total_charge_key": "charge", "total_spin_key": "spin"}
        )
        head_config = SimpleNamespace(head_name="Default", key_specification=key_spec)

        dataset = load_dataset_for_path(
            file_path=db_path,
            r_max=5.0,
            z_table=AtomicNumberTable([1, 8]),
            heads=["Default"],
            head_config=head_config,
        )
        item = dataset[0]
        assert float(item.total_charge) == charge
        assert float(item.total_spin) == spin
