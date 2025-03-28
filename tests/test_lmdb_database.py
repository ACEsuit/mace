import os
import tempfile

import numpy as np
import torch
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator

from mace.data.lmdb_dataset import LMDBDataset
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.fairchem_dataset.lmdb_dataset_tools import LMDBDatabase


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
