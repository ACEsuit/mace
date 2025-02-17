""" Create molecular clusters

Core functionality taken from: https://github.com/imagdau/aseMolec/blob/main/aseMolec/anaAtoms.py
"""

from ase import neighborlist
import ase.geometry
from scipy import sparse

import ase.data
from tqdm import tqdm

cluster_key = "cluster_id"


def modif_natural_cutoffs(at, fct):
    """Modify the natural cutoffs for a given Atoms object based on the provided factor."""
    if isinstance(fct, (int, float)):
        return neighborlist.natural_cutoffs(at, mult=fct)
    elif isinstance(fct, dict):
        cutoffs = neighborlist.natural_cutoffs(at, mult=1)
        try:
            return [cutoff * fct[element] for cutoff, element in zip(cutoffs, at.get_chemical_symbols())]
        except KeyError as e:
            raise KeyError(f"Element {e.args[0]} is missing from the provided cutoff dictionary.") from e
    else:
        raise TypeError(f"Unsupported type for `fct`: expected int, float, or dict, got {type(fct).__name__}.")


def get_clusters(at, fct=1.0):
    """Get the number of molecules and the molecule IDs for a given Atoms object."""
    cutOff = modif_natural_cutoffs(at, fct)
    nbLst = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    nbLst.update(at)
    conMat = nbLst.get_connectivity_matrix(sparse=True)
    Nmol, molID = sparse.csgraph.connected_components(conMat)
    return Nmol, molID, conMat


def assign_clusters_traj(traj, fct=1.0):
    """Assigns the cluster IDs to a trajectory of Atoms objects."""
    traj_clusters = [get_clusters(at, fct)[1] for at in traj]
    [
        traj[i].arrays.update({cluster_key: traj_clusters[i]})
        for i in tqdm(range(len(traj)))
    ]
    return traj_clusters, traj


if __name__ == "__main__":
    import sys
    import ase.io
    
    # usage:
    # python create-clusters.py file.xyz output.xyz

    # Get file name from command-line arguments or prompt user if not provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("Enter the input file name (e.g., 'file.xyz'): ").strip()

    # Read trajectory from the input file
    traj = ase.io.read(input_file, ":")

    # Compute clusters
    clusters, traj = assign_clusters_traj(traj, fct=1.0)

    # Determine the default output file name
    default_output_file = f"{input_file.split('.')[0]}_cluster.xyz"

    # Get output file path, displaying the default in the prompt
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = input(
            f"Enter the output file name (press Enter to use default: '{default_output_file}'): "
        ).strip()
        if not output_file:
            output_file = default_output_file

    # Save the trajectory with cluster information to the output file
    ase.io.write(output_file, traj)
    print(f"Clustered trajectory saved to {output_file}")
