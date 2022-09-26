# Adapted from: https://gist.github.com/Linux-cpp-lisp

from typing import Tuple, List

import torch


@torch.jit.script
def torch_divmod(a, b) -> Tuple[torch.Tensor, torch.Tensor]:
    d = torch.div(a, b, rounding_mode="floor")
    m = a % b
    return d, m


@torch.jit.script
def primitive_neighbor_list_torch(
    quantities: str,
    pbc: Tuple[bool, bool, bool],
    cell,
    positions,
    cutoff: float,
    self_interaction: bool = False,
    use_scaled_positions: bool = False,
    max_nbins: int = int(1e6),
    device: str ='cuda',
) -> List[torch.Tensor]:
    """Compute a neighbor list for an atomic configuration.
    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.
    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.
    Parameters:
    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are
            * 'i' : first atom index
            * 'j' : second atom index
            * 'd' : absolute distance
            * 'D' : distance vector
            * 'S' : shift vector (number of cell boundaries crossed by the bond
              between atom i and j). With the shift vector S, the
              distances D between atoms can be computed from:
              D = positions[j]-positions[i]+S.dot(cell)
    pbc: array_like
        3-tuple indicating giving periodic boundaries in the three Cartesian
        directions.
    cell: 3x3 matrix
        Unit cell vectors. Must be completed.
    positions: list of xyz-positions
        Atomic positions.  Anything that can be converted to an ndarray of
        shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2), ...]. If
        use_scaled_positions is set to true, this must be scaled positions.
    cutoff: float or dict
        Cutoff for neighbor search. It can be:
            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood. See :func:`~ase.neighborlist.natural_cutoffs`
              for an example on how to get such a list.
    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    use_scaled_positions: bool
        If set to true, positions are expected to be scaled positions.
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.
    Returns:
    i, j, ... : array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a)-1, but the order of (i,j)
        pairs is not guaranteed.
    """

    # Naming conventions: Suffixes indicate the dimension of an array. The
    # following convention is used here:
    #     c: Cartesian index, can have values 0, 1, 2
    #     i: Global atom index, can have values 0..len(a)-1
    #     xyz: Bin index, three values identifying x-, y- and z-component of a
    #          spatial bin that is used to make neighbor search O(n)
    #     b: Linearized version of the 'xyz' bin index
    #     a: Bin-local atom index, i.e. index identifying an atom *within* a
    #        bin
    #     p: Pair index, can have value 0 or 1
    #     n: (Linear) neighbor index

    # Return empty neighbor list if no atoms are passed here
    if len(positions) == 0:
        assert False

    # Compute reciprocal lattice vectors.
    recip_cell = torch.linalg.pinv(cell).T
    b1_c, b2_c, b3_c = recip_cell[0], recip_cell[1], recip_cell[2]

    # Compute distances of cell faces.
    l1 = torch.linalg.norm(b1_c)
    l2 = torch.linalg.norm(b2_c)
    l3 = torch.linalg.norm(b3_c)
    pytorch_scalar_1 = torch.as_tensor(1.0)
    face_dist_c = torch.hstack(
        [
            1 / l1 if l1 > 0 else pytorch_scalar_1,
            1 / l2 if l2 > 0 else pytorch_scalar_1,
            1 / l3 if l3 > 0 else pytorch_scalar_1,
        ]
    )
    assert face_dist_c.shape == (3,)

    # we don't handle other fancier cutoffs
    max_cutoff: float = cutoff

    # We use a minimum bin size of 3 A
    bin_size = max(max_cutoff, 3)
    # Compute number of bins such that a sphere of radius cutoff fits into
    # eight neighboring bins.
    nbins_c = torch.maximum(
        (face_dist_c / bin_size).to(dtype=torch.long), torch.ones(3, dtype=torch.long, device=device)
    )
    nbins = torch.prod(nbins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while nbins > max_nbins:
        nbins_c = torch.maximum(nbins_c // 2, torch.ones(3, dtype=torch.long, device=device))
        nbins = torch.prod(nbins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search = torch.ceil(bin_size * nbins_c / face_dist_c).to(dtype=torch.long)
    neigh_search_x, neigh_search_y, neigh_search_z = (
        neigh_search[0],
        neigh_search[1],
        neigh_search[2],
    )

    # If we only have a single bin and the system is not periodic, then we
    # do not need to search neighboring bins
    pytorch_scalar_int_0 = torch.as_tensor(0, dtype=torch.long, device=device)
    neigh_search_x = (
        pytorch_scalar_int_0 if nbins_c[0] == 1 and not pbc[0] else neigh_search_x
    )
    neigh_search_y = (
        pytorch_scalar_int_0 if nbins_c[1] == 1 and not pbc[1] else neigh_search_y
    )
    neigh_search_z = (
        pytorch_scalar_int_0 if nbins_c[2] == 1 and not pbc[2] else neigh_search_z
    )

    # Sort atoms into bins.
    if use_scaled_positions:
        scaled_positions_ic = positions
        positions = torch.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic = torch.linalg.solve(cell.T, positions.T).T
    bin_index_ic = torch.floor(scaled_positions_ic * nbins_c).to(dtype=torch.long).to(device)
    cell_shift_ic = torch.zeros_like(bin_index_ic, device=device)

    for c in range(3):
        if pbc[c]:
            # (Note: torch.divmod does not exist in older numpies)
            cell_shift_ic[:, c], bin_index_ic[:, c] = torch_divmod(
                bin_index_ic[:, c], nbins_c[c]
            )
        else:
            bin_index_ic[:, c] = torch.clip(bin_index_ic[:, c], 0, nbins_c[c] - 1)

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = bin_index_ic[:, 0] + nbins_c[0] * (
        bin_index_ic[:, 1] + nbins_c[1] * bin_index_ic[:, 2]
    )

    # atom_i contains atom index in new sort order.
    atom_i = torch.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]

    # Find max number of atoms per bin
    max_natoms_per_bin = torch.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_natoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -torch.ones(nbins, max_natoms_per_bin.item(), dtype=torch.long, device=device)
    for i in range(int(max_natoms_per_bin.item())):
        # Create a mask array that identifies the first atom of each bin.
        mask = torch.cat(
            (torch.ones(1, dtype=torch.bool, device=device), bin_index_i[:-1] != bin_index_i[1:]),
            dim=0,
        )
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask], i] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = torch.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0

    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_natoms_per_bin**2.
    # atom_pairs_pn_np = np.indices(
    #     (max_natoms_per_bin, max_natoms_per_bin), dtype=int
    # ).reshape(2, -1)
    atom_pairs_pn = torch.cartesian_prod(
        torch.arange(max_natoms_per_bin), torch.arange(max_natoms_per_bin)
    )
    atom_pairs_pn = atom_pairs_pn.T.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neightuple_nn = []
    secnd_at_neightuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of atoms between two bins, assuming
    # that each bin contains exactly max_natoms_per_bin atoms. We then throw
    # out pairs involving pad atoms with atom index -1 below.
    binz_xyz, biny_xyz, binx_xyz = torch.meshgrid(
        torch.arange(nbins_c[2], device=device),
        torch.arange(nbins_c[1], device=device),
        torch.arange(nbins_c[0], device=device),
        indexing="ij",
    )
    # The memory layout of binx_xyz, biny_xyz, binz_xyz is such that computing
    # the respective bin index leads to a linearly increasing consecutive list.
    # The following assert statement succeeds:
    #     b_b = (binx_xyz + nbins_c[0] * (biny_xyz + nbins_c[1] *
    #                                     binz_xyz)).ravel()
    #     assert (b_b == torch.arange(torch.prod(nbins_c))).all()

    # First atoms in pair.
    _first_at_neightuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-int(neigh_search_z.item()), int(neigh_search_z.item()) + 1):
        for dy in range(-int(neigh_search_y.item()), int(neigh_search_y.item()) + 1):
            for dx in range(
                -int(neigh_search_x.item()), int(neigh_search_x.item()) + 1
            ):
                # Bin index of neighboring bin and shift vector.
                shiftx_xyz, neighbinx_xyz = torch_divmod(binx_xyz + dx, nbins_c[0])
                shifty_xyz, neighbiny_xyz = torch_divmod(biny_xyz + dy, nbins_c[1])
                shiftz_xyz, neighbinz_xyz = torch_divmod(binz_xyz + dz, nbins_c[2])
                neighbin_b = (
                    neighbinx_xyz
                    + nbins_c[0] * (neighbiny_xyz + nbins_c[1] * neighbinz_xyz)
                ).ravel()

                # Second atom in pair.
                _secnd_at_neightuple_n = atoms_in_bin_ba[neighbin_b][
                    :, atom_pairs_pn[1]
                ]

                # Shift vectors.
                # TODO: was np.resize:
                # _cell_shift_vector_x_n_np = np.resize(
                #     shiftx_xyz.reshape(-1, 1).numpy(),
                #     (int(max_natoms_per_bin.item() ** 2), shiftx_xyz.numel()),
                # ).T
                # _cell_shift_vector_y_n_np = np.resize(
                #     shifty_xyz.reshape(-1, 1).numpy(),
                #     (int(max_natoms_per_bin.item() ** 2), shifty_xyz.numel()),
                # ).T
                # _cell_shift_vector_z_n_np = np.resize(
                #     shiftz_xyz.reshape(-1, 1).numpy(),
                #     (int(max_natoms_per_bin.item() ** 2), shiftz_xyz.numel()),
                # ).T
                # this basically just tiles shiftx_xyz.reshape(-1, 1) n times
                _cell_shift_vector_x_n = shiftx_xyz.reshape(-1, 1).repeat(
                    (1, int(max_natoms_per_bin.item() ** 2))
                )
                # assert _cell_shift_vector_x_n.shape == _cell_shift_vector_x_n_np.shape
                # assert np.allclose(
                #     _cell_shift_vector_x_n.numpy(), _cell_shift_vector_x_n_np
                # )
                _cell_shift_vector_y_n = shifty_xyz.reshape(-1, 1).repeat(
                    (1, int(max_natoms_per_bin.item() ** 2))
                )
                # assert _cell_shift_vector_y_n.shape == _cell_shift_vector_y_n_np.shape
                # assert np.allclose(
                #     _cell_shift_vector_y_n.numpy(), _cell_shift_vector_y_n_np
                # )
                _cell_shift_vector_z_n = shiftz_xyz.reshape(-1, 1).repeat(
                    (1, int(max_natoms_per_bin.item() ** 2))
                )
                # assert _cell_shift_vector_z_n.shape == _cell_shift_vector_z_n_np.shape
                # assert np.allclose(
                #     _cell_shift_vector_z_n.numpy(), _cell_shift_vector_z_n_np
                # )

                # We have created too many pairs because we assumed each bin
                # has exactly max_natoms_per_bin atoms. Remove all surperfluous
                # pairs. Those are pairs that involve an atom with index -1.
                mask = torch.logical_and(
                    _first_at_neightuple_n != -1, _secnd_at_neightuple_n != -1
                )
                if mask.sum() > 0:
                    first_at_neightuple_nn += [_first_at_neightuple_n[mask]]
                    secnd_at_neightuple_nn += [_secnd_at_neightuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]

    # Flatten overall neighbor list.
    first_at_neightuple_n = torch.cat(first_at_neightuple_nn)
    secnd_at_neightuple_n = torch.cat(secnd_at_neightuple_nn)
    cell_shift_vector_n = torch.vstack(
        [
            torch.cat(cell_shift_vector_x_n),
            torch.cat(cell_shift_vector_y_n),
            torch.cat(cell_shift_vector_z_n),
        ]
    ).T

    # Add global cell shift to shift vectors
    cell_shift_vector_n += (
        cell_shift_ic[first_at_neightuple_n] - cell_shift_ic[secnd_at_neightuple_n]
    )

    # Remove all self-pairs that do not cross the cell boundary.
    if not self_interaction:
        m = torch.logical_not(
            torch.logical_and(
                first_at_neightuple_n == secnd_at_neightuple_n,
                (cell_shift_vector_n == 0).all(dim=1),
            )
        )
        first_at_neightuple_n = first_at_neightuple_n[m]
        secnd_at_neightuple_n = secnd_at_neightuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]

    # For nonperiodic directions, remove any bonds that cross the domain
    # boundary.
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neightuple_n = first_at_neightuple_n[m]
            secnd_at_neightuple_n = secnd_at_neightuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]

    # Sort neighbor list.
    i = torch.argsort(first_at_neightuple_n)
    first_at_neightuple_n = first_at_neightuple_n[i]
    secnd_at_neightuple_n = secnd_at_neightuple_n[i]
    cell_shift_vector_n = cell_shift_vector_n[i]

    # Compute distance vectors.
    distance_vector_nc = (
        positions[secnd_at_neightuple_n]
        - positions[first_at_neightuple_n]
        + cell_shift_vector_n.to(cell.dtype).matmul(cell)  # TODO! .T?
    )
    abs_distance_vector_n = torch.sqrt(
        torch.sum(distance_vector_nc * distance_vector_nc, dim=1)
    )

    # We have still created too many pairs. Only keep those with distance
    # smaller than max_cutoff.
    mask = abs_distance_vector_n < max_cutoff
    first_at_neightuple_n = first_at_neightuple_n[mask]
    secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
    cell_shift_vector_n = cell_shift_vector_n[mask]
    distance_vector_nc = distance_vector_nc[mask]
    abs_distance_vector_n = abs_distance_vector_n[mask]

    # Assemble return tuple.
    retvals = []
    for q in quantities:
        if q == "i":
            retvals += [first_at_neightuple_n]
        elif q == "j":
            retvals += [secnd_at_neightuple_n]
        elif q == "D":
            retvals += [distance_vector_nc]
        elif q == "d":
            retvals += [abs_distance_vector_n]
        elif q == "S":
            retvals += [cell_shift_vector_n]
        else:
            raise ValueError("Unsupported quantity specified.")

    return retvals