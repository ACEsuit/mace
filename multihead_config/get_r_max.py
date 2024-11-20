import ase
import ase.data.covalent_radii as radii

def radii_from_z_pair(z1, z2):
    return (radii(z1) + radii(z2)) / 2.0
