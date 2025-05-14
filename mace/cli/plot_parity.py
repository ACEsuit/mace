import argparse
from typing import List

import matplotlib.pyplot as plt
from ase.io import read
from aseMolec import pltProps as pp
from aseMolec import extAtoms as ea
import numpy as np

def get_prop(db, type, prop='', peratom=False, E0={}):
    if peratom:
        N = lambda a : a.get_global_number_of_atoms()
    else:
        N = lambda a : 1
    if type == 'info':
        return np.array([a.info[prop] / N(a) for a in db if prop in a.info])
    if type == 'arrays':
        return np.array([a.arrays[prop] / N(a) for a in db if prop in a.arrays], dtype=object)
    if type == 'cell':
        return np.array(list(map(lambda a : a.cell/N(a), db)))
    if type == 'meth':
        return np.array(list(map(lambda a : getattr(a, prop)()/N(a), db)))
    if type == 'atom':
        if not E0:
            E0 = get_E0(db, prop)
        return np.array(list(map(lambda a : (np.sum([E0[s] for s in a.get_chemical_symbols()]))/N(a), db)))
    if type == 'bind':
        if not E0:
            E0 = get_E0(db, prop)
        return np.array(list(map(lambda a : (a.info['energy'+prop]-np.sum([E0[s] for s in a.get_chemical_symbols()]))/N(a), db)))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mace training statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs", help="path to MACE output XYZ configurations", required=True
    )
    parser.add_argument(
        "--output", help="output path", required=True
    )
    parser.add_argument(
        "--format", help="output format", type=str, default="pdf"
    )
    parser.add_argument(
        "--labels", help="labels for the plots", default=["REF", "MACE"], nargs="+"
    )
    parser.add_argument(
        "--plot_energy", help="plot energies", action="store_true", default=False, required=False
    )
    parser.add_argument(
        "--plot_force", help="plot forces", action="store_true", default=False, required=False
    )
    parser.add_argument(
        "--plot_stress", help="plot stresses", action="store_true", default=False, required=False
    )
    parser.add_argument(
        "--plot_polarisation", help="plot polarisations", action="store_true", default=False, required=False
    )
    parser.add_argument(
        "--plot_bec", help="plot becs", action="store_true", default=False, required=False
    )
    parser.add_argument(
        "--plot_polarisability", help="plot polarisabilities", action="store_true", default=False, required=False
    )
    return parser.parse_args()

def plot(
        data, 
        output_path: str, 
        format: str = "pdf",
        labels: List[str] = ["REF", "MACE"],
        plot_energy: bool = True,
        plot_force: bool = False,
        plot_stress: bool = False,
        plot_polarisation: bool = False,
        plot_bec: bool = False,
        plot_polarisability: bool = False
    ) -> None:
   
    if plot_energy == True:
        plt.figure(figsize=(3,3), dpi=100)
        pp.plot_prop(get_prop(data, 'info', 'REF_energy', False) / [len(d.numbers) for d in data], \
            get_prop(data, 'info', 'MACE_energy', False) / [len(d.numbers) for d in data], \
            title=r'Energy per atom $(\rm eV)$ ', labs=labels, rel=False)
        plt.tight_layout()
        plt.savefig(output_path + "_energy." + format, bbox_inches='tight')

    if plot_force == True:
        plt.figure(figsize=(9,6), dpi=100)
        plt.subplot(1,3,1)
        pp.plot_prop(np.concatenate([force[:,0] for force in get_prop(data, 'arrays', 'REF_forces', False)]), \
                    np.concatenate([force[:,0] for force in get_prop(data, 'arrays', 'MACE_forces', False)]), \
                    title=r'Force-x $(\rm eV/A)$ ', labs=labels, rel=False)
        plt.subplot(1,3,2)
        pp.plot_prop(np.concatenate([force[:,1] for force in get_prop(data, 'arrays', 'REF_forces', False)]), \
                    np.concatenate([force[:,1] for force in get_prop(data, 'arrays', 'MACE_forces', False)]), \
                    title=r'Force-y $(\rm eV/A)$ ', labs=labels, rel=False)
        plt.subplot(1,3,3)
        pp.plot_prop(np.concatenate([force[:,2] for force in get_prop(data, 'arrays', 'REF_forces', False)]), \
                    np.concatenate([force[:,2] for force in get_prop(data, 'arrays', 'MACE_forces', False)]), \
                    title=r'Force-z $(\rm eV/A)$ ', labs=labels, rel=False)
        plt.tight_layout()
        plt.savefig(output_path + "_force." + format, bbox_inches='tight')

    if plot_polarisation == True:

        # Calculate the polarisation quantum
        cell = get_prop(data, 'cell', False).reshape(-1,3,3) # [n_graphs, 3, 3]
        volume = np.linalg.det(cell)
        polarisation_quantum = cell # [n_graphs, 3, 3]

        # modulo ignore zero components to leave pol unfolded and avoid divide by 0
        polarisation_quantum = np.where(
            polarisation_quantum == 0, 
            1e12, 
            polarisation_quantum
        )

        ref_polarisation = get_prop(data, 'info', 'REF_polarisation', False).reshape(-1,3)
        pred_polarisation = get_prop(data, 'info', 'MACE_polarisation', False).reshape(-1,3)

        ref_polarisation *= volume.reshape(-1, 1)
        pred_polarisation *= volume.reshape(-1, 1)

        ref_polarisation = np.diagonal(np.fmod(np.expand_dims(ref_polarisation, axis=1), polarisation_quantum), axis1=-2, axis2=-1)
        pred_polarisation = np.diagonal(np.fmod(np.expand_dims(pred_polarisation, axis=1), polarisation_quantum), axis1=-2, axis2=-1)

        plt.figure(figsize=(9,6), dpi=100)
        plt.subplot(1,3,1)
        pp.plot_prop(ref_polarisation[:,0], \
                    pred_polarisation[:,0], \
                    title=r'Polarisation-x $(\rm eV/A^2)$ ', labs=labels, rel=False)
        plt.subplot(1,3,2)
        pp.plot_prop(ref_polarisation[:,1], \
                    pred_polarisation[:,1], \
                    title=r'Polarisation-y $(\rm eV/A^2)$ ', labs=labels, rel=False)
        plt.subplot(1,3,3)
        pp.plot_prop(ref_polarisation[:,2], \
                    pred_polarisation[:,2], \
                    title=r'Polarisation-z $(\rm eV/A^2)$ ', labs=labels, rel=False)
        plt.tight_layout()
        plt.savefig(output_path + "_polarisation." + format, bbox_inches='tight')

    if plot_bec:
        plt.figure(figsize=(9,9), dpi=100)
        plt.subplot(3,3,1)
        pp.plot_prop(np.concatenate([bec.reshape(-1,9)[:,0] for bec in get_prop(data, 'arrays', 'REF_bec', False)]), \
                    np.concatenate([bec.reshape(-1,9)[:,0] for bec in get_prop(data, 'arrays', 'MACE_bec', False)]), \
                    title=r'BECS-xx $(\rm |e|)$ ', labs=labels, rel=False)
        plt.subplot(3,3,2)
        pp.plot_prop(np.concatenate([bec.reshape(-1,9)[:,1] for bec in get_prop(data, 'arrays', 'REF_bec', False)]), \
                    np.concatenate([bec.reshape(-1,9)[:,1] for bec in get_prop(data, 'arrays', 'MACE_bec', False)]), \
                    title=r'BECS-xy $(\rm |e|)$ ', labs=labels, rel=False)
        plt.subplot(3,3,3)
        pp.plot_prop(np.concatenate([bec.reshape(-1,9)[:,2] for bec in get_prop(data, 'arrays', 'REF_bec', False)]), \
                    np.concatenate([bec.reshape(-1,9)[:,2] for bec in get_prop(data, 'arrays', 'MACE_bec', False)]), \
                    title=r'BECS-xz $(\rm |e|)$ ', labs=labels, rel=False)
        plt.subplot(3,3,4)
        pp.plot_prop(np.concatenate([bec.reshape(-1,9)[:,3] for bec in get_prop(data, 'arrays', 'REF_bec', False)]), \
                    np.concatenate([bec.reshape(-1,9)[:,3] for bec in get_prop(data, 'arrays', 'MACE_bec', False)]), \
                    title=r'BECS-yx $(\rm |e|)$ ', labs=labels, rel=False)
        plt.subplot(3,3,5)
        pp.plot_prop(np.concatenate([bec.reshape(-1,9)[:,4] for bec in get_prop(data, 'arrays', 'REF_bec', False)]), \
                    np.concatenate([bec.reshape(-1,9)[:,4] for bec in get_prop(data, 'arrays', 'MACE_bec', False)]), \
                    title=r'BECS-yy $(\rm |e|)$ ', labs=labels, rel=False)
        plt.subplot(3,3,6)
        pp.plot_prop(np.concatenate([bec.reshape(-1,9)[:,5] for bec in get_prop(data, 'arrays', 'REF_bec', False)]), \
                    np.concatenate([bec.reshape(-1,9)[:,5] for bec in get_prop(data, 'arrays', 'MACE_bec', False)]), \
                    title=r'BECS-yz $(\rm |e|)$ ', labs=labels, rel=False)
        plt.subplot(3,3,7)
        pp.plot_prop(np.concatenate([bec.reshape(-1,9)[:,6] for bec in get_prop(data, 'arrays', 'REF_bec', False)]), \
                    np.concatenate([bec.reshape(-1,9)[:,6] for bec in get_prop(data, 'arrays', 'MACE_bec', False)]), \
                    title=r'BECS-zx $(\rm |e|)$ ', labs=labels, rel=False)
        plt.subplot(3,3,8)
        pp.plot_prop(np.concatenate([bec.reshape(-1,9)[:,7] for bec in get_prop(data, 'arrays', 'REF_bec', False)]), \
                    np.concatenate([bec.reshape(-1,9)[:,7] for bec in get_prop(data, 'arrays', 'MACE_bec', False)]), \
                    title=r'BECS-zy $(\rm |e|)$ ', labs=labels, rel=False)
        plt.subplot(3,3,9)
        pp.plot_prop(np.concatenate([bec.reshape(-1,9)[:,8] for bec in get_prop(data, 'arrays', 'REF_bec', False)]), \
                    np.concatenate([bec.reshape(-1,9)[:,8] for bec in get_prop(data, 'arrays', 'MACE_bec', False)]), \
                    title=r'BECS-zz $(\rm |e|)$ ', labs=labels, rel=False)
        plt.tight_layout()
        plt.savefig(output_path + "_bec." + format, bbox_inches='tight')

    if plot_polarisability == True:
        plt.figure(figsize=(9,9), dpi=100)
        plt.subplot(3,3,1)
        pp.plot_prop(get_prop(data, 'info', 'REF_polarisability', False).reshape(-1,9)[:,0], \
                    get_prop(data, 'info', 'MACE_polarisability', False).reshape(-1,9)[:,0], \
                    title=r'Polarisability-xx $(\rm A)$ ', labs=labels, rel=False)
        plt.subplot(3,3,2)
        pp.plot_prop(get_prop(data, 'info', 'REF_polarisability', False).reshape(-1,9)[:,1], \
                    get_prop(data, 'info', 'MACE_polarisability', False).reshape(-1,9)[:,1], \
                    title=r'Polarisability-xy $(\rm A)$ ', labs=labels, rel=False)
        plt.subplot(3,3,3)
        pp.plot_prop(get_prop(data, 'info', 'REF_polarisability', False).reshape(-1,9)[:,2], \
                    get_prop(data, 'info', 'MACE_polarisability', False).reshape(-1,9)[:,2], \
                    title=r'Polarisability-xz $(\rm A)$ ', labs=labels, rel=False)
        plt.subplot(3,3,4)
        pp.plot_prop(get_prop(data, 'info', 'REF_polarisability', False).reshape(-1,9)[:,3], \
                    get_prop(data, 'info', 'MACE_polarisability', False).reshape(-1,9)[:,3], \
                    title=r'Polarisability-yx $(\rm A)$ ', labs=labels, rel=False)
        plt.subplot(3,3,5)
        pp.plot_prop(get_prop(data, 'info', 'REF_polarisability', False).reshape(-1,9)[:,4], \
                    get_prop(data, 'info', 'MACE_polarisability', False).reshape(-1,9)[:,4], \
                    title=r'Polarisability-yy $(\rm A)$ ', labs=labels, rel=False)
        plt.subplot(3,3,6)
        pp.plot_prop(get_prop(data, 'info', 'REF_polarisability', False).reshape(-1,9)[:,5], \
                    get_prop(data, 'info', 'MACE_polarisability', False).reshape(-1,9)[:,5], \
                    title=r'Polarisability-yz $(\rm A)$ ', labs=labels, rel=False)
        plt.subplot(3,3,7)
        pp.plot_prop(get_prop(data, 'info', 'REF_polarisability', False).reshape(-1,9)[:,6], \
                    get_prop(data, 'info', 'MACE_polarisability', False).reshape(-1,9)[:,6], \
                    title=r'Polarisability-zx $(\rm A)$ ', labs=labels, rel=False)
        plt.subplot(3,3,8)
        pp.plot_prop(get_prop(data, 'info', 'REF_polarisability', False).reshape(-1,9)[:,7], \
                    get_prop(data, 'info', 'MACE_polarisability', False).reshape(-1,9)[:,7], \
                    title=r'Polarisability-zy $(\rm A)$ ', labs=labels, rel=False)
        plt.subplot(3,3,9)
        pp.plot_prop(get_prop(data, 'info', 'REF_polarisability', False).reshape(-1,9)[:,8], \
                    get_prop(data, 'info', 'MACE_polarisability', False).reshape(-1,9)[:,8], \
                    title=r'Polarisability-zz $(\rm A)$ ', labs=labels, rel=False)
        plt.tight_layout()
        plt.savefig(output_path + "_polarisability." + format, bbox_inches='tight')


def run(args: argparse.Namespace) -> None:

    # Load data and prepare input
    atoms_list = read(args.configs, index=":")
    plot(
        atoms_list,  
        output_path=args.output, 
        format=args.format,
        labels=args.labels,
        plot_energy=args.plot_energy,
        plot_force=args.plot_force,
        plot_stress=args.plot_stress,
        plot_polarisation=args.plot_polarisation,
        plot_bec=args.plot_bec,
        plot_polarisability=args.plot_polarisability
    )
    

def main() -> None:
    args = parse_args()
    run(args)
    

if __name__ == "__main__":
    main()
