from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration, config_from_atoms
from fairchem.core.datasets import AseDBDataset
from torch.utils.data import Dataset
from mace.tools.utils import AtomicNumberTable
from ase.io.extxyz import save_calc_results
import numpy as np
import os

from tqdm import tqdm

class LMDBDataset(Dataset):
    def __init__(self, file_path, r_max, z_table, **kwargs):
        dataset_paths = file_path.split(":") # using : split multiple paths
        # make sure each of the path exist
        for path in dataset_paths:
            assert os.path.exists(path)
        config_kwargs = {}
        super(LMDBDataset, self).__init__() # pylint: disable=super-with-arguments
        self.AseDB = AseDBDataset(config=dict(src=dataset_paths, **config_kwargs))
        self.r_max = r_max
        self.z_table = z_table

        self.kwargs = kwargs
        self.transform = kwargs['transform'] if 'transform' in kwargs else None

    def __len__(self):
        return len(self.AseDB)

    #def __getitem__(self, indices):
    #    single_datum = False
    #    if isinstance(indices, int):  # Handle single index case for compatibility
    #        indices = [indices]
    #        single_datum = True
    #    
    #    atomic_data_list = []
    #    for index in indices:
    #        try:
    #            atoms = self.AseDB.get_atoms(self.AseDB.ids[index])
    #        except Exception as e:
    #            import ipdb; ipdb.set_trace()
    #            print("Error at index:", index)
    #            print("Total IDs:", len(self.AseDB.ids))
    #            raise e

    #        assert np.sum(atoms.get_cell() == atoms.cell) == 9

    #        config = Configuration(
    #            atomic_numbers=atoms.numbers,
    #            positions=atoms.positions,
    #            energy=atoms.calc.results['energy'],
    #            forces=atoms.calc.results['forces'],
    #            stress=atoms.calc.results['stress'],
    #            virials=np.zeros(atoms.get_stress().shape),
    #            dipole=np.zeros(atoms.get_forces()[0].shape),
    #            charges=np.zeros(atoms.numbers.shape),
    #            weight=1.0,
    #            head=None,
    #            energy_weight=1.0,
    #            forces_weight=1.0,
    #            stress_weight=1.0,
    #            virials_weight=1.0,
    #            config_type=None,
    #            pbc=np.array(atoms.pbc),
    #            cell=np.array(atoms.cell),
    #            alex_config_id=None,
    #        )

    #        if config.head is None:
    #            config.head = self.kwargs.get("head")
    #        
    #        try:
    #            atomic_data = AtomicData.from_config(
    #                config,
    #                z_table=self.z_table,
    #                cutoff=self.r_max,
    #                heads=self.kwargs.get("heads", ["Default"]),
    #            )
    #        except Exception as e:
    #            import ipdb; ipdb.set_trace()
    #            raise e

    #        if self.transform:
    #            atomic_data = self.transform(atomic_data)
    #        
    #        atomic_data_list.append(atomic_data)

    #    if single_datum:
    #        return atomic_data_list[0]

    #    return atomic_data_list 
    def __getitem__(self, index):
        try:
            atoms = self.AseDB.get_atoms(self.AseDB.ids[index])
        except:
            import ipdb; ipdb.set_trace()
            print(index)
            print(len(self.AseDB.ids))
            raise NotImplementedError

        assert np.sum(atoms.get_cell() == atoms.cell) == 9

        #import ipdb; ipdb.set_trace()
        config = Configuration(
            atomic_numbers=atoms.numbers,
            positions=atoms.positions,
            energy=atoms.calc.results['energy'],
            forces=atoms.calc.results['forces'],
            stress=atoms.calc.results['stress'] if "stress" in atoms.calc.results.keys() else np.zeros(6),
            virials=np.zeros(atoms.get_stress().shape) if "stress" in atoms.calc.results.keys() else np.zeros(6),
            dipole=np.zeros(atoms.get_forces()[0].shape),
            charges=np.zeros(atoms.numbers.shape),
            weight=1.0,
            head=None, # do not asign head according to h5
            energy_weight=1.0,
            forces_weight=1.0,
            stress_weight=1.0,
            virials_weight=1.0,
            config_type=None,
            pbc=np.array(atoms.pbc),
            cell=np.array(atoms.cell),
            alex_config_id=None,
        )
        if config.head is None:
            config.head = self.kwargs.get("head")
        try:
            atomic_data = AtomicData.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=self.r_max,
                    heads=self.kwargs.get("heads", ["Default"]),
            )
        except Exception as e:
            import ipdb; ipdb.set_trace()
            raise e

        if self.transform:
            atomic_data = self.transform(atomic_data)
        return atomic_data

    def get_charge_spin(self):
        total_num = len(self)
        self.charges = []
        self.spins = []

        for i in tqdm(range(total_num)):
            atoms = self.AseDB.get_atoms(self.AseDB.ids[i])
            charge = atoms.info["charge"]
            spin = atoms.info['spin']

            self.charges.append(charge) 
            self.spins.append(spin) 

        self.charges = np.array(self.charges)
        self.spins = np.array(self.spins)

        #self.nutual_mask_charge = self.charges == 0
        #self.nutual_mask_spins = self.spins == 0

        #import ipdb; ipdb.set_trace()

        #if set_nutual:
        #    # Filter IDs where both charge and spin are zero (neutral)
        #    neutral_mask = self.nutual_mask_charge & self.nutual_mask_spins
        #    self.AseDB.ids = [self.AseDB.ids[i] for i in range(len(self.AseDB.ids)) if neutral_mask[i]] 

if __name__ == "__main__":
    #db = LMDBDataset(None, 5.0, AtomicNumberTable(range(1, 120)))
    #print(db[0])

    from mace import tools
    omat_atoms = {1:-1.1358876323426095,2:-0.003795109740622673,3:-0.26137794687222415,4:-0.025045132573949584,5:-0.3242568108155506,6:-1.3723587812092162,7:-3.13632841313978,8:-1.5331066241287237,9:-0.5984956871214576,10:-0.01241601,11:-0.2169899231508232,12:-0.10499894119050433,13:-0.23216891833743025,14:-0.8597521216915182,15:-1.912816962054931,16:-0.9083135205363799,17:-0.28072137932145075,18:-0.07943334653030147,19:-0.25878452937466057,20:-0.13137651548624146,21:-2.2169836535390353,22:-2.5816718636259735,23:-3.865056277692038,24:-5.674557896827097,25:-5.4812706052458,26:-3.6743144856228827,27:-1.9583492267837823,28:-1.1959642876298515,29:-0.9440072214248614,30:-0.33965318063803035,31:-0.47645693440259207,32:-0.937479426049947,33:-1.715518526828663,34:-0.7855321438234154,35:-0.24661132275887546,36:0.33636220087476904,37:-0.18729276685342847,38:-0.09892084696028075,39:-2.129067584703469,40:-2.1740677919375493,41:-3.1647973850043067,42:-4.478210677081609,43:-3.347148839329221,44:-2.417839678192404,45:-1.5684019761348906,46:-1.4477836815925238,47:-0.4596111779579501,48:-0.35548998713701246,49:-0.5363832847724515,50:-1.030269938448027,51:-1.4459768380792797,52:-0.6974480097100907,53:-0.21332890591466697,54:0.30428744824468235,55:-0.14311310848823514,56:-0.04342178720121778,57:-0.4849998869902074,58:-1.2742473882695573,59:-0.32554957939034534,60:-0.3300357666879792,61:-0.32311962359203483,62:-0.2859524766258597,63:-8.223818889121638,64:-10.311804681812571,65:-0.32859738500996066,66:-0.3370070351307926,67:-0.34250826785830235,68:-0.3663075717539106,69:-0.21974414042460674,70:3.300950065870339,71:-0.3824266025326068,72:-3.5570552482914573,73:-3.6055668877270897,74:-4.756266378767607,75:-4.882582483949777,76:-3.117102848040901,77:-1.4628088344196877,78:-0.5113053914398008,79:-0.33122096593191547,80:-0.25496294392435825,81:-0.44357719516515814,82:-0.9374740611599515,83:-1.3499626208523796,89:-0.32851999482612837,90:-0.8469682906659018,91:-2.444020846514142,92:-4.937428371399484,93:-7.716344114285612,94:-10.681081032695984}

    atomic_numbers = list(omat_atoms.keys())
    print(atomic_numbers)

    train_file = "/lustre/fsn1/projects/rech/gax/unh55hx/origin_data/data/omat/train/aimd-from-PBE-1000-npt" 
    z_table = tools.get_atomic_number_table_from_zs(atomic_numbers)
    head = "omat"
    heads = ["omat", "spice"]

    import ipdb; ipdb.set_trace()

    db = LMDBDataset(train_file, r_max=6.0, z_table=z_table, head=head, heads=list(heads))
    #head_args.train_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys())

    from mace.tools import torch_geometric 
    loader = torch_geometric.dataloader.DataLoader(
        db, batch_size=128, num_workers=12, shuffle=False
    )
    for b in loader:
        print(b)
