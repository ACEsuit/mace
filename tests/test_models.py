import numpy as np
import torch_geometric
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from LieACE import data, modules, tools

torch.set_default_dtype(torch.float64)

config = data.utils.Configuration(
  atomic_numbers=np.array([8, 1, 1]),
    positions=np.array([
        [0.0, -2.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        
    ]),
    forces=np.array([
        [0.0, -1.3, 0.0],
        [1.0, 0.2, 0.0],
        [0.0, 1.1, 0.3],
    ]),
    energy=-1.5,
)

table = tools.AtomicNumberTable([1, 8])

irreps = o3.Irreps("1o")
t = torch.tensor
r  = R.from_euler('z',60,degrees=True)
rot = r.as_matrix()
positions2 = np.array(rot @ config.positions.T).T
config2 = data.utils.Configuration(
  atomic_numbers=np.array([8, 1, 1]),
    positions=positions2,
    forces=np.array([
        [0.0, -1.3, 0.0],
        [1.0, 0.2, 0.0],
        [0.0, 1.1, 0.3],
    ]),
    energy=-1.5,
)

def test_multiace_model():
    atomic_energies = np.array([1.0, 3.0], dtype=float)
    model_config = dict(
        r_max=4,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=modules.interaction_classes['ComplexAgnosticResidualInteractionBlock'],
        num_interactions=3,
        num_elements=2,
        hidden_irreps=o3.Irreps('32x0e'),
        num_radial_coupling=5,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        correlation=3,
    )
    model = modules.InvariantMultiACE(**model_config)


    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(config2, z_table=table, cutoff=3.0)

    data_loader = torch_geometric.data.DataLoader(
        dataset=[atomic_data, atomic_data2,],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))

    output = model(batch)
    assert torch.allclose(output['energy'][0],output['energy'][1])


test_multiace_model()


def test_model_cuda():
    atomic_energies = np.array([1.0, 3.0], dtype=float)
    model_config = dict(
        r_max=4,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=modules.interaction_classes['ComplexAgnosticResidualInteractionBlock'],
        num_interactions=3,
        num_elements=2,
        hidden_irreps=o3.Irreps('32x0e'),
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        correlation=2,
        device='cuda'
    )
    model = modules.InvariantMultiACE(**model_config).to('cuda')
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)

    data_loader = torch_geometric.data.DataLoader(
        dataset=[atomic_data, atomic_data, atomic_data, atomic_data, atomic_data],
        batch_size=5,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to('cuda')
    output = model(batch)
    assert torch.allclose(output['energy'][0],output['energy'][1])
test_model_cuda()

def benchmark_model():
    atomic_energies = np.array([1.0, 3.0], dtype=float)
    model_config = dict(
        r_max=4,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=modules.interaction_classes['ComplexAgnosticInteractionBlock'],
        num_interactions=3,
        num_elements=2,
        hidden_irreps=o3.Irreps('32x0e'),
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        correlation=2,
        device='cuda'
    )
    model = modules.InvariantMultiACE(**model_config).to('cuda')
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)

    data_loader = torch_geometric.data.DataLoader(
        dataset=[atomic_data, atomic_data, atomic_data, atomic_data, atomic_data],
        batch_size=5,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to('cuda')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
       for _ in range(100):  # any normal python code, really!
           batch = next(iter(data_loader)).to('cuda')
           y = model(batch)
    # NOTE: some columns were removed for brevity
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
