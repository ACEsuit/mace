import numpy as np
import torch_geometric
import torch
from e3nn import o3

from LieACE import data, modules, tools

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
rot = np.array(irreps.D_from_angles(alpha=t(0.5), beta=t(0.5), gamma=t(0.0), k=t(0)))
positions2 = np.array(config.positions @ rot.T)
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
        max_ell=2,
        interaction_cls=modules.interaction_classes['LinearResidualInteractionBlock'],
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps('2x0e'),
        atomic_energies=atomic_energies,
        num_avg_neighbors=2,
        correlation=1,
    )
    model = modules.InvariantMultiACE(**model_config)

    #assert tools.count_parameters(model) == 75944

    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(config2, z_table=table, cutoff=3.0)

    data_loader = torch_geometric.data.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))

    output = model(batch)
    print(output)


test_multiace_model()

