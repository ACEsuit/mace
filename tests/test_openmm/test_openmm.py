from e3nn.util import jit
import torch
from ase.io import read
from mace import data
from mace.tools import torch_geometric, utils
from mace.calculators import MACE_openmm
torch.set_default_dtype(torch.float64)

def test_openmm():
    at = read("test_one_mol.xyz")
    model_name = "MACE_model_run-123.model"

    model = torch.load(model_name)
    model_compiled = jit.compile(model)

    config = data.config_from_atoms(at)
    z_table = utils.AtomicNumberTable(
            [int(z) for z in model.atomic_numbers])
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=model.r_max
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    res_compiled = model_compiled(batch.to_dict())

    openmm_calc = MACE_openmm(model_name, at)

    res_openmm = openmm_calc(torch.tensor(at.get_positions()))

    assert torch.allclose(res_compiled["energy"], res_openmm[0])
    assert torch.allclose(res_compiled["forces"], res_openmm[1])

    module = jit.script(openmm_calc)
    module.save('openmm_MACE.pt')

if __name__ == "__main__":
    test_openmm()