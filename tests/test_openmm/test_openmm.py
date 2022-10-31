from e3nn.util import jit
import torch
from ase.io import read
from mace import data
from mace.tools import torch_geometric, utils
from mace.calculators import MACE_openmm

torch.set_default_dtype(torch.float64)


def test_openmm():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    at = read("test_one_mol.xyz")
    model_name = "MACE_model_run-123.model"

    model = torch.load(model_name)
    model_compiled = jit.compile(model)
    model_compiled.to(device)

    config = data.config_from_atoms(at)
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=model.r_max)
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to(device)
    res_compiled = model_compiled(batch.to_dict())

    openmm_calc = MACE_openmm(model_name, at, nl="torch_nl", device=device)
    openmm_calc2 = MACE_openmm(model_name, at, nl="torch_nl_n2", device=device)

    res_openmm = openmm_calc(torch.tensor(at.get_positions()).to(device))
    res_openmm2 = openmm_calc2(torch.tensor(at.get_positions()).to(device))

    assert torch.allclose(res_compiled["energy"], res_openmm[0])
    assert torch.allclose(res_compiled["energy"], res_openmm2[0])

    module = jit.script(openmm_calc)
    module.save("openmm_MACE.pt")
    module2 = jit.script(openmm_calc2)
    module2.save("openmm_MACE2.pt")


if __name__ == "__main__":
    test_openmm()
