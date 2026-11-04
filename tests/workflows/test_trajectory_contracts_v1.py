"""Native training recipes resume with the same weights and optimizer state.

Historical final vectors remain frozen. Numerical comparisons with legacy are
in tests/parity; this checks each native recipe through the actual launcher.
"""

import copy
import json

import numpy as np
import pytest
from ase.io import write

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.workflows.test_replay_elements_v1 import published_small
from tests.workflows.test_run_train_dipole_polar import fixture_fitting_configs
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, on_v1, training_records

pytestmark = needs_the_v1_engine
RECIPES = [
    ("energy", "reference"), ("multihead", "reference"),
    ("dipole", "reference"), ("dielectric", "reference"),
    pytest.param("energy", "cueq", marks=[pytest.mark.gpu, pytest.mark.cueq]),
    *[pytest.param(name, "reference", marks=pytest.mark.network) for name in
      ("foundation", "foundation_multi", "foundation_json", "foundation_pseudo")],
    pytest.param("foundation_json", "cueq", marks=[pytest.mark.network, pytest.mark.gpu, pytest.mark.cueq]),
]


@pytest.mark.parametrize("recipe,backend", RECIPES)
def test_recipe_training_and_resume(tmp_path, fitting_configs, request, recipe, backend):
    from mace_torch.calculators import MACECalculator
    from mace_torch.deploy.loader import load_deployed
    from safetensors.numpy import load_file

    device = "cuda" if backend == "cueq" else "cpu"
    foundation = recipe.startswith("foundation")
    multihead = recipe == "multihead" or foundation and recipe != "foundation"
    names = ("memory", "pbe", "mp2") if multihead else ("default",)
    observables = ["dipole", "polarizability"] if recipe == "dielectric" else (
        ["dipole"] if recipe == "dipole" else ["energy", "forces", "stress"])
    document = {
        "runtime": {"name": "fit", "seed": 5, "device": device},
        "precision": {"model": "float64"},
        "data": {"heads": {}, "valid_fraction": 0.2, "pin_memory": False},
        "model": {"model": recipe if recipe in ("dipole", "dielectric") else "scale_shift",
                  "backend": backend, "observables": observables, "r_max": 3.5,
                  "num_channels": 4, "hidden_irreps": "0e+1o+2e", "max_ell": 2,
                  "correlation": 3, "readout": {"mlp_irreps": "4x0e+4x1o+4x2e"}},
        "training": {"max_num_epochs": 2, "batch_size": 5, "valid_batch_size": 5,
                     "lr": 0.001, "ema": {"enabled": True, "decay": 0.99},
                     "scheduler": {"kind": {"kind": "exponential", "gamma": 0.9}}},
        "loss": {"weights": {name: 1.0 for name in observables}},
    }
    if recipe in ("dipole", "dielectric"):
        document["loss"]["use_structure_weights"] = {name: False for name in observables}
    for number, name in enumerate(names):
        frames = [atoms.copy() for atoms in fitting_configs]
        for index, atoms in enumerate(frames):
            if recipe in ("dipole", "dielectric"):
                atoms.info["config_weight"] = (0.0, 2.0, 3.0)[index % 3]
            if len(atoms) > 1:
                atoms.info["REF_energy"] += number * 0.2
            for old, new in (("REF_dipoles", "dipole"), ("REF_polarizability", "polarizability")):
                if old in atoms.info:
                    atoms.info[new] = atoms.info.pop(old)
        source = tmp_path / f"{name}.xyz"
        write(source, frames)
        document["data"]["heads"][name] = {"train_file": str(source)}
    if foundation:
        path = request.getfixturevalue("published_small")
        parent = load_deployed(path)
        source_head, = parent.heads
        document["finetune"] = {"foundation_model": str(path), "element_table": "data"}
        document["model"] = {"backend": backend, "observables": observables}
        for head in document["data"]["heads"].values():
            head.update(readout_from=source_head, e0s={"kind": "foundation", "head": source_head})
        del parent
        if recipe == "foundation_pseudo":
            document["finetune"]["pseudolabels"] = {"enabled": True, "heads": ["memory"], "batch_size": 5}
    flags = []
    if recipe == "foundation_json":
        heads = {}
        for number, name in enumerate(("pbe", "mp2")):
            table = tmp_path / f"{name}-e0s.json"
            table.write_text(json.dumps({1: number * 0.1, 8: -number * 0.2}))
            heads[name] = {"E0s": str(table)}
        flags = ["--heads", json.dumps(heads)]

    def run(directory, epochs, restart=False):
        settings = copy.deepcopy(document)
        settings["training"]["max_num_epochs"] = epochs
        settings["runtime"]["restart_latest"] = restart
        config = tmp_path / f"{directory}.json"
        config.write_text(json.dumps(settings))
        result = on_v1("--config", str(config), "--work_dir", str(tmp_path / directory), *flags)
        assert result.returncode == 0, result.stderr
        if restart:
            assert "Resuming from" in result.stderr
            assert "start afresh" not in result.stderr
        return tmp_path / directory

    whole = run("whole", 2)
    split = run("split", 1)
    before = load_file(split / "fit.run-000001.safetensors")
    run("split", 2, restart=True)
    expected = load_file(whole / "fit.run-000002.safetensors")
    actual = load_file(split / "fit.run-000002.safetensors")
    assert expected.keys() == actual.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=FP64_CPU_REFERENCE.atol,
                                   rtol=FP64_CPU_REFERENCE.rtol, err_msg=key)
    sidecars = [json.loads((root / "fit.run-000002.json").read_text()) for root in (whole, split)]
    for key in ("state", "optimizer", "scheduler", "ema", "model_tensors"):
        assert sidecars[0][key] == sidecars[1][key], key
    assert any(not np.array_equal(before[key], actual[key]) for key in sidecars[1]["model_tensors"])
    records = training_records(split, "fit")
    assert {row["epoch"] for row in records if row["mode"] == "opt"} == {0, 1}
    assert all(np.isfinite(row["loss"]) for row in records)
    assert {row["head"] for row in records if row["mode"] == "valid"} == set(names)
    loaded = load_deployed(split / "fit.safetensors", device=device)
    assert set(loaded.heads) == set(names)
    assert loaded.config.model.clebsch_gordan_basis == "reduced"
    assert dict(loaded.config.loss.use_structure_weights) == document["loss"].get(
        "use_structure_weights", {}
    )
    if recipe == "foundation_json":
        for number, name in enumerate(("pbe", "mp2")):
            assert loaded.e0s[name] == {1: number * 0.1, 8: -number * 0.2}
    if recipe == "foundation_pseudo":
        from mace_torch.finetune.pseudolabels import load_pseudolabel_artifact
        labels, provenance = load_pseudolabel_artifact(split / "pseudolabels/memory")
        assert labels and provenance.foundation_head == source_head
        assert all({"energy", "forces", "stress"} <= item.properties.keys() for item in labels)
    accelerated = {name: module for name, module in loaded.engine.named_modules()
                   if type(module).__name__.startswith("CuEq")}
    called = set()
    hooks = [module.register_forward_hook(lambda m, a, o, name=name: called.add(name))
             for name, module in accelerated.items()]
    try:
        for name in names:
            calculator = MACECalculator(models=loaded, head=name, device=device)
            for atoms in fitting_configs:
                calculator.calculate(atoms)
                for observable in observables:
                    assert np.isfinite(calculator.results[observable]).all()
    finally:
        for hook in hooks:
            hook.remove()
    if backend == "cueq":
        assert accelerated and called == accelerated.keys()
