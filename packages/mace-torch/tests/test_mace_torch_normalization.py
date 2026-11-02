"""Normalization survives training-data boundaries and head remapping."""

import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.cli import main
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.normalization import NormalizationValues
from mace_core.observables import ObservableCatalogue, ObservableSpec
from mace_torch.deploy.loader import load_deployed
from mace_torch.finetune.stages import build
from mace_torch.models.normalization import ObservableNormalization
from mace_torch.train.checkpoint import write_model

from tests.golden.harness import CLOSED_FORM_FP64


@pytest.mark.parametrize("buffer", ["scale", "shift"])
def test_checkpoint_cannot_broadcast_one_heads_buffers_over_other_heads(buffer):
    spec = ObservableSpec(name="q", irreps="0e", per_atom=False, units="u")
    normalizer = ObservableNormalization(
        spec,
        [NormalizationValues(1.0, (0.0,)), NormalizationValues(2.0, (3.0,))],
        torch.float64,
    )
    state = normalizer.to_canonical()
    state[buffer] = state[buffer][:1]
    with pytest.raises(ValueError, match=f"normalization.{buffer}: expected shape"):
        normalizer.load_canonical(state)


@pytest.mark.parametrize(
    "per_atom,extensive", [(True, False), (False, False), (False, True)]
)
def test_normalization_commutes_with_rotations_and_keeps_gradients(per_atom, extensive):
    spec = ObservableSpec(
        name="q", irreps="0e+1o", per_atom=per_atom, extensive=extensive, units="u"
    )
    normalizer = ObservableNormalization(
        spec,
        [
            NormalizationValues(2.0, (3.0, 0.0, 0.0, 0.0)),
            NormalizationValues(4.0, (5.0, 0.0, 0.0, 0.0)),
        ],
        torch.float64,
    )
    head, batch, ptr = (
        torch.tensor([1, 0]),
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 2, 3]),
    )
    values = torch.arange((3 if per_atom else 2) * 4, dtype=torch.float64).reshape(
        -1, 4
    )
    values.requires_grad_()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    transformed = torch.cat([values[:, :1], values[:, 1:] @ rotation], dim=1)
    expected = normalizer(values, head, batch, ptr)
    actual = normalizer(transformed, head, batch, ptr)
    torch.testing.assert_close(actual[:, :1], expected[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[:, 1:], expected[:, 1:] @ rotation, rtol=0, atol=0
    )
    expected.sum().backward()
    scales = torch.tensor(
        [4.0, 4.0, 2.0] if per_atom else [4.0, 2.0], dtype=torch.float64
    )
    torch.testing.assert_close(
        values.grad, scales[:, None].expand_as(values), rtol=0, atol=0
    )
    counts = torch.tensor([2.0, 1.0], dtype=torch.float64) if extensive else 1.0
    shifts = torch.tensor(
        [5.0, 5.0, 3.0] if per_atom else [5.0, 3.0], dtype=torch.float64
    )
    torch.testing.assert_close(expected[:, 0], values[:, 0] * scales + shifts * counts)


@fp64_only
@pytest.mark.parametrize("normalize_second", [False, True])
def test_head_statistics_reload_selection_and_finetune(tmp_path, normalize_second):
    heads = {}
    for head, e0, multiplier in [("a", 10.0, 1.0), ("b", 200.0, 2.0)]:
        frames = []
        for value in (1.0, 3.0):
            atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
            atoms.info.update(
                REF_energy=2 * (e0 + value * multiplier), q=2 * value * multiplier
            )
            frames.append(atoms)
        write(tmp_path / f"{head}.xyz", frames)
        frames[0].info.update(REF_energy=1e9, q=1e9)
        write(tmp_path / f"{head}-valid.xyz", frames[:1])
        heads[head] = {
            "train_file": str(tmp_path / f"{head}.xyz"),
            "valid_file": str(tmp_path / f"{head}-valid.xyz"),
            "e0s": {"kind": "table", "values": {1: e0}},
        }
    catalogue = ObservableCatalogue(
        observables=(
            ObservableSpec(
                name="energy", irreps="0e", per_atom=False, extensive=True, units="eV"
            ),
            ObservableSpec(
                name="q", irreps="0e", per_atom=False, extensive=True, units="u"
            ),
        )
    )
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path)},
            "data": {"heads": heads, "pin_memory": False},
            "catalogue": catalogue.model_dump(),
            "model": {
                "observables": ["energy", "q"],
                "num_channels": 2,
                "num_interactions": 1,
                "max_ell": 0,
                "max_L": 0,
                "correlation": 1,
                "radial_mlp": [4],
                "normalization": {
                    "energy": {head: {"kind": "std"} for head in heads},
                    "q": {
                        head: {"kind": "std"}
                        for head in heads
                        if head == "a" or normalize_second
                    },
                },
            },
            "loss": {"weights": {"energy": 1.0, "q": 1.0}},
        }
    )
    built = build(config)
    energy = built.model.get_submodule("backbone.outputs.energy_head")
    torch.testing.assert_close(energy.scale, torch.tensor([1.0, 2.0]), rtol=0, atol=0)
    torch.testing.assert_close(energy.shift, torch.tensor([2.0, 4.0]), rtol=0, atol=0)
    normalizer = built.model.get_submodule("backbone.outputs.normalization.q")
    assert isinstance(normalizer, ObservableNormalization)
    torch.testing.assert_close(
        normalizer.scale,
        torch.tensor([1.0, 2.0 if normalize_second else 1.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        normalizer.shift[:, 0],
        torch.tensor([2.0, 4.0 if normalize_second else 0.0]),
        rtol=0,
        atol=0,
    )
    path = write_model(tmp_path / "model", built.model, built.metadata)
    loaded = load_deployed(path)
    assert loaded.config.model.normalization == config.model.normalization
    batch = next(iter(built.data.train_eval_loaders["b"])).graph
    expected = built.model(batch, compute=())
    for name in ("energy", "q"):
        torch.testing.assert_close(
            loaded.engine(batch, compute=()).get(name),
            expected.get(name),
            rtol=0,
            atol=0,
        )

    selected = tmp_path / "selected.safetensors"
    assert (
        main(
            [
                "model",
                "select-head",
                "--model",
                str(path),
                "--head",
                "b",
                "--output",
                str(selected),
            ]
        )
        == 0
    )
    selected_model = load_deployed(selected)
    selected_batch = {**batch, "head": torch.zeros_like(batch["head"])}
    for name in ("energy", "q"):
        torch.testing.assert_close(
            selected_model.engine(selected_batch, compute=()).get(name),
            expected.get(name),
            rtol=CLOSED_FORM_FP64.rtol,
            atol=CLOSED_FORM_FP64.atol,
        )

    # A renamed fine-tune head inherits buffers, even on differently labelled data.
    document = config.to_resolved_dict()
    document["data"]["heads"] = {"new": {**heads["a"], "readout_from": "b"}}
    document["finetune"]["foundation_model"] = str(path)
    document["model"] = {"observables": ["energy", "q"]}
    fine = build(ResolvedConfig.from_dict(document))
    for role in ("energy_head", "normalization.q"):
        source = loaded.model.get_submodule(f"outputs.{role}")
        target = fine.model.get_submodule(f"backbone.outputs.{role}")
        for buffer in ("scale", "shift"):
            torch.testing.assert_close(
                target.get_buffer(buffer), source.get_buffer(buffer)[1:], rtol=0, atol=0
            )

    for file in tmp_path.glob("*.xyz"):
        file.unlink()
    # Loading and evaluating never consult training files to reconstruct buffers.
    reloaded = load_deployed(selected)
    torch.testing.assert_close(
        reloaded.engine(selected_batch, compute=()).get("q"),
        expected.get("q"),
        rtol=CLOSED_FORM_FP64.rtol,
        atol=CLOSED_FORM_FP64.atol,
    )
