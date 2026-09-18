import ast
import inspect

import pytest
import torch

from mace.calculators import mace_torchsim
from mace.cli import convert_e3nn_cueq, run_train
from mace.tools import model_script_utils
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.utils import AtomicNumberTable


class _HiddenIrrepsStub:
    @staticmethod
    def slices():
        return [slice(0, 1)]


class _ConversionModelStub(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.products = []
        self.interactions = []
        self.cueq_config = config.get("cueq_config")

    def to(self, *args, **kwargs):
        return self


@pytest.mark.parametrize(
    ("device", "kwargs", "expected"),
    [
        ("cuda", {}, True),
        (torch.device("cuda"), {}, True),
        ("cuda", {"conv_fusion": False}, False),
        ("cpu", {}, False),
    ],
)
def test_conversion_fusion_policy(monkeypatch, device, kwargs, expected):
    monkeypatch.setattr(
        convert_e3nn_cueq,
        "extract_config_mace_model",
        lambda model: {
            "hidden_irreps": _HiddenIrrepsStub(),
            "correlation": 2,
            "num_interactions": 0,
        },
    )

    converted = convert_e3nn_cueq.run(_ConversionModelStub(), device=device, **kwargs)

    assert converted.cueq_config.conv_fusion is expected


def test_training_conversion_forwards_fusion_flag():
    tree = ast.parse(inspect.getsource(run_train.run))
    conversion_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_e3nn_to_cueq"
    ]

    assert len(conversion_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in conversion_calls[0].keywords}
    assert isinstance(keywords["conv_fusion"], ast.Attribute)
    assert keywords["conv_fusion"].attr == "cueq_conv_fusion"


@pytest.mark.parametrize(
    ("device", "requested", "expected"),
    [("cuda", False, False), ("cuda", True, True), ("cpu", True, False)],
)
def test_only_cueq_uses_training_fusion_policy(
    monkeypatch, device, requested, expected
):
    args = build_default_arg_parser().parse_args(
        [
            "--name",
            "test",
            "--device",
            device,
            "--scaling",
            "no_scaling",
            "--hidden_irreps",
            "4x0e",
            "--only_cueq",
            "True",
            "--cueq_conv_fusion",
            str(requested),
        ]
    )
    args.compute_energy = True
    args.compute_forces = False
    args.compute_dipole = False
    args.compute_polarizability = False
    args.compute_magforces = False
    args.mean = 0.0
    monkeypatch.setattr(
        model_script_utils,
        "_build_model",
        lambda _args, model_config, _foundation_config, _heads: model_config[
            "cueq_config"
        ],
    )

    cueq_config, _ = model_script_utils.configure_model(
        args,
        train_loader=None,
        atomic_energies=[0.0],
        heads=["Default"],
        z_table=AtomicNumberTable([1]),
    )

    assert cueq_config.conv_fusion is expected


def test_torchsim_requests_convolution_fusion(monkeypatch):
    class ConversionComplete(Exception):
        pass

    class ModelStub(torch.nn.Module):
        pass

    calls = []

    def capture_conversion(model, **kwargs):
        calls.append(kwargs)
        raise ConversionComplete

    monkeypatch.setattr(mace_torchsim, "_TORCHSIM_IMPORT_ERROR", None)
    monkeypatch.setattr(convert_e3nn_cueq, "run", capture_conversion)

    with pytest.raises(ConversionComplete):
        mace_torchsim.MaceTorchSimModel(
            ModelStub(),
            device=torch.device("cuda"),
            dtype=torch.float32,
            enable_cueq=True,
        )

    assert calls == [{"device": "cuda", "conv_fusion": True}]
