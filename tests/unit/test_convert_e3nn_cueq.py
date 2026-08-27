import ast
import inspect

import pytest
import torch

from mace.calculators import mace_torchsim
from mace.cli import convert_e3nn_cueq, convert_e3nn_hybrid, run_train


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


@pytest.mark.parametrize("device", ["cuda", torch.device("cuda")])
def test_conversion_defaults_to_unfused(monkeypatch, device):
    monkeypatch.setattr(
        convert_e3nn_cueq,
        "extract_config_mace_model",
        lambda model: {
            "hidden_irreps": _HiddenIrrepsStub(),
            "correlation": 2,
            "num_interactions": 0,
        },
    )

    converted = convert_e3nn_cueq.run(_ConversionModelStub(), device=device)

    assert converted.cueq_config.conv_fusion is False


@pytest.mark.parametrize("device", ["cuda", torch.device("cuda")])
def test_conversion_enables_requested_cuda_fusion(monkeypatch, device):
    monkeypatch.setattr(
        convert_e3nn_cueq,
        "extract_config_mace_model",
        lambda model: {
            "hidden_irreps": _HiddenIrrepsStub(),
            "correlation": 2,
            "num_interactions": 0,
        },
    )

    converted = convert_e3nn_cueq.run(
        _ConversionModelStub(), device=device, conv_fusion=True
    )

    assert converted.cueq_config.conv_fusion is True


def test_conversion_keeps_requested_cpu_fusion_disabled(monkeypatch):
    monkeypatch.setattr(
        convert_e3nn_cueq,
        "extract_config_mace_model",
        lambda model: {
            "hidden_irreps": _HiddenIrrepsStub(),
            "correlation": 2,
            "num_interactions": 0,
        },
    )

    converted = convert_e3nn_cueq.run(
        _ConversionModelStub(), device="cpu", conv_fusion=True
    )

    assert converted.cueq_config.conv_fusion is False


def test_training_conversion_explicitly_disables_fusion():
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
    assert isinstance(keywords["conv_fusion"], ast.Constant)
    assert keywords["conv_fusion"].value is False


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


def test_torchsim_fallback_reports_conversion_failure(monkeypatch, caplog):
    class ModelStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("r_max", torch.tensor(5.0))
            self.register_buffer("atomic_numbers", torch.tensor([1]))

    def fail_conversion(*args, **kwargs):
        raise ValueError("unsupported")

    monkeypatch.setattr(mace_torchsim, "_TORCHSIM_IMPORT_ERROR", None)
    monkeypatch.setattr(convert_e3nn_cueq, "run", fail_conversion)
    monkeypatch.setattr(convert_e3nn_hybrid, "run", lambda model, **kwargs: model)

    mace_torchsim.MaceTorchSimModel(
        ModelStub(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        enable_cueq=True,
    )

    assert "cuEquivariance conversion failed" in caplog.text
    assert "conv_fusion" not in caplog.text
