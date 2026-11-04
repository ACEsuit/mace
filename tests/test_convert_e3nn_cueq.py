import pytest
import torch

from mace.cli import convert_e3nn_cueq


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
    ("device", "conv_fusion", "expected"),
    [
        ("cuda", None, True),
        (torch.device("cuda"), None, True),
        ("cuda", False, False),
        ("cpu", True, True),
    ],
)
def test_conversion_selects_requested_convolution_fusion(
    monkeypatch, device, conv_fusion, expected
):
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
        _ConversionModelStub(), device=device, conv_fusion=conv_fusion
    )

    assert converted.cueq_config.conv_fusion is expected
