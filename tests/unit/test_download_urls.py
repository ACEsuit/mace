"""Pure URL-normalization tests for the foundation-model downloaders.
No network access: downloads are monkeypatched.

Split out of tests/test_calculator.py."""

from pathlib import Path

import pytest

import mace.calculators.foundations_models as foundations_models
from mace.calculators import mace_off


@pytest.mark.parametrize(
    ("model_spec", "expected_key"),
    [
        (None, "medium"),
        ("small", "small"),
        ("medium", "medium"),
        ("large", "large"),
    ],
)
def test_mace_off_download_uses_raw_githubusercontent_urls(
    tmp_path, monkeypatch, model_spec, expected_key
):
    downloaded = {}

    def fake_download(url, filename, timeout=120):
        downloaded["url"] = url
        Path(filename).write_bytes(b"fake")
        return filename, {}

    class DummyCalculator:
        def __init__(self, model_paths, device, default_dtype, **kwargs):
            self.model_paths = model_paths
            self.device = device
            self.default_dtype = default_dtype
            self.kwargs = kwargs

    monkeypatch.setattr(foundations_models, "get_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(foundations_models, "_urlretrieve_with_timeout", fake_download)
    monkeypatch.setattr(foundations_models, "MACECalculator", DummyCalculator)

    kwargs = {"device": "cpu"}
    if model_spec is not None:
        kwargs["model"] = model_spec

    calc = foundations_models.mace_off(**kwargs)

    expected_url = foundations_models.mace_off_urls[expected_key]
    expected_path = tmp_path / Path(expected_url).name
    assert downloaded["url"] == expected_url
    assert downloaded["url"].startswith("https://raw.githubusercontent.com/")
    assert calc.model_paths == str(expected_path)
    assert expected_path.exists()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true",
            "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_small.model",
        ),
        (
            "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model?raw=true",
            "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_medium.model",
        ),
        (
            "https://example.com/model.pt?download=1",
            "https://example.com/model.pt?download=1",
        ),
    ],
)
def test_normalize_github_download_url(url, expected):
    assert foundations_models._normalize_github_download_url(url) == expected


