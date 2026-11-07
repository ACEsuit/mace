"""The native reader refuses checkpoint and Python/scripted artifact inputs."""

import os
import subprocess

import pytest

pytestmark = pytest.mark.bin_libmace


@pytest.mark.parametrize("suffix", [".json", ".safetensors", ".model"])
def test_libmace_never_falls_back_to_a_python_or_scripted_reader(tmp_path, suffix):
    path = tmp_path / ("not-a-native-package" + suffix)
    path.write_text('{"format": "not-a-native-package"}')
    result = subprocess.run(
        [os.environ["MACE_LIBMACE_DRIVER"], str(path)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert result.stderr
