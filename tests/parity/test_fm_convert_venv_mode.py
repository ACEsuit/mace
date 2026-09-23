"""The packaged extraction: a pinned legacy release in its own environment.

A user with only v1 installed has no legacy package to unpickle against, so the
converter builds an environment with a pinned legacy release and runs the same
extraction script there. Run on the anchors, it has to write exactly the
tensors and configuration the in-tree extraction writes. The frozen tree is a
version that was never released, so the pin is the last release that was, and
this is the check that the two read a checkpoint the same way.

Opt-in: it installs the pinned release and its dependencies from the network.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from mace_core.weights.neutral_format import read_neutral
from mace_torch.deploy.legacy import (
    PINNED_LEGACY,
    extract,
    extractor_path,
    pinned_environment,
)

pytestmark = [pytest.mark.network]

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


@pytest.fixture(scope="module")
def pinned(tmp_path_factory):
    return pinned_environment(tmp_path_factory.mktemp("pinned") / "environment")


@pytest.mark.parametrize("anchor", ["tiny_scaleshift", "tiny_mace"])
def test_the_packaged_extraction_writes_the_development_tensors(
    fp64, tmp_path, pinned, anchor
):
    source = GOLDEN / "models" / f"{anchor}.model"
    here = read_neutral(
        runpy.run_path(str(extractor_path()))["extract"](source, tmp_path / "here")
    )
    packaged = read_neutral(extract(source, tmp_path / "packaged", python=pinned))

    assert packaged.tensors.keys() == here.tensors.keys()
    for key, value in here.tensors.items():
        assert np.array_equal(value, packaged.tensors[key]), key
    assert packaged.sidecar.config == here.sidecar.config
    assert packaged.sidecar.derived == here.sidecar.derived
    assert packaged.sidecar.provenance.source_version == PINNED_LEGACY.split("==")[1]
