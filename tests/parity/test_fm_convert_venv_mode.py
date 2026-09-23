"""The packaged extraction: a pinned legacy release in its own environment.

A user with only v1 installed has no legacy package to unpickle against, so the
converter builds an environment with a pinned legacy release and runs the same
extraction script there. Run on the anchor, it has to write exactly the
tensors the in-tree extraction writes: the pin is the release the frozen tree
is, so any difference is the environment and not the checkpoint.

Opt-in: it installs the pinned release and its dependencies from the network.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from mace_core.weights.neutral_format import read_neutral
from mace_torch.deploy.legacy import extract, extractor_path, pinned_environment

pytestmark = [pytest.mark.network]

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def test_the_packaged_extraction_writes_the_development_tensors(fp64, tmp_path_factory):
    source = GOLDEN / "models" / "tiny_scaleshift.model"
    output = tmp_path_factory.mktemp("artifacts")
    here = read_neutral(
        runpy.run_path(str(extractor_path()))["extract"](source, output / "here")
    )
    python = pinned_environment(tmp_path_factory.mktemp("pinned") / "environment")
    packaged = read_neutral(extract(source, output / "packaged", python=python))

    assert packaged.tensors.keys() == here.tensors.keys()
    for key, value in here.tensors.items():
        assert np.array_equal(value, packaged.tensors[key]), key
    assert packaged.sidecar.config == here.sidecar.config
    assert packaged.sidecar.provenance.source_version == "0.3.17"
