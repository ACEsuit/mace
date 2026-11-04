"""Unavailable LES requests fail explicitly; polar fields retain their input path."""

from __future__ import annotations

import numpy as np
import pytest
from ase.io import read
from conftest import fp64_only
from mace_torch.calculators import MACECalculator
from mace_torch.cli.eval_configs import main
from mace_torch.cli.run_train import run
from test_mace_torch_full_batch import configuration


@pytest.mark.parametrize(
    "option,value",
    [
        ("compute_bec", True),
        ("eps_infty", 2.5),
        ("electric_field_unit", 2.0),
        ("keep_neutral", False),
    ],
)
def test_unsupported_les_options_are_not_silently_ignored(option, value):
    with pytest.raises(NotImplementedError, match=option + ".*LES"):
        MACECalculator("missing.json", **{option: value})


@fp64_only
def test_inactive_les_defaults_preserve_energy_inference_and_bec_is_refused(tmp_path):
    config = configuration(tmp_path, max_num_epochs=1)
    trained = run(config)
    atoms = read(config.data.heads["default"].train_file, index=-1)
    plain = MACECalculator(trained.checkpoint_path)
    compatible = MACECalculator(
        trained.checkpoint_path,
        compute_bec=False,
        eps_infty=None,
        electric_field_unit=1.0,
        keep_neutral=True,
    )
    for calculator in (plain, compatible):
        calculator.calculate(atoms)
        assert "bec" not in calculator.implemented_properties
    assert plain.results.keys() == compatible.results.keys()
    for key, value in plain.results.items():
        np.testing.assert_array_equal(value, compatible.results[key])
    output = tmp_path / "bec.xyz"
    with pytest.raises(ValueError, match="model does not declare 'BEC'"):
        main(
            [
                "--configs",
                str(config.data.heads["default"].train_file),
                "--model",
                str(trained.checkpoint_path),
                "--output",
                str(output),
                "--compute_bec",
            ]
        )
    assert not output.exists()
