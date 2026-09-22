"""The run's own logging, and what the other ranks of a run do with theirs."""

from __future__ import annotations

import logging

import pytest
from mace_core.config.runtime import RuntimeConfig
from mace_torch.train import setup_logging


@pytest.fixture(autouse=True)
def restore_root():
    """Leave the root logger as it was found, for whatever runs next."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def runtime(tmp_path, **settings) -> RuntimeConfig:
    return RuntimeConfig(name="run", work_dir=tmp_path, **settings)


def test_the_first_rank_writes_a_file_beside_the_run(tmp_path):
    path = setup_logging(runtime(tmp_path))
    assert path is not None
    assert path == tmp_path / "logs" / "run.log"
    logging.getLogger("x").info("hello")
    assert "hello" in path.read_text(encoding="utf-8")


def test_the_other_ranks_write_no_file(tmp_path):
    """Eight processes appending to one file interleave their lines."""
    assert setup_logging(runtime(tmp_path), rank=3) is None


def test_the_other_ranks_still_report_a_failure(tmp_path, caplog):
    """A rank that fails is the one whose output matters."""
    setup_logging(runtime(tmp_path), rank=3)
    assert logging.getLogger().level == logging.WARNING


def test_the_configured_level_reaches_the_first_rank(tmp_path):
    setup_logging(runtime(tmp_path, log_level="DEBUG"))
    assert logging.getLogger().level == logging.DEBUG


def test_setting_it_up_twice_does_not_double_every_line(tmp_path):
    """A notebook does this, and so does a test."""
    path = setup_logging(runtime(tmp_path))
    setup_logging(runtime(tmp_path))
    logging.getLogger("x").info("once")
    assert path is not None
    assert path.read_text(encoding="utf-8").count("once") == 1
