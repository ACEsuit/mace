"""`--log_level`, which decides what a run says and not what it records.

`setup_logger` puts the root logger at DEBUG and then attaches three handlers:
the console and `<tag>.log` at the requested level, and `<tag>_debug.log` at
DEBUG unconditionally. So the flag quiets two of the three destinations, and the
one it does not touch is the one on disk -- raising the level does not make a run
write less, it makes it say less.

Nothing covered any of that, and the flag is on every run.
"""

import logging

import pytest

from mace.tools.utils import setup_logger


@pytest.fixture(name="clean_root")
def fixture_clean_root():
    """`setup_logger` mutates the root logger, so it has to be handed back."""
    root = logging.getLogger()
    handlers, filters, level = root.handlers[:], root.filters[:], root.level
    root.handlers, root.filters = [], []
    yield root
    for handler in root.handlers:
        handler.close()
    root.handlers, root.filters, root.level = handlers, filters, level


def logs(clean_root, tmp_path, level, rank=0):
    """Set up logging, emit one record per level, and read back both files."""
    setup_logger(level=level, tag="run", directory=str(tmp_path), rank=rank)
    logging.info("an info line")
    logging.debug("a debug line")
    for handler in clean_root.handlers:
        handler.flush()
    return (
        (tmp_path / "run.log").read_text(encoding="utf-8"),
        (tmp_path / "run_debug.log").read_text(encoding="utf-8"),
    )


def test_the_default_level_keeps_debug_out_of_the_main_log(clean_root, tmp_path):
    main, _ = logs(clean_root, tmp_path, "INFO")

    assert "an info line" in main
    assert "a debug line" not in main


def test_debug_lets_it_through(clean_root, tmp_path):
    """`--log_level=DEBUG` is how the configuration dump and the swallowed
    plotter and compile failures become visible: they are all logged at DEBUG."""
    main, _ = logs(clean_root, tmp_path, "DEBUG")

    assert "an info line" in main
    assert "a debug line" in main


@pytest.mark.parametrize("level", ["INFO", "DEBUG", "WARNING"])
def test_the_debug_file_is_written_whatever_the_level(clean_root, tmp_path, level):
    """The handler for `<tag>_debug.log` is pinned at DEBUG and the root logger is
    too, so this file has everything at every setting. It is what makes a failed
    run diagnosable after the fact, and it is also why `--log_level=WARNING` does
    not reduce what a long run writes to disk.
    """
    _, debug = logs(clean_root, tmp_path, level)

    assert "an info line" in debug
    assert "a debug line" in debug


def test_a_level_above_info_silences_the_main_log_without_silencing_the_run(
    clean_root, tmp_path
):
    main, debug = logs(clean_root, tmp_path, "WARNING")

    assert main == ""
    assert "an info line" in debug


def test_only_rank_zero_logs(clean_root, tmp_path):
    """The filter sits on the logger rather than on a handler, so a non-zero rank
    writes nothing at all -- neither file, not even the debug one. That is what
    keeps a distributed run's log readable, and also why a rank-1 crash leaves no
    trace of its own."""
    main, debug = logs(clean_root, tmp_path, "DEBUG", rank=1)

    assert main == ""
    assert debug == ""
