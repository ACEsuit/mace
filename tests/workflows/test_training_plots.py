"""`--plot` and `--plot_frequency`: the plots a training run draws for itself.

Different from `mace_plot_train`, which draws them afterwards from a results log.
These come from `TrainingPlotter` inside the run, and `--plot` defaults to True, so
every default training writes one.

Until the fix in this branch it wrote none. `plot_epoch_dependence` unpacks a
`(key, axis label)` pair and then passed the pair itself as matplotlib's `label`,
which matplotlib refuses -- two labels for one dataset -- and `run_train` wraps the
call in a bare `except` that logs at DEBUG. So the default was on, the output was
nothing, and the only trace was a line nobody sees at the default log level. These
tests assert the files exist rather than that the run succeeded, because a run that
silently stops plotting still succeeds.
"""

import ase.io
import pytest

from tests.helpers import base_mace_params, make_fitting_configs, run_mace_train

pytest.importorskip("matplotlib")


def train(tmp_path, name, **extra):
    """Run a short training and return (png names, combined output)."""
    ase.io.write(tmp_path / "fit.xyz", make_fitting_configs())
    params = base_mace_params()
    params.update(
        {
            "name": name,
            "hidden_irreps": "8x0e",
            "checkpoints_dir": str(tmp_path / f"ckpt_{name}"),
            "model_dir": str(tmp_path / "model"),
            "results_dir": str(tmp_path / f"results_{name}"),
            "log_dir": str(tmp_path / f"logs_{name}"),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": 3,
            "eval_interval": 1,
            "log_level": "DEBUG",
        }
    )
    params.pop("swa", None)
    params.pop("start_swa", None)
    params.update(extra)
    result = run_mace_train(params, capture_output=True, text=True)
    assert result.returncode == 0, "plotting must never take the training down"
    output = (result.stdout or "") + (result.stderr or "")
    return sorted(p.name for p in tmp_path.rglob("*.png")), output


def test_a_default_run_draws_a_plot(tmp_path):
    """`--plot` defaults to True, so this is what every training does unasked."""
    drawn, output = train(tmp_path, "default")

    assert drawn, "the default run drew nothing"
    assert "Plotting failed" not in output


def test_the_plot_is_a_real_png(tmp_path):
    """A PNG header and a size a figure would have, since an empty or truncated
    file would satisfy a mere existence check."""
    drawn, _ = train(tmp_path, "content")
    path = next(p for p in tmp_path.rglob("*.png") if p.name == drawn[0])
    content = path.read_bytes()

    assert content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(content) > 10_000, "a PNG this small is an empty figure"


def test_the_filename_names_the_head_and_the_stage(tmp_path):
    """How a multihead run's plots stay distinguishable, and how a stage-two run's
    plots stay separate from its stage-one ones."""
    drawn, _ = train(tmp_path, "named")

    assert any("Default" in name for name in drawn), drawn
    assert any("stage_one" in name for name in drawn), drawn


def test_turning_it_off_draws_nothing(tmp_path):
    """`--plot False` skips the attempt entirely, which is the way to run without
    paying for a figure."""
    drawn, output = train(tmp_path, "off", plot="False")

    assert drawn == []
    assert "Plotting failed" not in output


def test_a_periodic_frequency_also_ends_with_a_plot(tmp_path):
    """`--plot_frequency N` adds a plotter that draws every N epochs. It writes to
    the same name each time, keyed on head and stage, so the count does not grow;
    what matters is that asking for periodic plots does not lose the final one.
    """
    drawn, output = train(tmp_path, "periodic", plot_frequency=1)

    assert drawn, "a periodic run drew nothing"
    assert "Plotting failed" not in output


def test_plotting_does_not_fail_the_run(tmp_path):
    """The bare `except` around the plotter is what let this bug hide for so long.
    Keeping the assertion that a run survives plotting means the handler can be
    narrowed later without anyone having to guess whether that was its purpose.
    """
    _, output = train(tmp_path, "survives")

    assert "Done" in output
    assert "Plotting failed" not in output
