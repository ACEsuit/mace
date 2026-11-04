"""The installed FPS kernel agrees with legacy at an aligned random state."""

import numpy as np
import pytest
from ase import Atoms
from mace_torch.finetune.subselect import farthest_point_indices

pytestmark = pytest.mark.usefixtures("isolated")


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_installed_fps_kernel_matches_legacy_after_aligning_start_state(seed):
    pytest.importorskip("fpsample")
    from mace.cli.fine_tuning_select import FPS

    points = np.random.RandomState(16).normal(size=(96, 8))
    frames = []
    for row in points:
        atoms = Atoms("H")
        atoms.info["mace_descriptors"] = {"H": row}
        frames.append(atoms)
    np.random.seed(seed)
    sampler = FPS(frames, 13)
    # Legacy's constructor draws an unused initial index; run() draws again.
    # Compare the same effective start while preserving V1's private RNG.
    generator = np.random.RandomState()
    generator.set_state(np.random.get_state())
    expected = sampler.run()
    state = np.random.get_state()
    actual = farthest_point_indices(points, 13, generator)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    assert len(set(actual)) == 13


def test_absent_fpsample_uses_exact_farthest_points(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "fpsample", None)
    points = np.array([[0.0], [1.0], [2.0], [10.0]])
    assert farthest_point_indices(points, 3, np.random.RandomState(0)) == [0, 3, 2]


def test_installed_fpsample_failure_is_not_changed_to_a_random_draw(monkeypatch):
    fpsample = pytest.importorskip("fpsample")

    def fail(*args, **kwargs):
        raise RuntimeError("sampling kernel failed")

    monkeypatch.setattr(fpsample, "fps_npdu_kdtree_sampling", fail)
    with pytest.raises(RuntimeError, match="sampling kernel failed"):
        farthest_point_indices(
            np.arange(12.0).reshape(6, 2), 3, np.random.RandomState(0)
        )


def test_selection_cli_uses_the_installed_kernel_and_writes_its_structures(
    tmp_path, monkeypatch
):
    import json

    from ase.io import read, write
    from mace_core.cli import main

    fpsample = pytest.importorskip("fpsample")
    source, descriptors, output = (
        tmp_path / name for name in ("pool.xyz", "rows.npy", "selected.xyz")
    )
    write(source, [Atoms("H", positions=[[index, 0.0, 0.0]]) for index in range(96)])
    np.save(descriptors, np.random.RandomState(16).normal(size=(96, 8)))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "runtime": {"seed": 7},
                "data": {
                    "heads": {
                        "pool": {
                            "train_file": str(source),
                            "subselect": {
                                "method": "fps",
                                "num_samples": 13,
                                "filtering": "none",
                                "descriptors": str(descriptors),
                            },
                        }
                    }
                },
            }
        )
    )
    calls = []
    kernel = fpsample.fps_npdu_kdtree_sampling

    def observe(*args, **kwargs):
        chosen = kernel(*args, **kwargs)
        calls.append(chosen.copy())
        return chosen

    monkeypatch.setattr(fpsample, "fps_npdu_kdtree_sampling", observe)
    assert (
        main(
            [
                "data",
                "select",
                "--config",
                str(config),
                "--head",
                "pool",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    actual = [int(atoms.positions[0, 0]) for atoms in read(output, index=":")]
    np.testing.assert_array_equal(actual, calls[0])
