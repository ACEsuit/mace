"""Teaching forward: independent execution, symmetry and production agreement."""

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tutorials.mace_forward import forward, load_fixture, spherical_harmonics

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]


def test_standalone_imports():
    tree = ast.parse((ROOT / "mace_forward.py").read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "").split(".")[0])
    assert set(imports) <= {"json", "pathlib", "jax", "numpy"}


def test_finite_energy_and_intermediates():
    fixture = load_fixture()
    energy, values = forward(fixture, fixture["graph"], return_intermediates=True)
    assert energy.shape == (1,)
    assert np.isfinite(energy).all()
    assert {"embedding", "harmonics", "radial", "layer_1", "layer_2"} <= values.keys()


def test_energy_matches_native_oracle():
    process = subprocess.run(
        [
            os.environ.get("MACE_TUTORIAL_ORACLE_PYTHON", sys.executable),
            str(ROOT / "tests/oracle.py"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    oracle = json.loads(process.stdout.splitlines()[-1])
    expected = oracle["energy"]
    directions = jnp.asarray(np.random.default_rng(53).normal(size=(20, 3)))
    np.testing.assert_allclose(
        spherical_harmonics(directions, 3), oracle["harmonics"], rtol=1e-12, atol=1e-12
    )
    fixture = load_fixture()
    np.testing.assert_allclose(
        forward(fixture, fixture["graph"]), expected, rtol=1e-10, atol=1e-10
    )


def test_rotation_translation_and_e0_order():
    fixture = load_fixture()
    expected = forward(fixture, fixture["graph"])
    graph = copy.deepcopy(fixture["graph"])
    rotation = np.linalg.qr(np.random.default_rng(42).normal(size=(3, 3)))[0]
    graph["positions"] = (
        np.asarray(graph["positions"]) @ rotation + np.array([2, -3, 1])
    ).tolist()
    np.testing.assert_allclose(forward(fixture, graph), expected, atol=1e-12)
    shifted = copy.deepcopy(fixture)
    shifted["e0"][0] = [value + 2 for value in fixture["e0"][0]]
    np.testing.assert_allclose(forward(shifted, graph) - expected, 6, atol=1e-12)


def test_harmonic_addition_theorem():
    directions = jnp.asarray(np.random.default_rng(53).normal(size=(20, 3)))
    harmonics = np.asarray(spherical_harmonics(directions, 3))
    for ell in range(4):
        np.testing.assert_allclose(
            np.sum(harmonics[:, ell * ell : (ell + 1) ** 2] ** 2, axis=1),
            2 * ell + 1,
            atol=1e-12,
        )


def test_notebooks_execute_headless():
    for notebook in sorted((ROOT / "notebooks").glob("*.py")):
        subprocess.run(
            [sys.executable, str(notebook)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
