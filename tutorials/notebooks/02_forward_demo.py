"""Inspect every block of the standalone energy forward."""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import jax
    import marimo as mo

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tutorials.mace_forward import forward, load_fixture

    jax.config.update("jax_enable_x64", True)
    return forward, load_fixture, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Inspect a water molecule

    This fixed, single-head model has two channels, two interaction layers,
    $\ell_{max}=1$ and correlation order two. Its contraction weights use the
    **full** CG basis. The JSON contains stored weights; path normalisation
    happens explicitly in the forward.

    Energy is assembled as
    $E=\sum_i(s\sum_t\epsilon_i^{(t)}+b)+\sum_i E^0_{Z_i}$.
    The isolated-atom energies are added after scale and shift.
    """)
    return


@app.cell
def _(forward, load_fixture, mo):
    fixture = load_fixture()
    energy, intermediates = forward(
        fixture, fixture["graph"], return_intermediates=True
    )
    mo.md(f"Energy: **{float(energy[0]):.12f} eV**")
    return (intermediates,)


@app.cell
def _(intermediates, mo):
    mo.accordion(
        {
            name: mo.md(f"Shape: `{tuple(value.shape)}`\n\n```text\n{value}\n```")
            for name, value in intermediates.items()
        }
    )
    return


if __name__ == "__main__":
    app.run()
