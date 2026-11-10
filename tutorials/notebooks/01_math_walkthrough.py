"""The angular algebra and neighbourhood correlations in MACE."""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tutorials.mace_forward import load_fixture, spherical_harmonics

    jax.config.update("jax_enable_x64", True)
    return jnp, load_fixture, mo, spherical_harmonics


@app.cell
def _(mo):
    mo.md(r"""
    # From directions to many-body energies

    A neighbour direction is a point on the unit sphere. Real spherical
    harmonics turn it into components that transform together under rotation:
    $Y^0=1$ and $Y^1=\sqrt{3}(x,y,z)$. The component normalisation obeys
    $\sum_m Y^\ell_m(\hat r)^2=2\ell+1$. This is why rotating the molecule
    changes individual components while preserving their squared norm.
    """)
    return


@app.cell
def _(jnp, mo, spherical_harmonics):
    directions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 2.0, 3.0]])
    harmonics = spherical_harmonics(directions, 3)
    harmonic_norms = {
        ell: jnp.sum(harmonics[:, ell * ell : (ell + 1) ** 2] ** 2, axis=1).tolist()
        for ell in range(4)
    }
    mo.md(f"Squared norms for three directions: `{harmonic_norms}`")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Coupling two representations

    A tensor product contains several angular momenta. Clebsch-Gordan
    coefficients select a component transforming with angular momentum $L$:
    $[u\otimes v]_{LM}=\sum_{mn} C^{LM}_{\ell m,\ell' n}u_m v_n$.
    For two vectors, the scalar channel is their dot product divided by
    $\sqrt{3}$; the factor normalises the coefficient tensor. A simultaneous
    rotation of both vectors leaves this scalar unchanged.
    """)
    return


@app.cell
def _(jnp, mo):
    left = jnp.array([1.0, 2.0, 3.0])
    right = jnp.array([2.0, -1.0, 4.0])
    scalar_coupling = jnp.dot(left, right) / jnp.sqrt(3.0)
    mo.md(f"Scalar coupling of {left} and {right}: **{float(scalar_coupling):.8f}**")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Messages and correlations

    A radial network supplies one scalar weight for each coupling path and
    channel. The receiver adds messages from its neighbours:
    $A_{icLM}=\sum_{j\in N(i),p}R_{pc}(r_{ij})
    [h_{jc}\otimes Y(\hat r_{ij})]_{pLM}$.
    Summation makes the result independent of neighbour order. The symmetric
    product then couples powers of this density:
    $B^{(\nu)}_{icLM}=\sum_{m_1\ldots m_\nu}
    U^{LM}_{m_1\ldots m_\nu}\prod_{k=1}^{\nu}A_{icm_k}$.
    For $\nu=2$, expanding $(a+b)^2$ exposes $2ab$, a correlation between
    distinct neighbours. Correlation order $\nu$ can describe interactions
    of the centre with up to $\nu$ neighbours in this layer.
    """)
    return


@app.cell
def _(jnp, mo):
    neighbour_messages = jnp.array([1.0, 2.0, -0.5])
    density = jnp.sum(neighbour_messages)
    diagonal = jnp.sum(neighbour_messages**2)
    cross_terms = density**2 - diagonal
    mo.md(
        f"Density: {float(density)}; squared density: {float(density**2)}; "
        f"cross-neighbour contribution: {float(cross_terms)}."
    )
    return


if __name__ == "__main__":
    app.run()
