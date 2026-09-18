import copy

import pytest
import torch

from mace.modules import TimeReversalSymmetrizedMACE
from mace.modules.extensions import MagneticSCFMACE
from mace.tools.torch_tools import default_dtype

from .test_magmace import (
    _build_small_magnetic_model,
    _make_magnetic_cluster_data,
    _random_rotation,
)

CUDA_AVAILABLE = torch.cuda.is_available()


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _flip(data):
    """Global reversal of every per-atom moment."""
    out = dict(data)
    out["magmom"] = -data["magmom"]
    return out


def _run(model, data, **kw):
    kw.setdefault("training", False)
    kw.setdefault("compute_force", True)
    kw.setdefault("compute_magforces", True)
    return model(dict(data), **kw)


class _TimeReversalOddModel(torch.nn.Module):
    """Base model plus a deliberately T-ODD term, linear in the moments.

    Without this, a test could pass merely because the fixture model is already
    approximately time-reversal symmetric. The added term changes sign under
    M -> -M, so any correct projection must remove it exactly.
    """

    def __init__(self, base, strength=0.25):
        super().__init__()
        self.base = base
        self.strength = strength

    def forward(self, data, **kw):
        out = self.base(data, **kw)
        odd = self.strength * data["magmom"].sum()
        out = dict(out)
        out["energy"] = out["energy"] + odd
        return out


# ----------------------------------------------------------
# The projection itself
# ----------------------------------------------------------
def test_wrapped_energy_equals_explicit_two_evaluation_average():
    """E_TR must be exactly the mean of the base at +M and -M."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        wrapped = TimeReversalSymmetrizedMACE(base).eval()

        e_plus = _run(base, data)["energy"].detach()
        e_minus = _run(base, _flip(data))["energy"].detach()
        e_wrapped = _run(wrapped, data)["energy"].detach()

        assert torch.allclose(e_wrapped, 0.5 * (e_plus + e_minus), atol=1e-10)


def test_energy_is_invariant_under_global_moment_reversal():
    """E(R, M) == E(R, -M), exactly and by construction."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        wrapped = TimeReversalSymmetrizedMACE(base).eval()

        e = _run(wrapped, data)["energy"].detach()
        e_flipped = _run(wrapped, _flip(data))["energy"].detach()
        assert torch.allclose(e, e_flipped, atol=1e-12)


def test_a_time_reversal_odd_contribution_is_removed():
    """The projection must kill a term that is deliberately odd in M.

    Guards against the suite passing only because the fixture happens to be
    nearly symmetric already: the unwrapped model must FAIL the same check.
    """
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        odd = _TimeReversalOddModel(base).double().eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)

        e = _run(odd, data)["energy"].detach()
        e_flipped = _run(odd, _flip(data))["energy"].detach()
        assert not torch.allclose(
            e, e_flipped, atol=1e-6
        ), "the T-odd probe is not actually odd, so this test guards nothing"

        wrapped = TimeReversalSymmetrizedMACE(odd).eval()
        w = _run(wrapped, data)["energy"].detach()
        w_flipped = _run(wrapped, _flip(data))["energy"].detach()
        assert torch.allclose(w, w_flipped, atol=1e-12)


# ----------------------------------------------------------
# Derivative parities
# ----------------------------------------------------------
def test_spatial_forces_are_even_under_moment_reversal():
    """F_R(R, M) == F_R(R, -M)."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)

        f = _run(wrapped, data)["forces"].detach()
        f_flipped = _run(wrapped, _flip(data))["forces"].detach()
        assert torch.allclose(f, f_flipped, atol=1e-10)


def test_magnetic_forces_are_odd_under_moment_reversal():
    """F_M(R, M) == -F_M(R, -M), in this repo's magforces = -dE/dM convention."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)

        mf = _run(wrapped, data)["magforces"].detach()
        mf_flipped = _run(wrapped, _flip(data))["magforces"].detach()
        assert torch.allclose(mf, -mf_flipped, atol=1e-10)
        assert mf.abs().max() > 0, "magforces are identically zero; test proves nothing"


def test_returned_derivatives_agree_with_autograd_of_the_wrapped_energy():
    """The parity combination must equal differentiating E_TR directly.

    This is what justifies combining branch outputs by parity instead of
    intercepting at the energy level.
    """
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        data = dict(data)
        data["positions"] = data["positions"].clone().requires_grad_(True)
        data["magmom"] = data["magmom"].clone().requires_grad_(True)

        out = wrapped(data, training=True, compute_force=True, compute_magforces=True)
        dE_dR, dE_dM = torch.autograd.grad(
            out["energy"].sum(), [data["positions"], data["magmom"]], retain_graph=True
        )
        # repo convention: forces = -dE/dR, magforces = -dE/dM
        assert torch.allclose(out["forces"], -dE_dR, atol=1e-9)
        assert torch.allclose(out["magforces"], -dE_dM, atol=1e-9)


def test_stress_and_virials_are_even_under_moment_reversal():
    """Rank-2 quantities pick up no sign under M -> -M, so they must be averaged."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        kw = dict(compute_stress=True, compute_virials=True)

        out = _run(wrapped, data, **kw)
        flipped = _run(wrapped, _flip(data), **kw)
        for key in ("stress", "virials"):
            assert out[key] is not None, f"{key} was not computed"
            assert torch.allclose(out[key], flipped[key], atol=1e-10), key


def test_energy_gradient_matches_magforces_for_calculator_style_input():
    """The caller does NOT pre-enable requires_grad -- the calculator path.

    Regression test: if the -M branch is built before the moments require grad, it
    becomes an independent leaf and d(E_TR)/dM captures only the +M half. The
    returned magforces stay correct in that case, so this must be checked through
    the ENERGY gradient, not the returned tensor.
    """
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        magmom = data["magmom"]
        assert not magmom.requires_grad, "fixture must start without requires_grad"

        out = wrapped(
            dict(data), training=True, compute_force=True, compute_magforces=True
        )
        (grad,) = torch.autograd.grad(
            out["energy"].sum(), magmom, retain_graph=True, allow_unused=True
        )
        assert grad is not None, "-M branch is disconnected from the caller's moments"
        assert torch.allclose(-grad, out["magforces"], atol=1e-10)


def test_tensor_values_are_preserved_even_though_requires_grad_is_enabled():
    """The wrapper may enable requires_grad (the base does so anyway) but must not
    change any tensor VALUE."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        before = {k: v.detach().clone() for k, v in data.items()}

        wrapped(dict(data), training=False, compute_force=True, compute_magforces=True)

        for k, v in before.items():
            assert torch.equal(data[k].detach(), v), f"wrapper changed values of '{k}'"


# ----------------------------------------------------------
# O(3) behaviour
# ----------------------------------------------------------
@pytest.mark.parametrize("improper", [False, True])
def test_axial_o3_transformation_law(improper):
    """E_TR(QR, det(Q) Q M) == E_TR(R, M) for proper and improper Q."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)

        Q = _random_rotation(seed=1, dtype=torch.float64)
        if improper:
            Q = -Q  # det(-Q) = -det(Q) in 3D, so this flips the determinant
        det = torch.det(Q)

        moved = dict(data)
        moved["positions"] = data["positions"] @ Q.T
        moved["magmom"] = det * (data["magmom"] @ Q.T)
        moved["cell"] = data["cell"] @ Q.T

        e = _run(wrapped, data)["energy"].detach()
        e_moved = _run(wrapped, moved)["energy"].detach()
        assert torch.allclose(e, e_moved, atol=1e-8)


# ----------------------------------------------------------
# Contract: batch, dtypes, devices, attributes, composition
# ----------------------------------------------------------
def test_input_batch_is_not_mutated():
    """The caller's dict and its tensors must come back untouched."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        before = copy.deepcopy({k: v.clone() for k, v in data.items()})
        keys_before = set(data)

        wrapped(data, training=False, compute_force=True, compute_magforces=True)

        assert set(data) == keys_before, "wrapper added or removed batch keys"
        for k, v in before.items():
            assert torch.equal(data[k], v), f"wrapper mutated data['{k}']"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supported_dtypes(dtype):
    with default_dtype(dtype):
        base = _build_small_magnetic_model().to(dtype).eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=dtype)
        out = _run(wrapped, data)
        assert out["energy"].dtype == dtype
        tol = 1e-12 if dtype == torch.float64 else 1e-5
        assert torch.allclose(
            out["energy"].detach(),
            _run(wrapped, _flip(data))["energy"].detach(),
            atol=tol,
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device")
def test_runs_on_cuda():
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().cuda().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = {
            k: v.cuda()
            for k, v in _make_magnetic_cluster_data(dtype=torch.float64).items()
        }
        out = _run(wrapped, data)
        assert out["energy"].is_cuda


@pytest.mark.parametrize(
    "flags",
    [
        {"compute_force": False, "compute_magforces": False},
        {"compute_force": True, "compute_magforces": False},
        {"compute_force": False, "compute_magforces": True},
    ],
)
def test_existing_derivative_flags_are_respected(flags):
    """Turning a derivative off must still turn it off through the wrapper."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base).eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        out = wrapped(dict(data), training=False, **flags)
        assert out["energy"] is not None
        for key, name in (
            ("forces", "compute_force"),
            ("magforces", "compute_magforces"),
        ):
            if not flags[name]:
                assert out[key] is None or torch.count_nonzero(out[key]) == 0


def test_checkpoint_can_be_loaded_then_wrapped():
    """An existing trained checkpoint is wrapped without retraining or param changes."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        before = {k: v.clone() for k, v in base.state_dict().items()}

        wrapped = TimeReversalSymmetrizedMACE(base).eval()

        after = wrapped.model.state_dict()
        assert set(after) == set(before)
        for k, v in before.items():
            assert torch.equal(after[k], v), f"wrapping changed parameter {k}"


def test_wrapper_exposes_base_model_attributes():
    """Calculators and export read attributes straight off the model."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base)
        for attr in ("r_max", "atomic_numbers"):
            assert hasattr(wrapped, attr), f"{attr} not reachable through the wrapper"
            assert torch.equal(
                torch.as_tensor(getattr(wrapped, attr)),
                torch.as_tensor(getattr(base, attr)),
            )


def test_double_wrapping_is_rejected():
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        once = TimeReversalSymmetrizedMACE(base)
        with pytest.raises(ValueError, match="already time-reversal symmetrised"):
            TimeReversalSymmetrizedMACE(once)


def test_wrapping_scf_is_rejected_with_the_correct_order_documented():
    """The projection must act BEFORE the SCF, so wrapping SCF is refused."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        scf = MagneticSCFMACE(base)
        with pytest.raises(ValueError, match="BEFORE the SCF"):
            TimeReversalSymmetrizedMACE(scf)


# ----------------------------------------------------------
# The batch-stacking helper
# ----------------------------------------------------------
def test_stack_time_reversed_builds_two_disconnected_copies():
    """Doubling the batch must keep the two copies as separate graphs."""
    with default_dtype(torch.float64):
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        n_nodes = data["positions"].shape[0]
        n_graphs = data["ptr"].numel() - 1

        out = TimeReversalSymmetrizedMACE.stack_time_reversed(dict(data))

        assert out["positions"].shape[0] == 2 * n_nodes
        assert torch.equal(out["magmom"][:n_nodes], data["magmom"])
        assert torch.equal(out["magmom"][n_nodes:], -data["magmom"])
        # second copy's edges point only at the second copy's nodes
        assert torch.equal(
            out["edge_index"][:, data["edge_index"].shape[1] :],
            data["edge_index"] + n_nodes,
        )
        assert out["batch"].max().item() == 2 * n_graphs - 1
        assert out["ptr"][-1].item() == 2 * n_nodes


def test_stack_time_reversed_does_not_mutate_the_input():
    with default_dtype(torch.float64):
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        before = {k: v.clone() for k, v in data.items()}
        TimeReversalSymmetrizedMACE.stack_time_reversed(dict(data))
        for k, v in before.items():
            assert torch.equal(data[k], v), f"stacking mutated '{k}'"


@pytest.mark.parametrize("batched", [True, False])
def test_both_modes_agree(batched):
    """batched=True and batched=False must be numerically interchangeable."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        kw = dict(
            training=False,
            compute_force=True,
            compute_magforces=True,
            compute_stress=True,
            compute_virials=True,
        )
        a = TimeReversalSymmetrizedMACE(base, batched=batched).eval()(dict(data), **kw)
        b = TimeReversalSymmetrizedMACE(base, batched=not batched).eval()(
            dict(data), **kw
        )
        for key in ("energy", "forces", "magforces", "stress", "virials"):
            assert a[key] is not None and b[key] is not None, key
            assert torch.allclose(a[key], b[key], atol=1e-10), key


def test_unstack_inverts_stack_for_an_even_and_an_odd_quantity():
    """unstack_time_reversed must average even outputs and antisymmetrise odd ones."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapper = TimeReversalSymmetrizedMACE(base)
        n_nodes, n_graphs = 4, 2
        plus = torch.arange(n_nodes * 3, dtype=torch.float64).reshape(n_nodes, 3)
        minus = plus * -3.0
        stacked = {
            "forces": torch.cat([plus, minus]),  # even -> averaged
            "magforces": torch.cat([plus, minus]),  # odd  -> antisymmetrised
            "scf_steps": 7,  # not a tensor -> passed through
        }
        out = wrapper.unstack_time_reversed(stacked, n_nodes, n_graphs, n_edges=5)
        assert torch.allclose(out["forces"], 0.5 * (plus + minus))
        assert torch.allclose(out["magforces"], 0.5 * (plus - minus))
        assert out["scf_steps"] == 7


def test_unstack_passes_through_tensors_that_were_not_doubled():
    """A tensor whose leading dim is not a doubled count must be left alone."""
    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapper = TimeReversalSymmetrizedMACE(base)
        odd_shaped = torch.randn(7, 3, dtype=torch.float64)
        out = wrapper.unstack_time_reversed({"node_feats": odd_shaped}, 4, 2, 5)
        assert torch.equal(out["node_feats"], odd_shaped)


def test_multi_structure_batch_of_differing_sizes():
    """Several graphs of DIFFERENT sizes, so the ptr/batch/edge_index offsets matter.

    The single-structure fixture cannot catch an offset bug in stack_time_reversed.
    """
    import numpy as np
    from ase.atoms import Atoms

    from mace import data
    from mace.tools import torch_geometric, utils

    with default_dtype(torch.float64):
        rng = np.random.default_rng(0)
        configs = []
        for n in (2, 3, 2):
            atoms = Atoms(
                numbers=[26] * n,
                positions=rng.normal(0, 1.5, (n, 3)) + np.arange(n)[:, None] * 1.7,
                cell=np.eye(3) * 9.0,
                pbc=[True] * 3,
            )
            atoms.info["REF_energy"] = float(rng.normal())
            atoms.new_array("REF_forces", rng.normal(0, 0.3, (n, 3)))
            atoms.new_array("REF_magmom", rng.normal(0, 0.5, (n, 3)))
            configs.append(atoms)

        keyspec = data.KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={"forces": "REF_forces", "magmom": "REF_magmom"},
        )
        z_table = utils.AtomicNumberTable([26])
        dataset = [
            data.AtomicData.from_config(
                data.config_from_atoms(c, key_specification=keyspec),
                z_table=z_table,
                cutoff=5.0,
            )
            for c in configs
        ]
        batch = next(
            iter(torch_geometric.dataloader.DataLoader(dataset, batch_size=3))
        ).to_dict()

        def fresh():
            return {
                k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()
            }

        base = _build_small_magnetic_model().double().eval()
        kw = dict(training=False, compute_force=True, compute_magforces=True)
        out_b = TimeReversalSymmetrizedMACE(base, batched=True).eval()(fresh(), **kw)
        out_s = TimeReversalSymmetrizedMACE(base, batched=False).eval()(fresh(), **kw)

        assert out_b["energy"].shape[0] == 3, "per-graph energies were collapsed"
        for key in ("energy", "forces", "magforces"):
            assert torch.allclose(out_b[key], out_s[key], atol=1e-10), key

        flipped = fresh()
        flipped["magmom"] = -flipped["magmom"]
        e = TimeReversalSymmetrizedMACE(base, batched=True).eval()(fresh(), **kw)[
            "energy"
        ]
        e_flip = TimeReversalSymmetrizedMACE(base, batched=True).eval()(flipped, **kw)[
            "energy"
        ]
        assert torch.allclose(e.detach(), e_flip.detach(), atol=1e-12)


def test_stacking_passes_scalar_tensors_through():
    """0-dim tensors are graph-independent and cannot be concatenated on dim 0."""
    with default_dtype(torch.float64):
        data = _make_magnetic_cluster_data(dtype=torch.float64)
        data["a_scalar"] = torch.tensor(3.0, dtype=torch.float64)
        out = TimeReversalSymmetrizedMACE.stack_time_reversed(data)
        assert out["a_scalar"].ndim == 0
        assert out["a_scalar"].item() == 3.0


def test_config_extraction_targets_the_unwrapped_model():
    """Tooling that dispatches on class name must be pointed at wrapped.model.

    Documents the known integration boundary rather than leaving it to be
    discovered at runtime.
    """
    from mace.tools.scripts_utils import extract_config_mace_model

    with default_dtype(torch.float64):
        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base)

        # It dispatches on the class NAME, so it does not see through the wrapper. It
        # returns an error dict rather than raising, which is easy to miss downstream.
        assert "error" in extract_config_mace_model(wrapped)

        # Supported route: extract from the unwrapped model, then re-apply the wrapper.
        config = extract_config_mace_model(wrapped.model)
        assert "error" not in config
        assert wrapped.model is base


def test_edge_forces_and_hessian_keep_their_undoubled_shapes():
    """Edge forces are per-EDGE and the Hessian is doubled in two dimensions.

    Neither is a per-node or per-graph quantity, so a leading-dimension table keyed
    only on nodes/graphs mis-handles both. Uses a system where n_edges != n_nodes so
    the edge case cannot pass by coincidence.
    """
    import numpy as np
    from ase.atoms import Atoms

    from mace import data
    from mace.tools import torch_geometric, utils

    with default_dtype(torch.float64):
        rng = np.random.default_rng(1)
        n = 4
        atoms = Atoms(
            numbers=[26] * n,
            positions=np.array([[0, 0, 0], [1.7, 0, 0], [0, 1.7, 0], [0, 0, 1.7]]),
            cell=np.eye(3) * 9.0,
            pbc=[True] * 3,
        )
        atoms.info["REF_energy"] = 0.0
        atoms.new_array("REF_forces", rng.normal(0, 0.3, (n, 3)))
        atoms.new_array("REF_magmom", rng.normal(0, 0.5, (n, 3)))
        keyspec = data.KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={"forces": "REF_forces", "magmom": "REF_magmom"},
        )
        z_table = utils.AtomicNumberTable([26])
        dataset = [
            data.AtomicData.from_config(
                data.config_from_atoms(atoms, key_specification=keyspec),
                z_table=z_table,
                cutoff=5.0,
            )
        ]
        batch = next(
            iter(torch_geometric.dataloader.DataLoader(dataset, batch_size=1))
        ).to_dict()
        n_nodes = batch["positions"].shape[0]
        n_edges = batch["edge_index"].shape[1]
        assert (
            n_edges != n_nodes
        ), "fixture must have n_edges != n_nodes to be meaningful"

        def fresh():
            return {
                k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()
            }

        base = _build_small_magnetic_model().double().eval()
        kw = dict(
            training=True,
            compute_force=True,
            compute_magforces=False,
            compute_edge_forces=True,
            compute_hessian=True,
        )
        ref = base(fresh(), **kw)
        got = TimeReversalSymmetrizedMACE(base, batched=True).eval()(fresh(), **kw)

        assert got["edge_forces"].shape == ref["edge_forces"].shape
        assert got["hessian"].shape == ref["hessian"].shape

        # and they must equal the explicit two-evaluation combination
        def at(moments):
            d = fresh()
            d["magmom"] = moments
            return base(d, **kw)

        plus, minus = at(batch["magmom"]), at(-batch["magmom"])
        assert torch.allclose(
            got["edge_forces"],
            0.5 * (plus["edge_forces"] + minus["edge_forces"]),
            atol=1e-10,
        )
        assert torch.allclose(
            got["hessian"], 0.5 * (plus["hessian"] + minus["hessian"]), atol=1e-10
        )


def test_attribute_lookup_does_not_recurse_before_the_model_is_set():
    """__getattr__ can fire during unpickling, before _modules exists.

    Reading _modules out of __dict__ (rather than touching self.model) turns what
    would be a RecursionError into a clean AttributeError.
    """
    bare = TimeReversalSymmetrizedMACE.__new__(TimeReversalSymmetrizedMACE)
    with pytest.raises(AttributeError):
        getattr(bare, "r_max")


def test_wrapped_model_survives_a_pickle_round_trip():
    with default_dtype(torch.float64):
        import pickle

        base = _build_small_magnetic_model().double().eval()
        wrapped = TimeReversalSymmetrizedMACE(base)
        restored = pickle.loads(pickle.dumps(wrapped))
        assert float(restored.r_max) == float(wrapped.r_max)
        assert restored.batched == wrapped.batched
