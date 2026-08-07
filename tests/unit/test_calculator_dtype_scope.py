"""Every calculator entry point must run the model at the model's dtype.

Extensions create tensors mid-forward without an explicit dtype and follow the
process-wide default. When that disagrees with the model, the mismatch survives
the forward through type promotion and only fails in the backward, as a dtype
error out of a linalg op -- which is what #1619 chased down for `calculate`.

An entry point is only safe if its model call sits inside a
`default_dtype(self.default_dtype)` block, so this checks the source rather
than exercising each method: several need optional extras or a model type the
unit suite does not build, and an untested entry point is exactly how the gap
reappeared in get_hessian, get_descriptors and get_dielectric_derivatives after
calculate was fixed.
"""

import ast
import inspect

import pytest

from mace.calculators import mace as mace_calc_module


def _unscoped_forwards(class_name):
    tree = ast.parse(inspect.getsource(mace_calc_module))
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    scopes = [
        (n.lineno, n.end_lineno)
        for n in ast.walk(cls)
        if isinstance(n, ast.With)
        and "default_dtype" in ast.dump(n.items[0].context_expr)
    ]
    unscoped = []
    for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(fn):
            is_forward = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "model"
            )
            if is_forward and not any(a <= node.lineno <= b for a, b in scopes):
                unscoped.append(f"{fn.name} (line {node.lineno})")
    return unscoped


@pytest.mark.parametrize("class_name", ["MACECalculator", "MagneticMACECalculator"])
def test_every_forward_runs_under_the_calculator_dtype(class_name):
    unscoped = _unscoped_forwards(class_name)
    assert not unscoped, (
        f"{class_name} calls the model outside a default_dtype scope in: "
        f"{', '.join(unscoped)}. Wrap the call in "
        "`with torch_tools.default_dtype(self.default_dtype):`"
    )
