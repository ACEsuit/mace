"""The fitness functions: always-green asserts about the shape of the v1 stack.

These are invariants, not milestones. Each states something the target
architecture must be true of, and each is active from day one, on a scaffold
that contains no model at all. Two of them are described in the plan as
"activating" when their axis lands, and the word is misleading: nothing here
is switched on later. The detector runs now, finds nothing now, and starts
finding things the moment the code it describes exists.

WHY THE DETECTORS ARE SELF-TESTED. A check with nothing to check passes, and
so does a broken one. The two are indistinguishable while `packages/` is
nearly empty, which is the whole period these tests are supposed to be
guarding. So every detector is also run against a synthetic file written to
violate it, and against one written to satisfy it. That is what makes a green
run here mean "the invariant holds" rather than "there was nothing to look
at".

The detectors themselves are in `v1_surface.py`.
"""

from __future__ import annotations

import pytest

from tests.architecture import v1_surface

# ---------------------------------------------------------------------------
# The invariants
# ---------------------------------------------------------------------------


def test_the_scan_reaches_the_v1_packages():
    """The precondition every test below depends on.

    Each of the four asserts "no violations found". That sentence is also true
    of a scan that opened no files, so the count is asserted first and once,
    here, rather than repeated in each of them.
    """
    roots = v1_surface.package_roots()
    assert [root.name for root in roots] == [
        "mace_core",
        "mace_jax",
        "mace_launcher",
        "mace_torch",
    ], (
        f"the v1 import roots found under packages/ are "
        f"{[root.name for root in roots]}. The detectors below scan whatever "
        f"this returns, so a package missing here is a package nothing checks."
    )
    _, scanned = v1_surface.scan(v1_surface.torchscript_violations, roots)
    assert scanned > 0, "no python files under packages/: the scan found nothing to read"


def test_forward_returns_maceoutputs():
    """Every v1 model's forward returns the typed output object.

    The legacy contract is a dict assembled by `get_outputs`, whose keys
    depend on which flags were passed and whose absence is discovered by a
    `KeyError` in a caller. v1 returns a dataclass, so a missing observable is
    a typed absence and a new one is a field rather than a convention.
    """
    problems, _ = v1_surface.scan(
        v1_surface.typed_output_violations, v1_surface.package_roots()
    )
    assert not problems, "\n".join(problems)


def test_construction_via_modelconfig():
    """Models are built from one config object, not from ad-hoc kwargs.

    `configure_model` reads about a hundred attributes off an argparse
    namespace and hands them to a constructor that has to accept all of them.
    One typed parameter is what lets the fully resolved configuration be
    validated once and stored in the model's own metadata.
    """
    problems, _ = v1_surface.scan(
        v1_surface.config_construction_violations, v1_surface.package_roots()
    )
    assert not problems, "\n".join(problems)


def test_no_torch_geometric_in_model_interface():
    """Zero `torch_geometric` anywhere in the v1 stack.

    Stated over the whole stack rather than over the model interface alone,
    because the interface is where the import would *show*, and the data layer
    is where it would come from. The vendored copy under `mace/tools/` is
    excluded from lint and from mypy; v1 owns its graph contract instead.
    """
    problems, _ = v1_surface.scan(
        v1_surface.torch_geometric_violations, v1_surface.package_roots()
    )
    assert not problems, "\n".join(problems)


def test_no_jit_in_live_v1_path():
    """No `jit.*` and no `@compile_mode` under `packages/`.

    The legacy tree has 52 `@compile_mode("script")` decorators across nine
    modules and they are staying there, frozen, for the length of the
    migration; that is a debt row, not a violation of this test. What this
    test owns is that none of them is reproduced in new code.
    """
    problems, _ = v1_surface.scan(
        v1_surface.torchscript_violations, v1_surface.package_roots()
    )
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# The detectors, tested against source they should reject
#
# Written as source text rather than as files on disk: what is being tested is
# the detector, and giving it a path it can quote is the only thing it wants
# from the filesystem.
# ---------------------------------------------------------------------------

CLEAN_MODEL = '''
"""A model the way the target architecture has it."""

from mace_core.config import ModelConfig
from mace_core.types import MACEOutput


class ExampleMACE(Module):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def forward(self, graph: GraphView) -> MACEOutput:
        return MACEOutput(energy=None)
'''


def test_the_typed_output_detector_accepts_the_target_shape():
    assert not v1_surface.typed_output_violations(CLEAN_MODEL, "clean.py")


@pytest.mark.parametrize(
    "forward, expected",
    [
        ("    def forward(self, graph):\n        return {}\n", "no return annotation"),
        (
            "    def forward(self, graph) -> Dict[str, Tensor]:\n        return {}\n",
            "not one of",
        ),
    ],
    ids=["unannotated", "returns-a-dict"],
)
def test_the_typed_output_detector_rejects_an_untyped_forward(forward, expected):
    source = f"class ExampleMACE(Module):\n{forward}"
    problems = v1_surface.typed_output_violations(source, "offender.py")
    assert len(problems) == 1, problems
    assert expected in problems[0]
    assert "ExampleMACE" in problems[0]


def test_the_typed_output_detector_ignores_a_block():
    """A readout returning a tensor is correct, and must not be flagged.

    This is the reason the detector keys on the class name and its bases
    rather than on "defines a forward": every block in the tree defines one,
    and almost none of them returns a `MACEOutput`.
    """
    source = (
        "class ReadoutBlock(Module):\n"
        "    def forward(self, features) -> Tensor:\n"
        "        return features\n"
    )
    assert not v1_surface.typed_output_violations(source, "block.py")


def test_the_typed_output_detector_follows_the_base_class():
    """A subclass of a model is a model, whatever it is called."""
    source = (
        "class ElectrostaticHead(BaseMACE):\n"
        "    def forward(self, graph) -> dict:\n"
        "        return {}\n"
    )
    problems = v1_surface.typed_output_violations(source, "offender.py")
    assert len(problems) == 1, problems


def test_the_config_detector_accepts_the_target_shape():
    assert not v1_surface.config_construction_violations(CLEAN_MODEL, "clean.py")


@pytest.mark.parametrize(
    "init, expected",
    [
        (
            "    def __init__(self, r_max: float, num_interactions: int) -> None:\n"
            "        pass\n",
            "takes 2 parameters",
        ),
        ("    def __init__(self) -> None:\n        pass\n", "takes 0 parameters"),
        (
            "    def __init__(self, config: dict) -> None:\n        pass\n",
            "has to be a *Config",
        ),
        (
            "    def __init__(self, config: ModelConfig, *, backend: str) -> None:\n"
            "        pass\n",
            "takes 2 parameters",
        ),
    ],
    ids=["kwargs", "no-config", "untyped-config", "keyword-only-extra"],
)
def test_the_config_detector_rejects_ad_hoc_construction(init, expected):
    source = f"class ExampleMACE(Module):\n{init}"
    problems = v1_surface.config_construction_violations(source, "offender.py")
    assert len(problems) == 1, problems
    assert expected in problems[0]


@pytest.mark.parametrize(
    "source",
    [
        "import torch_geometric\n",
        "from torch_geometric.data import Data\n",
        "import torch_geometric.data as tg\n",
        "from mace.tools.torch_geometric import Batch\n",
    ],
    ids=["import", "from-import", "aliased", "the-vendored-copy"],
)
def test_the_torch_geometric_detector_rejects_every_spelling(source):
    problems = v1_surface.torch_geometric_violations(source, "offender.py")
    assert problems, source


def test_the_torch_geometric_detector_accepts_torch_itself():
    source = "import torch\nfrom torch import Tensor\n"
    assert not v1_surface.torch_geometric_violations(source, "clean.py")


@pytest.mark.parametrize(
    "source",
    [
        "from e3nn.util.jit import compile_mode\n\n\n@compile_mode('script')\nclass A:\n    pass\n",
        "import torch\n\nm = torch.jit.script(model)\n",
        "from e3nn.util import jit\n\nm = jit.compile(model)\n",
        "import torch\n\n\n@torch.jit.unused\ndef helper():\n    pass\n",
        "import torch\n\nx = torch.jit.annotate(List[int], [])\n",
        "import torch\n\nif torch.jit.is_scripting():\n    pass\n",
    ],
    ids=[
        "compile-mode-decorator",
        "torch-jit-script",
        "e3nn-jit-compile",
        "jit-unused-decorator",
        "jit-annotate",
        "is-scripting-branch",
    ],
)
def test_the_torchscript_detector_rejects_every_spelling(source):
    problems = v1_surface.torchscript_violations(source, "offender.py")
    assert problems, source


def test_the_torchscript_detector_accepts_torch_compile():
    """`torch.compile` is the compiled path v1 uses, and must not be flagged."""
    source = "import torch\n\ncompiled = torch.compile(model)\n"
    assert not v1_surface.torchscript_violations(source, "clean.py")


def test_the_torchscript_detector_finds_the_legacy_tree_it_describes():
    """The detector against the code the debt rows are about.

    The strongest available evidence that these detectors work is the frozen
    tree, which is full of what they look for: if a scan of `mace/modules/`
    came back clean, the detector would be broken rather than the tree
    surprising. It doubles as the count the `DEBT-JIT-MODULES` row quotes.
    """
    roots = [v1_surface.REPO_ROOT / "mace" / "modules"]
    problems, scanned = v1_surface.scan(v1_surface.torchscript_violations, roots)
    assert scanned > 10
    assert len(problems) > 50, (
        f"only {len(problems)} TorchScript uses found in mace/modules/, which "
        f"has 52 @compile_mode decorators alone: the detector is broken"
    )
