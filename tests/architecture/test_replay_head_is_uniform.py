"""No v1 source treats the replay head as a special case.

A fine-tune trains the new head beside a head that replays the foundation's
own data. In v1 both are ordinary heads, read and trained by the same code, and
the only thing telling them apart is their configuration. Legacy tells them
apart by name: it compares head names against ``"pt_head"``, looks the head up
by that name and tests for membership, in several places across the training
driver and its tools. Each such branch is a place where the replay head can
behave differently from the one described by its configuration.

The check runs over the parsed syntax tree. It flags a head-name literal used as
a comparison operand, a subscript key, a lookup argument or a ``match`` value,
and an ``isinstance`` against a replay or pretraining type. Mentioning the name
in a docstring or a message is fine; branching on it is not. The same checker
run over the frozen tree has to find the legacy branches, or it proves nothing.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = REPO_ROOT / "packages"
LEGACY = REPO_ROOT / "mace"

REPLAY_HEAD_NAMES = frozenset({"pt_head", "pretraining", "replay"})
REPLAY_TYPE_MARKERS = ("replay", "pretrain", "pthead")
LOOKUP_METHODS = frozenset({"index", "get", "pop", "remove", "count"})


def _is_head_name(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in REPLAY_HEAD_NAMES
    )


def _names_a_replay_type(node: ast.AST) -> bool:
    for child in ast.walk(node):
        name = (
            child.id
            if isinstance(child, ast.Name)
            else child.attr
            if isinstance(child, ast.Attribute)
            else None
        )
        if name is not None and any(
            marker in name.lower().replace("_", "") for marker in REPLAY_TYPE_MARKERS
        ):
            return True
    return False


def replay_branches(source: str) -> list[int]:
    """The line of every place the code branches on the replay head's name or
    type, in the order they appear."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare):
            if any(_is_head_name(side) for side in [node.left, *node.comparators]):
                lines.append(node.lineno)
        elif isinstance(node, ast.Subscript):
            if _is_head_name(node.slice):
                lines.append(node.lineno)
        elif isinstance(node, ast.MatchValue):
            if _is_head_name(node.value):
                lines.append(node.value.lineno)
        elif isinstance(node, ast.Call):
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr in LOOKUP_METHODS
                and any(_is_head_name(argument) for argument in node.args)
            ):
                lines.append(node.lineno)
            elif (
                isinstance(function, ast.Name)
                and function.id in {"isinstance", "issubclass"}
                and len(node.args) == 2
                and _names_a_replay_type(node.args[1])
            ):
                lines.append(node.lineno)
    return sorted(lines)


def _sources(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
    )


V1_SOURCES = [path for path in _sources(PACKAGES) if "src" in path.parts]


@pytest.mark.parametrize(
    "snippet",
    [
        'if head == "pt_head": pass',
        'if "pt_head" in heads: pass',
        'if head != "pt_head": pass',
        'weights = config["pt_head"]',
        'position = heads.index("pt_head")',
        'config = heads_config.get("pt_head")',
        'match head:\n    case "pt_head": pass',
        "if isinstance(head, ReplayHead): pass",
        "if isinstance(head, (heads.PretrainingHead, Other)): pass",
    ],
)
def test_the_checker_catches_a_branch_on_the_replay_head(snippet):
    assert replay_branches(snippet), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        '"""The pt_head is the replay head."""',
        'raise ValueError("pt_head needs a train file")',
        'help_text = "weight of the pt_head"',
        "if isinstance(head, HeadConfig): pass",
    ],
)
def test_the_checker_leaves_a_mention_alone(snippet):
    assert not replay_branches(snippet), snippet


def test_the_checker_finds_the_legacy_branches():
    """The ticket counts at least four in legacy. A checker finding fewer
    there would pass on v1 for the wrong reason."""
    found = {
        path.relative_to(REPO_ROOT).as_posix(): replay_branches(path.read_text())
        for path in _sources(LEGACY)
    }
    found = {path: lines for path, lines in found.items() if lines}
    assert sum(len(lines) for lines in found.values()) >= 4, found


def test_there_are_v1_sources_to_check():
    assert len(V1_SOURCES) > 50


@pytest.mark.parametrize(
    "path", V1_SOURCES, ids=[str(path.relative_to(PACKAGES)) for path in V1_SOURCES]
)
def test_no_v1_source_branches_on_the_replay_head(path):
    lines = replay_branches(path.read_text())
    assert not lines, (
        f"{path.relative_to(REPO_ROOT)} branches on the replay head at lines "
        f"{lines}. The replay head is an ordinary head: describe what differs "
        f"in its configuration and read that instead of its name."
    )
