"""AST detectors for the shape the v1 stack must have.

The fitness functions in `test_model_shape.py` are always-green asserts about
the target architecture, active from the day the scaffold is empty. That
creates a problem worth naming: a detector with nothing to detect passes, and
so does a detector that is broken. This repository has been bitten by exactly
that shape before -- a benchmark job whose every case skipped, publishing a
0-byte artifact and reporting success.

So the detectors live here as pure functions over source text, and each is
exercised in the test module against a synthetic file built to violate it.
The suite therefore proves the detector works today, while the tree it
actually scans is still empty, and the assertion starts biting the moment the
first model lands without anyone re-enabling anything.

Every check is done over the parsed syntax tree, never over the file's text. A
textual search for `MACEOutput` or `jit` would match docstrings, comments and
this very module.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable, Iterator, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = REPO_ROOT / "packages"

def package_roots() -> List[Path]:
    """The import root of each v1 distribution: `<directory>/src/<import name>`.

    Derived from the tree rather than listed, so a fifth package is scanned
    without an edit here and a renamed one cannot leave a silent hole. The
    `isidentifier` filter is what distinguishes the import root from the
    hyphenated distribution directory above it.
    """
    roots = []
    for source_dir in sorted(PACKAGES.glob("*/src/*")):
        if source_dir.is_dir() and source_dir.name.isidentifier():
            roots.append(source_dir)
    return roots


#: Where a top-level model lives, relative to a package's import root.
#: `docs/reforge/target_layout.md` puts `BaseMACE` in `models/base.py` and the
#: factory in `models/build.py`.
MODEL_SUBDIRECTORY = "models"

#: The typed return of a model's forward. Both spellings are accepted: CORE-1
#: writes `MACEOutput`, the target layout writes `MACEOutputs`, and which one
#: the codebase settles on is CORE-1's decision, not this file's. What this
#: file owns is that a model returns the typed object rather than a dict.
OUTPUT_TYPES = ("MACEOutput", "MACEOutputs")

#: A class is a top-level model if its name ends in one of these, or if it
#: derives from something that does. Blocks and readouts also define `forward`
#: and legitimately return tensors, so "has a forward" is too wide a net.
MODEL_NAME_SUFFIXES = ("MACE",)


def python_files(roots: Sequence[Path]) -> Iterator[Path]:
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            yield path


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _annotation_names(node: ast.AST | None) -> List[str]:
    """Every identifier appearing in an annotation, subscripts included."""
    if node is None:
        return []
    return [child.id for child in ast.walk(node) if isinstance(child, ast.Name)] + [
        child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
    ]


def _base_names(node: ast.ClassDef) -> List[str]:
    names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def is_model_class(node: ast.ClassDef) -> bool:
    candidates = [node.name, *_base_names(node)]
    return any(
        name.endswith(suffix) for name in candidates for suffix in MODEL_NAME_SUFFIXES
    )


def model_classes(source: str, path: str) -> List[ast.ClassDef]:
    tree = ast.parse(source, filename=path)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and is_model_class(node)
    ]


def _method(node: ast.ClassDef, name: str):
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child.name == name:
                return child
    return None


# ---------------------------------------------------------------------------
# The four detectors
#
# Each takes source text and the path it came from, and returns a list of
# sentences. An empty list means the file is clean; the caller never has to
# interpret a truth value.
# ---------------------------------------------------------------------------


def typed_output_violations(source: str, path: str) -> List[str]:
    """A model's `forward` must be annotated as returning the typed output."""
    problems = []
    for node in model_classes(source, path):
        forward = _method(node, "forward")
        if forward is None:
            continue
        names = _annotation_names(forward.returns)
        if not names:
            problems.append(
                f"{path}:{forward.lineno}: {node.name}.forward has no return "
                f"annotation. A model returns one of {list(OUTPUT_TYPES)}, "
                f"never an unannotated dict"
            )
        elif not any(name in OUTPUT_TYPES for name in names):
            problems.append(
                f"{path}:{forward.lineno}: {node.name}.forward returns "
                f"{'/'.join(names)}, not one of {list(OUTPUT_TYPES)}"
            )
    return problems


def config_construction_violations(source: str, path: str) -> List[str]:
    """A model is built from one config object, not from a list of kwargs.

    The legacy shape this rules out is `configure_model`, which reads about a
    hundred `args.*` attributes and passes them positionally into a model
    whose `__init__` therefore has to accept all of them. One typed parameter
    is what makes the construction reviewable and the resolved configuration
    storable in the model's metadata.
    """
    problems = []
    for node in model_classes(source, path):
        init = _method(node, "__init__")
        if init is None:
            continue
        arguments = init.args
        positional = [
            argument
            for argument in [*arguments.posonlyargs, *arguments.args]
            if argument.arg != "self"
        ]
        keyword_only = list(arguments.kwonlyargs)
        parameters = positional + keyword_only
        if len(parameters) != 1:
            problems.append(
                f"{path}:{init.lineno}: {node.name}.__init__ takes "
                f"{len(parameters)} parameters "
                f"({', '.join(argument.arg for argument in parameters) or 'none'}); "
                f"a model is constructed from one config object"
            )
            continue
        names = _annotation_names(parameters[0].annotation)
        if not any(name.endswith("Config") for name in names):
            problems.append(
                f"{path}:{init.lineno}: {node.name}.__init__ takes "
                f"{parameters[0].arg!r} annotated {'/'.join(names) or 'nothing'}; "
                f"the one parameter has to be a *Config"
            )
    return problems


def torch_geometric_violations(source: str, path: str) -> List[str]:
    """No v1 file may import `torch_geometric`, vendored or upstream.

    The legacy `AtomicData` subclasses `torch_geometric.data.Data`, so the
    graph type, the batching and the collation all arrive from a vendored copy
    of a library that is excluded from lint and mypy. v1 owns its own graph
    contract; an import here is the tell that it does not.

    Matched on any component of the dotted path, not just the first, so that
    the vendored spelling `mace.tools.torch_geometric` is named for what it is.
    The import guard rejects it too, for being a legacy import, but a reader
    who sees only that message learns the wrong lesson from it.
    """
    problems = []
    tree = ast.parse(source, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "torch_geometric" in alias.name.split("."):
                    problems.append(f"{path}:{node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "torch_geometric" in module.split("."):
                problems.append(f"{path}:{node.lineno}: imports from {module}")
    return problems


#: TorchScript entry points. `compile_mode` is e3nn's decorator, which marks a
#: module for `e3nn.util.jit.compile`; the rest are torch's own.
TORCHSCRIPT_DECORATORS = ("compile_mode",)
TORCHSCRIPT_CALLS = (
    "script",
    "trace",
    "compile",
    "annotate",
    "unused",
    "ignore",
    "is_scripting",
    "script_if_tracing",
)


def _is_jit_attribute(node: ast.AST) -> bool:
    """True for `jit.<name>` and `torch.jit.<name>` where name is TorchScript."""
    if not isinstance(node, ast.Attribute) or node.attr not in TORCHSCRIPT_CALLS:
        return False
    owner = node.value
    if isinstance(owner, ast.Name):
        return owner.id == "jit"
    if isinstance(owner, ast.Attribute):
        return owner.attr == "jit"
    return False


def torchscript_violations(source: str, path: str) -> List[str]:
    """No `jit.*` and no `@compile_mode` in the live v1 path.

    TorchScript is banned under `packages/` rather than discouraged. It is
    being removed from PyTorch, and it shapes the code it touches: no PEP 604
    unions, `torch.jit.annotate` for empty containers, `is_scripting()`
    branches inside a forward. `torch.compile` is the compiled path, with
    eager as the always-working reference.
    """
    problems = []
    tree = ast.parse(source, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = None
                if isinstance(target, ast.Name):
                    name = target.id
                elif isinstance(target, ast.Attribute):
                    name = target.attr
                if name in TORCHSCRIPT_DECORATORS:
                    problems.append(
                        f"{path}:{node.lineno}: @{name} on {node.name}"
                    )
                elif _is_jit_attribute(target):
                    problems.append(
                        f"{path}:{node.lineno}: @jit.{target.attr} on {node.name}"
                    )
        if _is_jit_attribute(node):
            problems.append(f"{path}:{node.lineno}: jit.{node.attr}")
        if isinstance(node, ast.ImportFrom) and node.module in {
            "torch.jit",
            "e3nn.util.jit",
        }:
            problems.append(f"{path}:{node.lineno}: imports from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"torch.jit", "e3nn.util.jit"}:
                    problems.append(f"{path}:{node.lineno}: imports {alias.name}")
    return sorted(set(problems))


Detector = Callable[[str, str], List[str]]


def scan(detector: Detector, roots: Sequence[Path]) -> Tuple[List[str], int]:
    """Run one detector over every python file under `roots`.

    Returns the problems and the number of files scanned. The count is
    returned, and asserted by the caller, because "no problems" and "no files"
    are the same answer otherwise, and today they are both true.
    """
    problems: List[str] = []
    scanned = 0
    for path in python_files(roots):
        scanned += 1
        problems += detector(path.read_text(encoding="utf-8"), _relative(path))
    return problems, scanned


def model_interface_roots() -> List[Path]:
    """The directories a top-level model may live in, where they exist yet."""
    return [
        root / MODEL_SUBDIRECTORY
        for root in package_roots()
        if (root / MODEL_SUBDIRECTORY).is_dir()
    ]
