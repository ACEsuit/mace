"""What each surface can emit, read out of ``mace/`` rather than remembered.

The golden schema has to know every name an evaluation can produce, and the
only durable way to know that is to derive it. Two rounds of defects here had
one shape: a hand-written table that looked complete, silently missed
something, and reported coverage. So the tables are gone and this module
parses the package instead -- once per surface, with the same failure rule
each time.

That rule is the important part. **An extractor that finds nothing reports
perfect coverage**, and so does one that finds nine tenths. Every scan
therefore reports two things: the keys it resolved, and the writes it could
*not* resolve, with file and line. A guard test asserts the second list is
empty, so a write this module cannot follow fails loudly instead of quietly
shrinking the surface it claims to describe.

Three surfaces:

``scan_calculator_surface``
    what an ase ``Calculator`` subclass in this package can put in
    ``self.results``. Not a regex over one spelling: it follows the aliased
    local, ``.update()``, the dict literal, and the suffixed keys, and it
    derives the committee suffix bases from the very set literal the source
    guards them with.

``scan_model_surface``
    what a ``forward`` can return, from every file that defines one -- the
    file list is discovered, because the version that named
    ``models.py``/``extensions.py`` was reading two writers out of four.

``scan_eval_surface``
    what the evaluation CLI writes onto its structures, with the store
    (``info`` or ``arrays``) it writes each one into, since that is what
    decides whether a name is a graph quantity or a per-atom one.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "mace"
EVAL_CLI = PACKAGE_ROOT / "cli" / "eval_configs.py"

#: Vendored, excluded from lint and mypy by the project, and not ours to
#: reason about.
_VENDORED = ("torch_geometric",)


@dataclass
class Scan:
    """One surface's derived key set.

    ``keys`` is grouped by owner (``path::Class``) so a coverage failure can
    say which family diverged, and ``unresolved`` is the honesty check: a
    write whose key this module could not compute, so that "found nothing"
    can never look like "nothing to find".
    """

    keys: Dict[str, Set[str]] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)

    @property
    def all_keys(self) -> Set[str]:
        out: Set[str] = set()
        for keys in self.keys.values():
            out |= keys
        return out

    def owners_of(self, key: str) -> List[str]:
        return sorted(owner for owner, keys in self.keys.items() if key in keys)


#: Writes whose key genuinely cannot be computed from the source, each with
#: the reason. Matched on the code text rather than a line number, so an edit
#: to the statement reopens the question instead of the allowlist silently
#: covering something else. This is the same discipline as the schema's
#: ``ignore_key``: an unresolved write is allowed only one at a time and only
#: in writing, because the alternative is a scan that shrinks quietly.
PASSTHROUGH_WRITES: Tuple[Tuple[str, str, str], ...] = (
    (
        "mace/calculators/mace_torchsim.py",
        "results[key] = v.clone() if self._use_cudagraphs else v",
        "the torchsim wrapper forwards whatever the wrapped model returned, "
        "key by key (mace/calculators/mace_torchsim.py:706-717), so its key "
        "set *is* the model surface's and is covered there. No static "
        "analysis can name the keys, because they come from a model chosen at "
        "runtime.",
    ),
)


def unexplained(scan: "Scan") -> List[str]:
    """Unresolved writes with no entry in :data:`PASSTHROUGH_WRITES`."""
    out = []
    for entry in scan.unresolved:
        if not any(
            path in entry and code in entry for path, code, _ in PASSTHROUGH_WRITES
        ):
            out.append(entry)
    return out


def _display(path: Path) -> str:
    """A repo-relative path, or the plain one for a file outside the tree.

    The scans are pointed at synthetic sources by their own tests, so this
    cannot assume every file it is handed lives under the repository.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def python_sources(root: Path = PACKAGE_ROOT) -> List[Path]:
    """Every non-vendored module in the package, in a stable order."""
    return sorted(
        path
        for path in root.rglob("*.py")
        if not any(part in _VENDORED for part in path.parts)
    )


# ---------------------------------------------------------------------------
# Literal collection
#
# Three kinds of literal are load-bearing when following a dict write:
#
#   * a set/list/tuple of strings, because a suffixed write is guarded by
#     membership in one (`if results_key in results_store_ensemble`), and that
#     set is the only statement of which keys get a `_comm` and a `_var`. The
#     previous guard restated its four members by hand next to a comment
#     saying they came from there;
#   * a list of tuples, because the write inside `for results_key, ret_key,
#     unit_conv in results_map:` takes its key from the first column;
#   * a dict, because `.update(some_dict)` names one.
#
# Collected per module and merged across every assignment to a name, plus
# `.extend`/`.append`, so a list built in two statements (as `results_map` is,
# for the polar case) is not read as only its first half. Merging across
# scopes over-approximates -- two functions may use the same variable name for
# different sets -- and that direction is the safe one for a guard: it can
# only demand a registration that turns out to be unnecessary, never miss one.
# ---------------------------------------------------------------------------


def _string_members(node: ast.AST) -> Set[str]:
    """The string constants of a set/list/tuple literal, or ``set([...])``."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in ("set", "frozenset", "list", "tuple") and node.args:
            return _string_members(node.args[0])
        return set()
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return {
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
    return set()


def _first_column(node: ast.AST) -> Set[str]:
    """The first element of every tuple in a list/tuple of tuples."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return set()
    out = set()
    for element in node.elts:
        if isinstance(element, (ast.Tuple, ast.List)) and element.elts:
            head = element.elts[0]
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                out.add(head.value)
    return out


def _dict_keys(node: ast.AST) -> Set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


@dataclass
class _Literals:
    sets: Dict[str, Set[str]] = field(default_factory=dict)
    columns: Dict[str, Set[str]] = field(default_factory=dict)
    dicts: Dict[str, Set[str]] = field(default_factory=dict)


def _collect_literals(tree: ast.AST) -> _Literals:
    found = _Literals()

    def record(name: str, value: ast.AST) -> None:
        for store, extract in (
            (found.sets, _string_members),
            (found.columns, _first_column),
            (found.dicts, _dict_keys),
        ):
            members = extract(value)
            if members:
                store.setdefault(name, set()).update(members)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and node.value is not None:
                    record(target.id, node.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.args
        ):
            name = node.func.value.id
            if node.func.attr == "extend":
                record(name, node.args[0])
            elif node.func.attr == "append":
                record(name, ast.List(elts=[node.args[0]], ctx=ast.Load()))
    return found


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


def _resolve(node: ast.AST, bindings: Dict[str, Set[str]]) -> Set[str]:
    """The strings a subscript key expression can take, or an empty set.

    Empty means "this module cannot tell", and the caller records it as
    unresolved rather than moving on.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        return set(bindings.get(node.id, ()))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolve(node.left, bindings)
        right = _resolve(node.right, bindings)
        if left and right:
            return {a + b for a in left for b in right}
    return set()


def _membership_bindings(
    test: ast.AST, literals: _Literals
) -> Dict[str, Set[str]]:
    """``x in SOME_SET`` constraints an ``if`` puts on its body.

    This is how a suffixed write gets its bases without anybody restating
    them: the source says ``if ... and results_key in results_store_ensemble``
    and the set literal three lines up says what that means.
    """
    found: Dict[str, Set[str]] = {}
    stack = [test]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.BoolOp):
            stack.extend(node.values)
            continue
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and isinstance(node.left, ast.Name)
        ):
            members = _string_members(node.comparators[0]) or literals.sets.get(
                getattr(node.comparators[0], "id", ""), set()
            )
            if members:
                found[node.left.id] = set(members)
    return found


# ---------------------------------------------------------------------------
# The write walker, shared by the calculator and model surfaces
# ---------------------------------------------------------------------------


class _WriteWalker:
    """Collect every key written into a tracked dict.

    ``is_target`` decides what "the dict" is, which is the only thing that
    differs between a calculator's ``self.results`` and a forward's return
    value. Everything else -- the aliased local, ``.update()``, the dict
    literal, the loop variable, the suffix -- is the same machinery, and every
    one of those forms exists in this package today or is one refactor away.
    """

    def __init__(
        self,
        path: Path,
        source: str,
        literals: _Literals,
        is_target,
        enter_functions: bool = True,
    ):
        self.path = path
        self.lines = source.splitlines()
        self.literals = literals
        self.is_target = is_target
        #: A calculator's ``self.results`` is an attribute, so a helper
        #: defined inside a method can write to it and the walk has to follow
        #: -- descending is the safe direction there. A forward's return dict
        #: is a local, and a nested closure's locals are a different function
        #: with its own names: the SCF model's LBFGS ``closure`` returns a
        #: tensor called ``energy``, and following it would read that
        #: function's variables as the enclosing forward's outputs.
        self.enter_functions = enter_functions
        self.keys: Set[str] = set()
        self.unresolved: List[str] = []

    def _complain(self, node: ast.AST) -> None:
        line = self.lines[node.lineno - 1].strip() if node.lineno else "?"
        self.unresolved.append(
            f"{_display(self.path)}:{node.lineno}: {line}"
        )

    def _record(self, node: ast.AST, slice_node: ast.AST, bindings) -> None:
        resolved = _resolve(slice_node, bindings)
        if resolved:
            self.keys |= resolved
        else:
            self._complain(node)

    def walk(self, body: Iterable[ast.stmt], bindings: Dict[str, Set[str]]) -> None:
        for node in body:
            self.statement(node, bindings)

    def statement(self, node: ast.stmt, bindings: Dict[str, Set[str]]) -> None:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not self.enter_functions
        ):
            return
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            for target in targets:
                if isinstance(target, ast.Subscript) and self.is_target(target.value):
                    self._record(node, target.slice, bindings)
                elif self.is_target(target) and isinstance(node.value, ast.Dict):
                    self.keys |= _dict_keys(node.value)
            return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
                and self.is_target(call.func.value)
            ):
                self._update_call(node, call)
            return
        if isinstance(node, ast.For):
            inner = dict(bindings)
            iterable = node.iter
            columns = (
                self.literals.columns.get(iterable.id, set())
                if isinstance(iterable, ast.Name)
                else _first_column(iterable)
            )
            if isinstance(node.target, ast.Tuple) and node.target.elts and columns:
                head = node.target.elts[0]
                if isinstance(head, ast.Name):
                    inner[head.id] = set(columns)
            self.walk(node.body, inner)
            self.walk(node.orelse, bindings)
            return
        if isinstance(node, ast.If):
            self.walk(
                node.body,
                {**bindings, **_membership_bindings(node.test, self.literals)},
            )
            self.walk(node.orelse, bindings)
            return
        for name in ("body", "orelse", "finalbody"):
            block = getattr(node, name, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                self.walk(block, bindings)
        for handler in getattr(node, "handlers", []) or []:
            self.walk(handler.body, bindings)

    def _update_call(self, node: ast.stmt, call: ast.Call) -> None:
        if call.keywords and not call.args:
            # `.update(key=value)` cannot express these names (they are not
            # identifiers in general) and nothing here uses it, but silence
            # would be the wrong default.
            self._complain(node)
            return
        for arg in call.args:
            keys = _dict_keys(arg)
            if not keys and isinstance(arg, ast.Name):
                keys = self.literals.dicts.get(arg.id, set())
            if keys:
                self.keys |= keys
            else:
                self._complain(node)


# ---------------------------------------------------------------------------
# The calculator surface
# ---------------------------------------------------------------------------


def _is_self_results(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "results"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _results_aliases(tree: ast.AST) -> Set[str]:
    """Locals that are the results dict under another name, both directions.

    ``results = self.results`` and ``results = {}; ...; self.results =
    results`` are the same write with the statements in a different order, and
    a scan that only understood the attribute spelling would report full
    coverage of a method that writes every one of its keys through the local.
    """
    aliases: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if isinstance(target, ast.Name) and _is_self_results(value):
            aliases.add(target.id)
        if _is_self_results(target) and isinstance(value, ast.Name):
            aliases.add(value.id)
    return aliases


def _calculator_classes(tree: ast.AST) -> List[ast.ClassDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(base, ast.Name) and base.id.endswith("Calculator"))
            or (isinstance(base, ast.Attribute) and base.attr.endswith("Calculator"))
            for base in node.bases
        )
    ]


def scan_calculator_surface(paths: Optional[Iterable[Path]] = None) -> Scan:
    """Every key an ase calculator in this package can put in ``self.results``."""
    scan = Scan()
    for path in paths if paths is not None else python_sources():
        source = path.read_text(encoding="utf-8")
        if "Calculator" not in source:
            continue
        tree = ast.parse(source)
        classes = _calculator_classes(tree)
        if not classes:
            continue
        literals = _collect_literals(tree)
        aliases = _results_aliases(tree)

        def is_target(node: ast.AST, _aliases=aliases) -> bool:
            return _is_self_results(node) or (
                isinstance(node, ast.Name) and node.id in _aliases
            )

        for klass in classes:
            walker = _WriteWalker(path, source, literals, is_target)
            walker.walk(klass.body, {})
            owner = f"{_display(path)}::{klass.name}"
            if walker.keys:
                scan.keys[owner] = walker.keys
            scan.unresolved.extend(walker.unresolved)
    return scan


# ---------------------------------------------------------------------------
# The model surface
# ---------------------------------------------------------------------------


def _returned_names(fn: ast.FunctionDef) -> Set[str]:
    return {
        node.value.id
        for node in _own_body(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
    }


def _own_body(fn: ast.FunctionDef):
    """Every node in ``fn``'s own body, not in any function defined inside it.

    A nested closure is a different function with its own names: the SCF
    model's LBFGS ``closure`` returns a tensor called ``energy``, and walking
    into it would make every ``energy = ...`` in the enclosing forward look
    like an output dict. The definition itself is dropped along with its body
    -- filtering only the children found on the way down leaves the nested
    ``def`` in the initial list and walks it anyway, which is a hole quiet
    enough to survive review.
    """
    stack = [
        node
        for node in fn.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            stack.append(child)


def scan_model_surface(paths: Optional[Iterable[Path]] = None) -> Scan:
    """Every key a ``forward`` in this package can return.

    The file list is discovered rather than named. Four files define such a
    forward -- ``mace/modules/models.py``, ``mace/modules/extensions.py``,
    ``mace/calculators/lammps_mace.py`` and
    ``mace/calculators/mace_torchsim.py`` -- and the two deployment wrappers
    are precisely what a deployment golden evaluates, so listing the two
    obvious ones is how ``total_energy_local`` stayed unknown to the schema.
    """
    scan = Scan()
    for path in paths if paths is not None else python_sources():
        source = path.read_text(encoding="utf-8")
        if "def forward" not in source:
            continue
        tree = ast.parse(source)
        literals = _collect_literals(tree)
        for klass in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in klass.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name != "forward":
                    continue
                names = _returned_names(fn)
                walker = _WriteWalker(
                    path,
                    source,
                    literals,
                    lambda node, _names=names: isinstance(node, ast.Name)
                    and node.id in _names,
                    enter_functions=False,
                )
                walker.walk(fn.body, {})
                keys = set(walker.keys)
                for node in _own_body(fn):
                    if isinstance(node, ast.Return) and isinstance(
                        node.value, ast.Dict
                    ):
                        keys |= _dict_keys(node.value)
                if keys:
                    owner = f"{_display(path)}::{klass.name}"
                    scan.keys[owner] = keys
                    # Only reported for a forward that does build a dict; a
                    # forward returning a tensor has no keys to miss.
                    scan.unresolved.extend(walker.unresolved)
    return scan


# ---------------------------------------------------------------------------
# The evaluation CLI surface
# ---------------------------------------------------------------------------


def scan_eval_surface(path: Path = EVAL_CLI) -> Tuple[Scan, Dict[str, str]]:
    """Every name the evaluation CLI writes onto a structure, and where.

    Returns the scan and a ``name -> "info" | "arrays"`` map, because the
    store is what says whether a quantity is per graph or per atom, and the
    same name can only be in one of them.

    The keys are recorded **without** the prefix. ``--info_prefix`` is an
    argument, so ``atoms.info[args.info_prefix + "energy"]`` names the channel
    ``energy``; baking the default in would give the schema a spelling per
    invocation.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    scan = Scan()
    stores: Dict[str, str] = {}
    owner = _display(path)
    keys: Set[str] = set()

    for node in ast.walk(tree):
        targets: List[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            store = target.value
            if not (
                isinstance(store, ast.Attribute)
                and store.attr in ("info", "arrays")
                and isinstance(store.value, ast.Name)
            ):
                continue
            name = _prefixed_name(target.slice)
            if name is None:
                line = lines[node.lineno - 1].strip()
                scan.unresolved.append(
                    f"{_display(path)}:{node.lineno}: {line}"
                )
                continue
            keys.add(name)
            previous = stores.setdefault(name, store.attr)
            if previous != store.attr:
                stores[name] = "info+arrays"
    if keys:
        scan.keys[owner] = keys
    return scan, stores


def _prefixed_name(slice_node: ast.AST) -> Optional[str]:
    """``args.info_prefix + "energy"`` -> ``"energy"``.

    A bare string constant is returned as it is: that is a hard-coded prefix,
    and it should fail to resolve against the schema rather than be quietly
    stripped of a prefix nobody wrote.
    """
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    if isinstance(slice_node, ast.BinOp) and isinstance(slice_node.op, ast.Add):
        right = slice_node.right
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            return right.value
    return None


# ---------------------------------------------------------------------------
# The input surface
#
# Not an output at all, but the same problem and the same answer. What an
# evaluation reads is decided by a `KeySpecification`, whose two dicts map a
# property name to an `atoms` key, and the property vocabulary is set in two
# places: the `DefaultKeys` enum (which also fixes the default spellings) and
# the `infos`/`arrays` lists in `update_keyspec_from_kwargs`, which say which
# of the two stores each one addresses. The calculators then add their own
# defaults on top, and those disagree with the enum -- `charge` against
# `total_charge`, `Qs` against `REF_charges`.
#
# All three are read here, so the harness's literal default table can be
# checked against the truth instead of drifting from it.
# ---------------------------------------------------------------------------


@dataclass
class InputSurface:
    #: property -> store ("info" or "arrays")
    stores: Dict[str, str] = field(default_factory=dict)
    #: property -> every default spelling the package uses for it
    spellings: Dict[str, Set[str]] = field(default_factory=dict)


def scan_input_surface() -> InputSurface:
    surface = InputSurface()
    enum_defaults = _default_keys_enum()
    keyspec_stores = _keyspec_property_stores()

    for prop, store in keyspec_stores.items():
        surface.stores[prop] = store
        if prop in enum_defaults:
            surface.spellings.setdefault(prop, set()).add(enum_defaults[prop])

    for prop, (store, spelling) in _calculator_keyspec_defaults().items():
        surface.stores.setdefault(prop, store)
        if spelling is not None:
            surface.spellings.setdefault(prop, set()).add(spelling)
    return surface


def _default_keys_enum(
    path: Path = PACKAGE_ROOT / "tools" / "default_keys.py",
) -> Dict[str, str]:
    """``DefaultKeys`` as ``property name -> default spelling``.

    ``keydict()`` lowercases each member's name and appends ``_key``, and
    ``update_keyspec_from_kwargs`` strips that suffix back off, so the
    property name is simply the lowercased member name.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for klass in ast.walk(tree):
        if not isinstance(klass, ast.ClassDef) or klass.name != "DefaultKeys":
            continue
        for node in klass.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                out[node.targets[0].id.lower()] = node.value.value
    return out


def _keyspec_property_stores(
    path: Path = PACKAGE_ROOT / "data" / "utils.py",
) -> Dict[str, str]:
    """Which store each property is read from, from ``update_keyspec_from_kwargs``.

    The function holds two lists of ``<property>_key`` names, one per store,
    and strips the suffix. Reading them is the only way to know that
    ``magmom`` is an ``arrays`` property and ``total_charge`` an ``info`` one
    without saying so twice.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for fn in ast.walk(tree):
        if (
            not isinstance(fn, ast.FunctionDef)
            or fn.name != "update_keyspec_from_kwargs"
        ):
            continue
        literals = _collect_literals(fn)
        for name, store in (("infos", "info"), ("arrays", "arrays")):
            for spelling in literals.sets.get(name, set()):
                if spelling.endswith("_key"):
                    out[spelling[: -len("_key")]] = store
    return out


def _calculator_keyspec_defaults(
    path: Path = PACKAGE_ROOT / "calculators" / "mace.py",
) -> Dict[str, Tuple[str, Optional[str]]]:
    """The calculators' own keyspec defaults, as ``property -> (store, key)``.

    Two shapes, both in ``mace/calculators/mace.py``: the literal
    ``info_keys = {...}`` fallback in each ``__init__``, and the
    ``arrays_keys.update({...})`` in the batch builder, whose values are the
    ``*_key`` constructor arguments rather than constants -- so the spelling
    is chased back to that argument's default.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults = _constructor_defaults(tree)
    out: Dict[str, Tuple[str, Optional[str]]] = {}

    def store_of(name: str) -> Optional[str]:
        if name == "info_keys":
            return "info"
        if name == "arrays_keys":
            return "arrays"
        return None

    for node in ast.walk(tree):
        store = None
        mapping: Optional[ast.Dict] = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            store = store_of(node.targets[0].id)
            mapping = node.value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Attribute)
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            store = store_of(node.func.value.attr)
            mapping = node.args[0]
        if store is None or mapping is None:
            continue
        for key, value in zip(mapping.keys, mapping.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            spelling: Optional[str] = None
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                spelling = value.value
            elif isinstance(value, ast.Attribute) and isinstance(
                value.value, ast.Name
            ):
                # self.<something>_key -> the constructor default
                spelling = defaults.get(value.attr)
            out[key.value] = (store, spelling)
    return out


def _constructor_defaults(tree: ast.AST) -> Dict[str, Any]:
    """``__init__`` keyword defaults that are plain strings, by argument name."""
    out: Dict[str, Any] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name != "__init__":
            continue
        args = fn.args.args[len(fn.args.args) - len(fn.args.defaults) :]
        for arg, default in zip(args, fn.args.defaults):
            if isinstance(default, ast.Constant) and isinstance(default.value, str):
                out.setdefault(arg.arg, default.value)
    return out
