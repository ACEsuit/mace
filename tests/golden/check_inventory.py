#!/usr/bin/env python3
"""Gate `feature_inventory.md` against the sources it claims to inventory (P0-0).

The inventory is the completeness contract of the rewrite: a feature that has
no row in it is a feature nobody decided about, and the absence of a row is a
bug rather than a decision. Prose cannot enforce that, so every enumerable
surface of the package is re-derived here from the source — by AST, never by
a hardcoded list — and compared against the rows of the inventory as a set.

Seventeen sets are compared. Ten of them are the feature surfaces the
inventory was built around (entry points, the two training parsers, the
per-CLI parsers, model classes, registries, losses, calculator params,
calculator exports, extras); the rest close the holes those ten leave: the
`--model` choices (which can name a class that does not exist), the
user-observable output keys (model forward dicts, the ASE calculator's
`results`, and what `mace_eval_configs` writes into the XYZ), the `MACE_*`
environment variables, the pytest markers, and the default property-key
enum that every labelled dataset on disk depends on.

On top of the set comparisons runs the per-dest disposition gate: every
argparse dest, in every parser, carries its own KEEP/MERGE/DROP. Coverage at
the level of a flag *group* is not coverage — a group row carries its
individual knobs only by implication, and nothing fails when one of them is
never mentioned. Four conditions fail a dest: no row, an empty disposition, a
`REVIEW` disposition, and a row for a dest the source no longer declares.

Flag rows are keyed on the **dest**, resolved exactly as argparse resolves it,
never on the option string: an option string is a spelling, a dest is a knob.
`--swa_lr` and `--stage_two_lr` are one dest and therefore one row (and one
disposition); `--config` is registered with `parser.add`, not `add_argument`,
and is invisible to an extractor that only knows the latter.

Run from anywhere:  python3 tests/golden/check_inventory.py
"""

from __future__ import annotations

import ast
import configparser
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INVENTORY = Path(__file__).with_name("feature_inventory.md")

VALID_DISPOSITIONS = ("KEEP", "MERGE", "DROP")

# --------------------------------------------------------------- pin vocabulary
#
# A `KEEP`/`MERGE` row has to name what protects the behaviour, and until this
# check existed the rule was only that the cell was not empty. The literal
# string "TODO" satisfied that, and the gate still finished with "all sources
# covered" — a completeness contract that accepts "TODO" as evidence is not
# one. So the cell now has to open with something a machine can resolve:
#
#   * a `⚠️ gap` marker — the honest "nothing pins this yet", already counted
#     in the tally and already required to be closed before the phase gate;
#   * a backticked path under `tests/` — which must exist on disk, be specific
#     enough to mean something, and whose `::node_id`, if it carries one, must
#     exist too. A pin naming a test that was renamed or never written is
#     worse than a gap marker, because it reads as coverage;
#   * a ticket id from a known family — Phase 0 pins name tests that are not
#     written yet, which is legitimate and the whole reason the column exists,
#     but "not written yet" and "not a ticket" have to stay distinguishable;
#   * one of the two pins below, where the enforcing thing is a CI job.

#: Ticket-id families. Derived from the destination/retirement columns of the
#: inventory itself, and checked against them at run time so a new family has
#: to be added here deliberately rather than appearing by typo.
TICKET_PREFIXES = (
    "ARCH", "BKD", "CFG", "CLI", "CORE", "DATA", "DEP", "EDU", "ELEC",
    "FM", "FT", "GOV", "INF", "MAG", "P0", "REL", "RET", "TRN",
)
TICKET_RE = re.compile(rf"(?:{'|'.join(TICKET_PREFIXES)})-\d+[a-z]?\b")
ANY_TICKET_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,7})-\d+[a-z]?\b")

#: Pins that are neither a test file nor a ticket, allowed one at a time with
#: the reason each is not a file. Both are cases where the only thing that can
#: fail is a CI job: an extras group is exercised by installing it, and there
#: is no in-tree test that can assert `pip install .[dev]` resolves.
#: Directories too coarse to be a pin. Existing on disk is not the same as
#: pinning something: `tests/` names the entire suite and `tests/unit` names a
#: whole tier, so either satisfies "a valid path" while telling a reader
#: nothing about which behaviour is protected — the same emptiness as "TODO",
#: wearing a path.
#:
#: This is a floor, not a ban on directories. `tests/extensions/magnetic` is a
#: legitimate pin: it is the whole of that family's coverage and splitting the
#: claim across its files would be noise. What separates the two is whether
#: the directory is *about* the row. So the rule is a depth: a pinning
#: directory must sit at least two levels under `tests/` — `tests/a/b` —
#: which admits every per-family and per-integration directory in the tree and
#: rejects exactly the tier-level ones. `tests/workflows` and `tests/unit` are
#: named here rather than left to the depth rule because they are the two a
#: hurried pin actually reaches for.
TOO_COARSE_TO_PIN = {
    "tests": "the entire suite",
    "tests/unit": "the whole fast CPU tier",
    "tests/workflows": "the whole e2e tier",
    "tests/backends": "the whole backend-parity tier",
    "tests/foundations": "the whole downloaded-model tier",
    "tests/extensions": "the whole optional-dependency tier",
    "tests/integrations": "the whole integrations tier",
    "tests/benchmarks": "the whole benchmark tier",
    "tests/golden": "the whole golden tier",
}

#: How deep under `tests/` a directory pin has to sit: `tests/a/b` passes,
#: `tests/a` does not. A file pin is exempt — a file is specific by
#: construction, and `tests/conftest.py` is a legitimate pin for the shared
#: fixtures even though it sits one level down.
MIN_PIN_DIRECTORY_DEPTH = 3

NON_TEST_PINS = {
    "the suite itself": (
        "`[test]` is what the test jobs install; nothing inside the suite can "
        "assert the extras group resolves, because the suite is what it "
        "installs"
    ),
    "the lint job itself": (
        "`[dev]` is what the lint job installs; same reason, and `pre-commit "
        "run --all-files` is the assertion"
    ),
}

COLUMNS = (
    "id",
    "feature",
    "source",
    "disposition",
    "pinned by",
    "destination",
    "retirement",
    "status",
)

# The thirteen per-CLI parsers of `mace/cli/`. Six of them are the `convert_*`
# family, and three of those six have no console entry point at all, so a list
# derived from `setup.cfg` misses them twice over: not as entry points and not
# as parsers. The list is derived from the directory, not written out, so a new
# CLI cannot be missed the same way.
CLI_DIR = REPO / "mace" / "cli"


@dataclass(frozen=True)
class Decl:
    """One enumerable thing in the source, with enough context to fix it."""

    key: str  # the identifier the inventory row must carry
    detail: str  # option strings, defining class, ... ("" when there is none)
    site: str  # file:line of the declaration


@dataclass
class SourceSet:
    label: str  # human name, printed per comparison
    prefix: str  # inventory id namespace, e.g. "train."
    noun: str  # what one element is called in the failure message
    decls: dict[str, Decl] = field(default_factory=dict)


# --------------------------------------------------------------------- helpers


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _str_consts(node: ast.AST) -> list[ast.Constant]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"function {name}() not found — the extractor is stale")


# ------------------------------------------------------------------- argparse


def _is_add_call(node: ast.AST) -> bool:
    """True for `parser.add_argument(...)` and configargparse's `parser.add(...)`.

    `add` is also `set.add`, so it only counts when the first string argument
    looks like an option string; `add_argument` also declares positionals, which
    do not.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in ("add_argument", "add"):
        return False
    strings = [a for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not strings:
        return False
    return node.func.attr == "add_argument" or strings[0].value.startswith("-")


def _resolve_dest(call: ast.Call) -> tuple[str, list[str], int]:
    """Resolve a dest exactly as argparse does, plus its option strings and line."""
    strings = [a for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    options = [a.value for a in strings]
    lineno = strings[0].lineno
    for kw in call.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value), options, lineno
    for option in options:
        if option.startswith("--"):
            return option[2:].replace("-", "_"), options, lineno
    return options[0].lstrip("-").replace("-", "_"), options, lineno


def _dests(scope: ast.AST, path: Path) -> dict[str, Decl]:
    out: dict[str, Decl] = {}
    for node in ast.walk(scope):
        if not _is_add_call(node):
            continue
        dest, options, lineno = _resolve_dest(node)
        if dest in out:  # a dest declared twice in one parser is still one knob
            continue
        out[dest] = Decl(dest, " ".join(options), f"{_rel(path)}:{lineno}")
    return out


# ------------------------------------------------------------- source surfaces


def _setup_cfg() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(REPO / "setup.cfg", encoding="utf-8")
    return parser


def source_entry_points() -> dict[str, Decl]:
    cfg = _setup_cfg()
    raw = cfg["options.entry_points"]["console_scripts"].strip().splitlines()
    out = {}
    for line in raw:
        name, target = (part.strip() for part in line.split("=", 1))
        out[name] = Decl(name, target, "setup.cfg")
    return out


def source_extras() -> dict[str, Decl]:
    cfg = _setup_cfg()
    return {name: Decl(name, "", "setup.cfg") for name in cfg["options.extras_require"]}


def source_train_dests() -> dict[str, Decl]:
    path = REPO / "mace" / "tools" / "arg_parser.py"
    return _dests(_func(_parse(path), "build_default_arg_parser"), path)


def source_preprocess_dests() -> dict[str, Decl]:
    path = REPO / "mace" / "tools" / "arg_parser.py"
    return _dests(_func(_parse(path), "build_preprocess_arg_parser"), path)


def source_cli_dests() -> dict[str, Decl]:
    """Every dest of every parser under `mace/cli/`, keyed `<module>.<dest>`.

    A dest is counted once per parser that declares it: `--device` in four
    different CLIs is four knobs with four defaults and four help strings, so
    it needs four dispositions.
    """
    out: dict[str, Decl] = {}
    for path in sorted(CLI_DIR.glob("*.py")):
        tree = _parse(path)
        dests = _dests(tree, path)
        if not dests:
            continue
        for dest, decl in dests.items():
            out[f"{path.stem}.{dest}"] = Decl(f"{path.stem}.{dest}", decl.detail, decl.site)
    return out


def source_model_choices() -> dict[str, Decl]:
    """The `--model` choices — CLI-selectable model names.

    Deliberately a separate set from the model classes: two of the choices name
    a class that exists nowhere in the tree and reach only a deprecation raise.
    """
    path = REPO / "mace" / "tools" / "arg_parser.py"
    for node in ast.walk(_func(_parse(path), "build_default_arg_parser")):
        if not _is_add_call(node):
            continue
        dest, _, _ = _resolve_dest(node)
        if dest != "model":
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                return {
                    c.value: Decl(c.value, "--model choice", f"{_rel(path)}:{c.lineno}")
                    for c in _str_consts(kw.value)
                }
    raise SystemExit("--model has no choices= list — the extractor is stale")


def _classes(path: Path) -> dict[str, Decl]:
    return {
        node.name: Decl(node.name, "", f"{_rel(path)}:{node.lineno}")
        for node in _parse(path).body
        if isinstance(node, ast.ClassDef)
    }


def source_model_classes() -> dict[str, Decl]:
    out = _classes(REPO / "mace" / "modules" / "models.py")
    out.update(_classes(REPO / "mace" / "modules" / "extensions.py"))
    return out


def source_loss_classes() -> dict[str, Decl]:
    return _classes(REPO / "mace" / "modules" / "loss.py")


def source_registries() -> dict[str, Decl]:
    """The string->class registries that connect CLI values to implementations."""
    path = REPO / "mace" / "modules" / "__init__.py"
    wanted = ("interaction_classes", "readout_classes", "scaling_classes", "gate_dict")
    out: dict[str, Decl] = {}
    for node in ast.walk(_parse(path)):
        # the registries are annotated assignments (`x: Dict[...] = {...}`)
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        registry = next((n for n in names if n in wanted), None)
        if registry is None or not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                out[key.value] = Decl(key.value, registry, f"{_rel(path)}:{key.lineno}")
    missing = set(wanted) - {d.detail for d in out.values()}
    if missing:
        raise SystemExit(f"registries not found: {sorted(missing)} — extractor is stale")
    return out


def _init_params(path: Path, class_name: str) -> dict[str, Decl]:
    for node in _parse(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = item.args.args + item.args.kwonlyargs
                    return {
                        a.arg: Decl(a.arg, class_name, f"{_rel(path)}:{a.lineno}")
                        for a in args
                        if a.arg != "self"
                    }
    raise SystemExit(f"{class_name}.__init__ not found — the extractor is stale")


def source_calculator_params() -> dict[str, Decl]:
    """`MACECalculator.__init__` plus whatever `MagneticMACECalculator` adds.

    The magnetic calculator is a second Calculator subclass rather than a mode
    of the first, so its `__init__` is a second public surface; taking the union
    keeps its extra knobs from being invisible.
    """
    path = REPO / "mace" / "calculators" / "mace.py"
    out = _init_params(path, "MACECalculator")
    for name, decl in _init_params(path, "MagneticMACECalculator").items():
        out.setdefault(name, decl)
    return out


def source_calculator_exports() -> dict[str, Decl]:
    path = REPO / "mace" / "calculators" / "__init__.py"
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return {
                name: Decl(name, "", f"{_rel(path)}:{node.lineno}")
                for name in ast.literal_eval(node.value)
            }
    raise SystemExit("mace/calculators/__init__.py has no __all__")


def source_model_output_keys() -> dict[str, Decl]:
    """Keys of the dicts the model `forward`s return — the contract every
    consumer (calculator, eval CLI, LAMMPS, training loop) reads."""
    out: dict[str, Decl] = {}
    for name in ("models.py", "extensions.py"):
        path = REPO / "mace" / "modules" / name
        for cls in _parse(path).body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name != "forward":
                    continue
                for node in ast.walk(fn):
                    if isinstance(node, ast.Dict) and node.keys and all(
                        isinstance(k, ast.Constant) and isinstance(k.value, str)
                        for k in node.keys
                    ):
                        for key in node.keys:
                            out.setdefault(
                                key.value,
                                Decl(key.value, cls.name, f"{_rel(path)}:{key.lineno}"),
                            )
                    # keys added after the literal, e.g. the SCF wrapper's
                    # extra outputs; `data[...]` is an input, not an output.
                    if isinstance(node, ast.Assign):
                        for tgt in node.targets:
                            if (
                                isinstance(tgt, ast.Subscript)
                                and isinstance(tgt.value, ast.Name)
                                and tgt.value.id != "data"
                                and isinstance(tgt.slice, ast.Constant)
                                and isinstance(tgt.slice.value, str)
                            ):
                                out.setdefault(
                                    tgt.slice.value,
                                    Decl(
                                        tgt.slice.value,
                                        cls.name,
                                        f"{_rel(path)}:{tgt.lineno}",
                                    ),
                                )
    return out


def source_calculator_result_keys() -> dict[str, Decl]:
    """What lands in `Calculator.results` — the ASE-facing surface.

    Four shapes contribute: `implemented_properties` lists, direct
    `self.results["k"] = ...` assignments, the `results_map` table, and the
    committee suffixes derived from `results_store_ensemble`.
    """
    path = REPO / "mace" / "calculators" / "mace.py"
    out: dict[str, Decl] = {}

    def add(key: str, lineno: int, detail: str) -> None:
        out.setdefault(key, Decl(key, detail, f"{_rel(path)}:{lineno}"))

    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == "implemented_properties":
                    for const in _str_consts(node.value):
                        add(const.value, const.lineno, "implemented_properties")
                if (
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Attribute)
                    and tgt.value.attr == "results"
                    and isinstance(tgt.slice, ast.Constant)
                ):
                    add(tgt.slice.value, tgt.lineno, "self.results[...]")
                if isinstance(tgt, ast.Name) and tgt.id == "results_map":
                    for elt in ast.walk(node.value):
                        if isinstance(elt, ast.Tuple) and isinstance(elt.elts[0], ast.Constant):
                            add(elt.elts[0].value, elt.lineno, "results_map")
                if isinstance(tgt, ast.Name) and tgt.id == "results_store_ensemble":
                    for const in _str_consts(node.value):
                        for suffix in ("_comm", "_var"):
                            add(const.value + suffix, const.lineno, "committee")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("extend", "append")
        ):
            base = node.func.value
            if isinstance(base, ast.Attribute) and base.attr == "implemented_properties":
                for const in _str_consts(node):
                    add(const.value, const.lineno, "implemented_properties")
            if isinstance(base, ast.Name) and base.id == "results_map":
                for elt in ast.walk(node):
                    if isinstance(elt, ast.Tuple) and isinstance(elt.elts[0], ast.Constant):
                        add(elt.elts[0].value, elt.lineno, "results_map")
    return out


def source_eval_output_keys() -> dict[str, Decl]:
    """The `atoms.info` / `atoms.arrays` keys `mace_eval_configs` writes.

    All of them are written as `info_prefix + "<key>"`, so the constant on the
    right of the concatenation is the key; the prefix itself is a flag.
    """
    path = REPO / "mace" / "cli" / "eval_configs.py"
    out: dict[str, Decl] = {}
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not isinstance(tgt, ast.Subscript) or not isinstance(tgt.value, ast.Attribute):
                continue
            where = tgt.value.attr
            if where not in ("info", "arrays"):
                continue
            slc = tgt.slice
            if isinstance(slc, ast.BinOp) and isinstance(slc.right, ast.Constant):
                out.setdefault(
                    slc.right.value,
                    Decl(slc.right.value, f"atoms.{where}", f"{_rel(path)}:{tgt.lineno}"),
                )
    return out


def source_env_vars() -> dict[str, Decl]:
    """`MACE_*` environment variables read anywhere in the package.

    Matched on the literal rather than on `os.environ`, because the MLIAP
    runtime reads its six through a helper and a call-site-shaped extractor
    would report none of them.
    """
    out: dict[str, Decl] = {}
    for path in sorted((REPO / "mace").rglob("*.py")):
        if "torch_geometric" in path.parts:  # vendored
            continue
        for const in _str_consts(_parse(path)):
            if re.fullmatch(r"MACE_[A-Z0-9_]+", const.value):
                out.setdefault(
                    const.value, Decl(const.value, "", f"{_rel(path)}:{const.lineno}")
                )
    return out


def source_pytest_markers() -> dict[str, Decl]:
    """Registered pytest markers. Most are capabilities; the inventory has to
    say which are not, since INF-5 generates its manifest from this list."""
    path = REPO / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    block = re.search(r"^markers = \[(.*?)^\]", text, re.S | re.M)
    if block is None:
        raise SystemExit("no markers list in pyproject.toml — the extractor is stale")
    start = text[: block.start(1)].count("\n") + 1
    out = {}
    for offset, line in enumerate(block.group(1).splitlines()):
        match = re.search(r'"([a-z_]+):', line)
        if match:
            out[match.group(1)] = Decl(match.group(1), "", f"{_rel(path)}:{start + offset}")
    return out


def source_default_keys() -> dict[str, Decl]:
    """`DefaultKeys` — the on-disk data contract. Every labelled dataset in the
    wild uses these names, so a silent rename breaks all of them at once."""
    path = REPO / "mace" / "tools" / "default_keys.py"
    for node in _parse(path).body:
        if isinstance(node, ast.ClassDef) and node.name == "DefaultKeys":
            return {
                item.targets[0].id: Decl(
                    item.targets[0].id,
                    item.value.value,
                    f"{_rel(path)}:{item.lineno}",
                )
                for item in node.body
                if isinstance(item, ast.Assign)
                and isinstance(item.targets[0], ast.Name)
                and isinstance(item.value, ast.Constant)
            }
    raise SystemExit("DefaultKeys not found — the extractor is stale")


def collect_sources() -> list[SourceSet]:
    return [
        SourceSet("entry points", "ep.", "entry points", source_entry_points()),
        SourceSet("mace_run_train dests", "train.", "dests", source_train_dests()),
        SourceSet("mace_prepare_data dests", "prep.", "dests", source_preprocess_dests()),
        SourceSet("mace/cli parser dests", "cli.", "dests", source_cli_dests()),
        SourceSet("--model choices", "choice.", "choices", source_model_choices()),
        SourceSet("model-level classes", "model.", "classes", source_model_classes()),
        SourceSet("registry entries", "reg.", "entries", source_registries()),
        SourceSet("loss classes", "loss.", "classes", source_loss_classes()),
        SourceSet("calculator params", "calc.param.", "params", source_calculator_params()),
        SourceSet("calculator exports", "calc.export.", "exports", source_calculator_exports()),
        SourceSet("optional extras", "extra.", "extras", source_extras()),
        SourceSet("model output keys", "out.model.", "keys", source_model_output_keys()),
        SourceSet("calculator result keys", "out.calc.", "keys", source_calculator_result_keys()),
        SourceSet("eval_configs output keys", "out.eval.", "keys", source_eval_output_keys()),
        SourceSet("MACE_* env vars", "env.", "variables", source_env_vars()),
        SourceSet("pytest markers", "marker.", "markers", source_pytest_markers()),
        SourceSet("default property keys", "key.", "keys", source_default_keys()),
    ]


# ------------------------------------------------------------------- inventory


@dataclass
class Row:
    ident: str
    feature: str
    source: str
    disposition: str
    pinned_by: str
    destination: str
    retirement: str
    status: str
    line: int

    @property
    def verdict(self) -> str:
        return self.disposition.split("—")[0].split("--")[0].strip()

    @property
    def reason(self) -> str:
        parts = re.split(r"—|--", self.disposition, maxsplit=1)
        return parts[1].strip() if len(parts) > 1 else ""


def read_rows() -> tuple[list[Row], list[str]]:
    """Parse the inventory's tables. A row is any table line whose first cell is
    a backticked identifier; everything else (headers, separators, the
    orientation tables) is prose as far as the gate is concerned."""
    rows: list[Row] = []
    problems: list[str] = []
    for lineno, line in enumerate(INVENTORY.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        match = re.fullmatch(r"`([^`]+)`", cells[0])
        if match is None:
            continue
        if len(cells) != len(COLUMNS):
            problems.append(
                f"{INVENTORY.name}:{lineno}: row `{match.group(1)}` has "
                f"{len(cells)} cells, expected {len(COLUMNS)} ({', '.join(COLUMNS)})"
            )
            continue
        rows.append(Row(match.group(1), *cells[1:], line=lineno))
    return rows, problems


def _node_id_exists(path: Path, node_id: str) -> bool:
    """Whether `node_id` is declared under `path` (a file or a directory).

    Two forms, because a pin has to be able to name the two kinds of thing
    that actually enforce a row:

    * `test_name` — a `def test_name(` somewhere under the path;
    * `TABLE[key]` — an entry of a module-level dict literal. A capability
      marker is not enforced by a function; it is enforced by having a probe
      in `CAPABILITY_PROBES`, and nothing but the entry itself can stand for
      that.
    """
    files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
    entry = re.fullmatch(r"(\w+)\[(\w+)\]", node_id)
    if entry is not None:
        table, key = entry.group(1), entry.group(2)
        return any(_dict_has_key(f, table, key) for f in files)
    needle = re.compile(rf"^\s*def {re.escape(node_id)}\s*\(", re.MULTILINE)
    return any(needle.search(f.read_text(encoding="utf-8")) for f in files)


def _dict_literal(path: Path, name: str) -> dict | None:
    """The string keys of a module-level `name = {...}`, or None."""
    for node in _parse(path).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name in targets:
            return {
                key.value: key
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    return None


def _dict_has_key(path: Path, name: str, key: str) -> bool:
    keys = _dict_literal(path, name)
    return keys is not None and key in keys


def _too_coarse(path: str) -> str | None:
    """Why `path` is too broad to be a pin, or None if it is specific enough."""
    named = TOO_COARSE_TO_PIN.get(path)
    if named is not None:
        return named
    if len(Path(path).parts) < MIN_PIN_DIRECTORY_DEPTH:
        return f"only {len(Path(path).parts)} levels deep"
    return None


def check_pins(rows: list[Row]) -> list[str]:
    """Every pin resolves to something: a file, a ticket, a gap, or a CI job.

    Non-empty is not a rule. This is what makes the `pinned by` column
    evidence rather than decoration — see the vocabulary comment at the top.
    """
    problems: list[str] = []
    for row in rows:
        pin = row.pinned_by.strip()
        if not pin or pin.startswith("—"):
            continue  # required-ness is check_row_hygiene's business

        # Every test path named anywhere in the cell has to resolve, not just
        # the leading one: a row pinned on "A + B" claims both.
        for span in re.findall(r"`([^`]+)`", pin):
            # `tests` with no slash is caught here too: it resolves on disk,
            # so leaving it to the free-text rule below would reject it for
            # the wrong reason.
            if span.rstrip("/") != "tests" and not span.startswith("tests/"):
                continue  # flag and command names quoted inside gap prose
            file_part, _, node_id = span.partition("::")
            target = REPO / file_part
            normalised = file_part.rstrip("/")
            if not target.exists():
                problems.append(
                    f"{INVENTORY.name}:{row.line}: `{row.ident}` is pinned by "
                    f"`{span}`, which does not exist on disk"
                )
            elif target.is_dir() and not node_id and _too_coarse(normalised):
                problems.append(
                    f"{INVENTORY.name}:{row.line}: `{row.ident}` is pinned by "
                    f"`{span}`, which is {_too_coarse(normalised)}. A pin has "
                    f"to say which behaviour is protected; a directory that "
                    f"broad is 'TODO' wearing a path. Name the file, the test, "
                    f"or a directory at least {MIN_PIN_DIRECTORY_DEPTH} levels "
                    f"deep (`tests/extensions/magnetic` is fine)"
                )
            elif node_id and not _node_id_exists(target, node_id):
                problems.append(
                    f"{INVENTORY.name}:{row.line}: `{row.ident}` is pinned by "
                    f"`{span}`, but no `{node_id}` exists under {file_part}"
                )

        if pin in NON_TEST_PINS:
            continue
        if pin.startswith("⚠️"):
            continue
        if re.match(r"`tests/", pin):
            continue
        if TICKET_RE.match(pin):
            continue

        lead = ANY_TICKET_RE.match(pin)
        if lead is not None:
            problems.append(
                f"{INVENTORY.name}:{row.line}: `{row.ident}` is pinned by "
                f"'{pin}', whose ticket family '{lead.group(1)}' is not one of "
                f"{'/'.join(TICKET_PREFIXES)}"
            )
        else:
            problems.append(
                f"{INVENTORY.name}:{row.line}: `{row.ident}` is pinned by "
                f"'{pin}', which is free text. A pin must open with a "
                f"⚠️ gap marker, a backticked path under tests/, a ticket id, "
                f"or one of {sorted(NON_TEST_PINS)}"
            )
    return problems


CONFTEST = REPO / "tests" / "conftest.py"


def check_marker_rows(rows: list[Row]) -> list[str]:
    """A capability marker must pin its own probe, not the file it lives in.

    Twelve of the thirteen `marker.*` rows pinned `tests/conftest.py`, and the
    only thing asserted about that was that the file exists. It does, and it
    always will, so every one of those pins passed for a reason that had
    nothing to do with the marker: `marker.anything` would have passed
    identically. That is the "TODO" failure again -- a cell that resolves
    without discriminating.

    What actually enforces a capability marker is having an entry in
    `CAPABILITY_PROBES`: that dict is what `pytest_runtest_setup` iterates, so
    a marker missing from it is registered, usable, silently never checked,
    and -- the part that matters -- invisible to the `MACE_REQUIRE_CAPS`
    skip-o-fail contract. So a capability row has to pin
    `tests/conftest.py::CAPABILITY_PROBES[<name>]`, which resolves to the
    entry itself and fails the moment that entry is renamed or dropped.

    The three cost markers (`slow`, `benchmark`, `timeout`) are the mirror
    rule: they have no probe and must not claim one, because absorbing them
    into the capability manifest is exactly the mistake the inventory row for
    `marker.timeout` was written to prevent.
    """
    problems: list[str] = []
    probes = _dict_literal(CONFTEST, "CAPABILITY_PROBES")
    if probes is None:
        return [
            f"{_rel(CONFTEST)}: no module-level CAPABILITY_PROBES dict — the "
            f"marker check is stale, and every marker row is unenforced until "
            f"it is fixed"
        ]

    marker_rows = {
        row.ident[len("marker.") :]: row
        for row in rows
        if row.ident.startswith("marker.")
    }
    for name in sorted(set(probes) - set(marker_rows)):
        problems.append(
            f"{_rel(CONFTEST)}: capability '{name}' has a probe but no "
            f"`marker.{name}` row in the inventory"
        )

    for name, row in sorted(marker_rows.items()):
        pin = row.pinned_by.strip()
        wanted = f"`tests/conftest.py::CAPABILITY_PROBES[{name}]`"
        claims_probe = "CAPABILITY_PROBES[" in pin
        if name in probes:
            if wanted not in pin:
                problems.append(
                    f"{INVENTORY.name}:{row.line}: `{row.ident}` is a "
                    f"capability marker pinned by '{pin}'. Pinning the file "
                    f"asserts only that tests/conftest.py exists, which is "
                    f"true for every marker and so discriminates none of "
                    f"them; pin {wanted} instead"
                )
        elif claims_probe:
            problems.append(
                f"{INVENTORY.name}:{row.line}: `{row.ident}` pins a "
                f"CAPABILITY_PROBES entry, but '{name}' has no probe. It is a "
                f"cost marker, and claiming a probe is what would sweep it "
                f"into the capability manifest"
            )
        elif pin == "`tests/conftest.py`":
            problems.append(
                f"{INVENTORY.name}:{row.line}: `{row.ident}` is a cost marker "
                f"pinned by the bare conftest, which every marker row could "
                f"claim. Name what applies it -- "
                f"`tests/conftest.py::pytest_collection_modifyitems` for the "
                f"directory-derived ones"
            )
    return problems


def check_ticket_prefixes(rows: list[Row]) -> list[str]:
    """The recognised families must cover the ones the inventory actually uses.

    Without this the constant above rots silently in the permissive
    direction: a new ticket family appears in a destination column, a pin
    names it, and the pin is rejected as free text for a reason that looks
    like a typo.
    """
    unknown: dict[str, int] = {}
    for row in rows:
        for column in (row.destination, row.retirement):
            for match in ANY_TICKET_RE.finditer(column):
                if match.group(1) not in TICKET_PREFIXES:
                    unknown.setdefault(match.group(1), row.line)
    return [
        f"{INVENTORY.name}:{line}: ticket family '{prefix}' is used by the "
        f"inventory but is not in TICKET_PREFIXES; add it there so a pin may "
        f"name it"
        for prefix, line in sorted(unknown.items())
    ]


def check_row_hygiene(rows: list[Row]) -> list[str]:
    """Rules that hold for every row, gated section or not."""
    problems: list[str] = []
    seen: dict[str, int] = {}
    for row in rows:
        if row.ident in seen:
            problems.append(
                f"{INVENTORY.name}:{row.line}: duplicate id `{row.ident}` "
                f"(first seen at line {seen[row.ident]})"
            )
        seen[row.ident] = row.line
        if row.verdict not in VALID_DISPOSITIONS:
            problems.append(
                f"{INVENTORY.name}:{row.line}: `{row.ident}` has disposition "
                f"'{row.disposition or '(empty)'}' — must be one of "
                f"{'/'.join(VALID_DISPOSITIONS)}"
            )
            continue
        if row.verdict == "DROP" and not row.reason:
            # A drop without a reason is a deletion nobody can review, and
            # REL-1's migration guide is generated from these reasons.
            problems.append(
                f"{INVENTORY.name}:{row.line}: `{row.ident}` is DROP with no "
                f"justification (write 'DROP — why')"
            )
        if row.verdict in ("KEEP", "MERGE"):
            if not row.pinned_by or row.pinned_by == "—":
                problems.append(
                    f"{INVENTORY.name}:{row.line}: `{row.ident}` is {row.verdict} "
                    f"with no pinning test and no ⚠️ gap marker"
                )
            if not row.destination or row.destination == "—":
                problems.append(
                    f"{INVENTORY.name}:{row.line}: `{row.ident}` is {row.verdict} "
                    f"with no destination ticket"
                )
            if not re.match(r"(RET-\d|n/a\b)", row.retirement):
                problems.append(
                    f"{INVENTORY.name}:{row.line}: `{row.ident}` is {row.verdict} "
                    f"with no retirement ticket (RET-N, or 'n/a — why')"
                )
        if not row.status:
            problems.append(f"{INVENTORY.name}:{row.line}: `{row.ident}` has no status")
    return problems


def check_set(source: SourceSet, rows: list[Row]) -> tuple[bool, list[str]]:
    """Compare one source surface against the rows in its id namespace.

    Missing rows and unusable dispositions are reported together: from the
    gate's point of view a dest with no row and a dest whose row says nothing
    are the same failure, and both are fixed in the same place.
    """
    have = {r.ident[len(source.prefix) :]: r for r in rows if r.ident.startswith(source.prefix)}
    missing = sorted(set(source.decls) - set(have))
    stale = sorted(set(have) - set(source.decls))
    undecided = sorted(
        key for key, row in have.items() if row.verdict not in VALID_DISPOSITIONS
    )
    ok = not (missing or stale or undecided)
    print(
        f"[{'ok' if ok else 'FAIL'}] {source.label}: "
        f"source={len(source.decls)} inventory={len(have)}"
    )
    report: list[str] = []
    offenders = sorted(set(missing) | set(undecided))
    if offenders:
        report.append(
            f"FAIL {source.label}: {len(offenders)} {source.noun} without a disposition"
        )
        width = max(len(o) for o in offenders)
        for key in offenders:
            decl = source.decls[key]
            detail = decl.detail or "—"
            report.append(f"  {key:<{width}}  {detail:<28}  {decl.site}")
    if stale:
        report.append(
            f"FAIL {source.label}: {len(stale)} rows for {source.noun} the source "
            f"no longer declares"
        )
        for key in stale:
            report.append(f"  {source.prefix}{key}  (inventory line {have[key].line})")
    return ok, report


def main() -> int:
    if not INVENTORY.exists():
        print(f"inventory not found: {INVENTORY}")
        return 1

    rows, problems = read_rows()
    reports: list[str] = []
    failed = bool(problems)

    for source in collect_sources():
        ok, report = check_set(source, rows)
        failed = failed or not ok
        reports.extend(report)

    hygiene = check_row_hygiene(rows)
    hygiene += check_pins(rows)
    hygiene += check_marker_rows(rows)
    hygiene += check_ticket_prefixes(rows)
    failed = failed or bool(hygiene)

    gaps = [r for r in rows if "⚠️" in r.pinned_by]
    review = [r for r in rows if r.verdict == "REVIEW" or "REVIEW" in r.disposition]
    counts = {d: sum(1 for r in rows if r.verdict == d) for d in VALID_DISPOSITIONS}
    print(
        f"\ntally: {len(rows)} rows — "
        + ", ".join(f"{n} {d}" for d, n in counts.items())
        + f"; {len(gaps)} carry a ⚠️ gap marker; {len(review)} carry a REVIEW disposition"
    )

    if problems or hygiene:
        print("\nROW ERRORS:")
        for problem in problems + hygiene:
            print(" -", problem)
    if reports:
        print()
        print("\n".join(reports))

    print()
    if failed:
        print("INVENTORY CHECK FAILED")
        return 1
    print("all sources covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
