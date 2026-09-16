"""`mace_core` reaches neither the frozen legacy package nor a framework.

The static import contracts in `.importlinter` say the same thing, but they
read the source. This is the runtime half, and it is what catches a reach-in
through `importlib` that no static reading sees.

It runs in a **subprocess**, and that is not incidental. The check is "after
importing every module in the package, is torch in `sys.modules`", and in a
pytest session that has already imported something else the answer says nothing
about this package. A fresh interpreter that imports `mace_core` and nothing
else is the only place the question has a meaning.
"""

from __future__ import annotations

import subprocess
import sys

#: Importing any of these from `mace_core` defeats the purpose of the package.
#: `mace` is the frozen oracle, which stops being independent of the thing it
#: judges the moment the new stack can reach it; the rest are the frameworks
#: the package exists to be free of.
FORBIDDEN = ("mace", "torch", "jax", "jaxlib", "e3nn")

PROBE = """
import importlib, json, pkgutil, sys
import mace_core

names = [mace_core.__name__]
names += [i.name for i in pkgutil.walk_packages(mace_core.__path__, "mace_core.")]
for name in names:
    importlib.import_module(name)

print(json.dumps({
    "modules": sorted({m.split(".", 1)[0] for m in sys.modules}),
    "imported": len(names),
}))
"""


def probe() -> dict:
    import json

    result = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_importing_everything_pulls_in_no_framework_and_no_legacy():
    reached = set(probe()["modules"]) & set(FORBIDDEN)
    assert not reached, (
        f"mace_core reached {sorted(reached)}. The package is the shared "
        f"contract between two frameworks, and the frozen oracle is only an "
        f"oracle while it is unreachable from here."
    )


def test_the_probe_imported_more_than_the_top_level_module():
    """Guards the test above: a probe that imported nothing would pass it."""
    assert probe()["imported"] > 1
