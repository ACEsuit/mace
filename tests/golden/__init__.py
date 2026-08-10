"""Golden fixtures, references and the harness that compares against them.

Importing the package registers this repository's calculator key spellings
into the harness's shared schema (see ``calculator_keys``). The import is
here rather than inside ``harness`` on purpose: the harness must stay free of
any knowledge of the framework under test, and every consumer reaches it
through this package, so the registration is never missed.
"""

from tests.golden import calculator_keys as _calculator_keys  # noqa: F401
