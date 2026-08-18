"""Golden fixtures, references and the harness that compares against them.

Importing the package registers this repository's key spellings into the
harness's shared schema -- all three surfaces, because there are three and
they do not agree: ``calculator_keys`` for what an ase calculator writes (and
for where every evaluation reads its *inputs*), ``model_keys`` for what a
``forward`` returns, and ``eval_keys`` for what the evaluation command line
writes onto its structures. The imports are here rather than inside
``harness`` on purpose: the harness must stay free of any knowledge of the
framework under test, and every consumer reaches it through this package, so
no registration is ever missed.
"""

from tests.golden import calculator_keys as _calculator_keys  # noqa: F401
from tests.golden import eval_keys as _eval_keys  # noqa: F401
from tests.golden import model_keys as _model_keys  # noqa: F401
