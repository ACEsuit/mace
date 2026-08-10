"""Golden fixtures, references and the harness that compares against them.

Importing the package registers this repository's key spellings into the
harness's shared schema -- both surfaces, because there are two and they do
not agree: ``calculator_keys`` for what an ase calculator writes and
``model_keys`` for what a ``forward`` returns. The imports are here rather
than inside ``harness`` on purpose: the harness must stay free of any
knowledge of the framework under test, and every consumer reaches it through
this package, so neither registration is ever missed.
"""

from tests.golden import calculator_keys as _calculator_keys  # noqa: F401
from tests.golden import model_keys as _model_keys  # noqa: F401
