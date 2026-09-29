"""A MACE kernel backend in its own distribution.

Registered under the entry point group ``mace.kernel_backends.torch`` as
``example``, and discovered only through it. See the distribution's README for
how it is put together and how to check a backend of your own.
"""

from importlib.metadata import PackageNotFoundError, version

from mace_backend_example.backend import ExampleBackend, ExampleCapabilities

try:
    __version__ = version("mace-backend-example")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["ExampleBackend", "ExampleCapabilities", "__version__"]
