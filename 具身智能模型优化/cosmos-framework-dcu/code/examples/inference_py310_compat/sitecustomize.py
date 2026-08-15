"""Small Python 3.10 shim for the official Cosmos inference entrypoint.

The framework declares Python >=3.10 but uses ``contextlib.chdir``, which was
added in Python 3.11.  This module is placed first on PYTHONPATH by the
inference launchers; the framework source remains unchanged.
"""

from __future__ import annotations

import contextlib
import os
import sys


# Keep optional inference packages behind the environment's site-packages. This
# supplies missing modules without shadowing compatible global dependencies.
optional_pydeps = os.environ.get("COSMOS_OPTIONAL_PYDEPS")
if optional_pydeps and os.path.isdir(optional_pydeps) and optional_pydeps not in sys.path:
    sys.path.append(optional_pydeps)


if not hasattr(contextlib, "chdir"):

    @contextlib.contextmanager
    def chdir(path: str | os.PathLike[str]):
        old_cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    contextlib.chdir = chdir  # type: ignore[attr-defined]
