"""Frozen v4.9.5 runtime package used by the v4.9.7 support scan.

Only ``gpr.py`` and ``io.py`` differ from the repository package at the
attested base commit.  The remaining modules are byte-identical to the full
v4.9.5 archived package.  Extending ``__path__`` therefore makes this overlay
a regular, importable package while retaining those identical repository
modules as a fallback.  Preflight verifies both the package/module origins
and the optimizer instrumentation before any production work begins.
"""

from pathlib import Path


_OVERLAY_PACKAGE = Path(__file__).resolve().parent
_REPOSITORY_PACKAGE = _OVERLAY_PACKAGE.parents[3] / "hps_gpr"
if not _REPOSITORY_PACKAGE.is_dir():
    raise ImportError(
        "cannot locate the attested repository hps_gpr package at "
        f"{_REPOSITORY_PACKAGE}"
    )

# Keep the overlay first so its instrumented gpr.py and io.py always win.
__path__ = [str(_OVERLAY_PACKAGE), str(_REPOSITORY_PACKAGE)]
__version__ = "0.1.0"

