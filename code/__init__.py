"""G.L.A.S.S. pathology pipeline.

This package intentionally shadows the stdlib `code` module. Torch imports
`pdb`, and `pdb` imports `code`, so we expose the stdlib console helpers here
as a compatibility layer before exporting the project modules.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path


def _load_stdlib_code():
    stdlib_dir = Path(sysconfig.get_paths()["stdlib"])
    stdlib_code = stdlib_dir / "code.py"
    spec = importlib.util.spec_from_file_location("_stdlib_code", stdlib_code)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load stdlib code module from {stdlib_code}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_code = _load_stdlib_code()

for _name in ("InteractiveInterpreter", "InteractiveConsole", "compile_command", "interact"):
    globals()[_name] = getattr(_stdlib_code, _name)

from .data import GlassDataset, SampleRecord

__all__ = [
    "GlassDataset",
    "SampleRecord",
    "InteractiveInterpreter",
    "InteractiveConsole",
    "compile_command",
    "interact",
]

