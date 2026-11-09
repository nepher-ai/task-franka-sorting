# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Files that readers never see half written: write next to the target, then rename over it."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def temp_path(path: Path) -> Path:
    """Sibling of ``path`` that a writer fills before :func:`os.replace` moves it into place."""
    return path.with_name(f".{path.name}.tmp")


def atomic_write(path: str | Path, write: Callable[[Path], None]) -> None:
    """Call ``write(temp)`` and rename ``temp`` over ``path``. A failed ``write`` leaves ``path`` untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = temp_path(path)
    try:
        write(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` in one rename."""
    atomic_write(path, lambda temp: temp.write_text(text, encoding="utf-8"))
