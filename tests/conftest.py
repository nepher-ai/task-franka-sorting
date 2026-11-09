# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python test setup.

``franka_sorting/__init__`` registers the Isaac Lab tasks and so needs Isaac Lab. The tests here cover the
pure modules only, so they load the package without running its ``__init__``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "source" / "franka_sorting" / "franka_sorting"

if "franka_sorting" not in sys.modules:
    _package = types.ModuleType("franka_sorting")
    _package.__path__ = [str(PACKAGE)]
    sys.modules["franka_sorting"] = _package
