# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sorting planner. The public names load :mod:`.sorting` on first use."""

from __future__ import annotations

from typing import Any

_EXPORTS = ("SortingPlanner", "Stage", "build_planner", "reach_limit")


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import sorting

    return getattr(sorting, name)


__all__ = list(_EXPORTS)
