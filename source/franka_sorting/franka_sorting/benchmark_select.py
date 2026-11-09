# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task selection by id list or by index, without Isaac Lab."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_INDEX = re.compile(r"^(\d+)$")
_RANGE = re.compile(r"^(\d+)\s*[~-]\s*(\d+)$")


def parse_task_indices(specs: Iterable[str], task_count: int) -> list[int]:
    """Sorted, de-duplicated task indices from specs such as ``137``, ``16~31``, ``16-31`` or ``0,5,16~31``.

    Ranges include both ends.

    Raises:
        ValueError: A spec is malformed, a range is reversed, or an index is outside ``0..task_count-1``.
    """
    valid = f"valid indices are 0..{task_count - 1}"
    indices: set[int] = set()
    for spec in specs:
        for piece in str(spec).split(","):
            piece = piece.strip()
            if not piece:
                continue
            if match := _INDEX.match(piece):
                first = last = int(match.group(1))
            elif match := _RANGE.match(piece):
                first, last = int(match.group(1)), int(match.group(2))
                if first > last:
                    raise ValueError(f"task index range {piece!r} is reversed")
            else:
                raise ValueError(f"task index {piece!r} is not an index or a range such as 16~31")
            if last >= task_count:
                raise ValueError(f"task index {piece!r} is out of range, {valid}")
            indices.update(range(first, last + 1))
    if not indices:
        raise ValueError(f"no task index given, {valid}")
    return sorted(indices)


def select_tasks(benchmark, task_id: str) -> tuple[str, ...]:
    """Task ids named by ``task_id``: ``"all"``, one id, or comma-separated ids. Benchmark order, no repeats."""
    pieces = [piece.strip() for piece in str(task_id).split(",") if piece.strip()]
    if len(pieces) <= 1:
        return benchmark.select(pieces[0] if pieces else str(task_id))
    chosen = {name for piece in pieces for name in benchmark.select(piece)}
    return tuple(name for name in benchmark.task_ids if name in chosen)


def shared_background(benchmark, task_ids: Sequence[str]) -> str:
    """The one background that every task in ``task_ids`` uses.

    Raises:
        ValueError: The tasks use more than one background.
    """
    names = sorted({benchmark.task(name).background for name in task_ids})
    if len(names) != 1:
        raise ValueError(
            f"the selected tasks use {len(names)} backgrounds ({', '.join(names)}). "
            "One run renders one background: pick tasks that share it, for example with --task_index."
        )
    return names[0]
