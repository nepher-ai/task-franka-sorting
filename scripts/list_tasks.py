# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the tasks of an EnvHub benchmark: index, id, background, table, container, tray, parts, and instruction.

Pure Python, no simulator::

    python scripts/list_tasks.py --env_id franka-sorting-parts-v2 --background boiler_room
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_sorting" / "franka_sorting"
if "franka_sorting" not in sys.modules:
    _stub = types.ModuleType("franka_sorting")
    _stub.__path__ = [str(PACKAGE)]
    sys.modules["franka_sorting"] = _stub

from franka_sorting.benchmark.bundle import DEFAULT_ENV_ID, load_benchmark, resolve_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env_id", default=DEFAULT_ENV_ID, help="EnvHub bundle id or directory.")
    parser.add_argument("--background", default=None, help="Only the tasks on this background.")
    parser.add_argument("--container", default=None, help="Only the tasks with this container.")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per task.")
    parser.add_argument("--summary", action="store_true", help="Print counts per background, table, container, tray.")
    args = parser.parse_args()
    benchmark = load_benchmark(resolve_bundle(args.env_id))
    rows = []
    for index, layout in enumerate(benchmark.layouts):
        if args.background and layout.background != args.background:
            continue
        if args.container and layout.container != args.container:
            continue
        parts = sum(1 for p in layout.placements if p.asset.kind == "item")
        rows.append(
            dict(
                index=index,
                task_id=layout.task_id,
                background=layout.background,
                table=layout.surface,
                container=layout.container,
                tray=layout.tray,
                parts=parts,
                instruction=layout.instruction,
            )
        )
    if args.summary:
        for key in ("background", "table", "container", "tray"):
            counts: dict[str, int] = {}
            for row in rows:
                counts[row[key]] = counts.get(row[key], 0) + 1
            print(f"{key}: {dict(sorted(counts.items()))}")
        print(f"tasks: {len(rows)}")
        return
    for row in rows:
        if args.json:
            print(json.dumps(row))
        else:
            print(
                f"{row['index']:4d}  {row['task_id']}  {row['background']}  {row['table']}  "
                f"{row['parts']:2d} parts  {row['instruction']}"
            )


if __name__ == "__main__":
    main()
