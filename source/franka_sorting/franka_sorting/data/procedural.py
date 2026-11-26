# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh tasks for data generation: random container, tray, table, part types, cell map, and count.

A procedural shard is a small bundle directory: its own ``benchmark.yaml``, ``tasks/``, and ``assets/stage.json``
and ``assets/catalog.json``, with the heavy asset folders linked to the master library. The stage loads it like any
other benchmark. Pure Python, no simulator.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from ..layout.catalog import AssetLibrary
from ..layout.sampler import LayoutError, SamplerConfig, sample_sorting_task
from ..layout.task_record import task_payload
from ..layout.validate import validate_layout
from .atomic import atomic_write_text

MAX_DRAWS = 8
"""Layout draws per task before the pair is skipped for another one."""


def config_containers(library: AssetLibrary, seed: int) -> list[str]:
    """Containers in a seeded order. Tray combinations are drawn per task."""
    names = sorted(library.containers)
    order = np.random.default_rng(np.random.SeedSequence([seed, 31337])).permutation(len(names))
    return [names[i] for i in order]


def procedural_tasks(
    library: AssetLibrary,
    stage: dict,
    *,
    shard: int,
    count: int,
    seed: int,
    background: str,
    config: SamplerConfig | None = None,
) -> list[dict]:
    """Task payloads of one procedural shard.

    Containers follow one seeded cycle over the whole run (``shard * count + k``). The tray combination, table,
    part types, cell map, and quota are random per task. The table is one of the stage surfaces.

    Raises:
        LayoutError: No pair of the library yields a valid layout.
    """
    config = config or SamplerConfig()
    containers = config_containers(library, seed)
    tables = [str(surface["name"]) for surface in stage["surfaces"]]
    payloads: list[dict] = []
    for k in range(count):
        cursor = shard * count + k
        for skip in range(len(containers)):
            container = containers[(cursor + skip) % len(containers)]
            layout = _draw(library, container, config, [seed, shard, k, skip])
            if layout is not None:
                break
        else:
            raise LayoutError(f"no valid layout for shard {shard}, task {k}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, shard, k, 7]))
        payloads.append(
            task_payload(
                layout,
                library,
                task_id=f"p{shard:03d}_{k:02d}_{container}",
                background=background,
                surface=tables[int(rng.integers(len(tables)))],
                table_center_xy=config.table_center,
                seed=seed,
            )
        )
    return payloads


def _draw(library: AssetLibrary, container: str, config: SamplerConfig, key: Sequence[int]):
    for attempt in range(MAX_DRAWS):
        rng = np.random.default_rng(np.random.SeedSequence([*key, attempt]))
        try:
            layout = sample_sorting_task(rng, library, container, config)
        except LayoutError:
            continue
        if not validate_layout(layout, library, config):
            return layout
    return None


def write_shard_bundle(master: str | Path, target: str | Path, payloads: Sequence[dict], background: str) -> Path:
    """Write the bundle of one shard under ``target`` and return it.

    Args:
        master: Library bundle (``franka-sorting-parts-v2``) that holds the asset files and ``stage.json``.
        target: Directory to fill. It is rebuilt when it exists.
        payloads: Task files from :func:`procedural_tasks`.
        background: Dome of the shard. It becomes the stage default.
    """
    master, target = Path(master), Path(target)
    stage = json.loads((master / "assets" / "stage.json").read_text(encoding="utf-8"))
    stage["background"] = background
    (target / "tasks").mkdir(parents=True, exist_ok=True)
    for stale in (target / "tasks").glob("*.json"):
        stale.unlink()
    for payload in payloads:
        atomic_write_text(target / "tasks" / f"{payload['task_id']}.json", json.dumps(payload, indent=1) + "\n")
    index = {
        "version": "1.0.0",
        "tasks": [{"task_id": p["task_id"], "layout": p["layout"], "background": p["background"]} for p in payloads],
    }
    atomic_write_text(target / "benchmark.yaml", yaml.safe_dump(index, sort_keys=False))
    assets = target / "assets"
    assets.mkdir(exist_ok=True)
    atomic_write_text(assets / "stage.json", json.dumps(stage, indent=1) + "\n")
    atomic_write_text(assets / "catalog.json", (master / "assets" / "catalog.json").read_text(encoding="utf-8"))
    for folder in sorted(path for path in (master / "assets").iterdir() if path.is_dir()):
        link_directory(folder, assets / folder.name)
    return target


def link_directory(source: Path, link: Path) -> None:
    """Make ``link`` a directory symlink to ``source``. Idempotent."""
    if link.exists() or link.is_symlink():
        if link.resolve() == source.resolve():
            return
        raise FileExistsError(f"{link} exists and is not a link to {source}")
    os.symlink(source, link, target_is_directory=True)


def unlink_directory(link: Path) -> None:
    """Remove a link made by :func:`link_directory` without touching the folder it shows."""
    if link.is_symlink():
        link.unlink()
        return
    if os.name == "nt" and hasattr(os.path, "isjunction") and os.path.isjunction(link):
        link.rmdir()
        return
    raise ValueError(f"{link} is not a link")
