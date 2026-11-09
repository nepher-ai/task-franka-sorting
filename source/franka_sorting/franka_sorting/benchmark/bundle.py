# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load one EnvHub benchmark: the task index, the task files, and the stage."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..layout.catalog import AssetLibrary, load_library
from .stage import read_stage
from .task import TASK_TIME_LIMIT_S, TaskLayout, layout_from_payload

DEFAULT_ENV_ID = "franka-sorting-parts-v0"
ALL_TASKS = "all"
"""Task selector that means every task of the benchmark."""


def resolve_bundle(env_id: str | None = None) -> Path:
    """Directory of EnvHub bundle ``env_id`` in the nepher cache.

    An ``env_id`` that is an existing directory with a ``benchmark.yaml`` is used as it is, so a bundle that is
    not in the cache (a fresh build, a test fixture) can be loaded by path.
    """
    env_id = env_id or DEFAULT_ENV_ID
    if (Path(env_id) / "benchmark.yaml").is_file():
        return Path(env_id).resolve()
    try:
        from nepher.storage.cache import get_cache_manager
    except ImportError as error:
        raise FileNotFoundError(
            f"EnvHub bundle {env_id} is loaded from the nepher cache. "
            f"Install nepher and run `nepher download {env_id}`."
        ) from error
    cache = get_cache_manager(category="manipulation")
    if not cache.is_cached(env_id):
        raise FileNotFoundError(
            f"EnvHub bundle {env_id} is not in the nepher cache ({cache.cache_dir}). Run `nepher download {env_id}`."
        )
    return cache.get_env_cache_path(env_id)


@dataclass
class Benchmark:
    """Tasks of one bundle, in ``benchmark.yaml`` order."""

    root: Path
    version: str
    task_ids: tuple[str, ...]
    layouts: tuple[TaskLayout, ...]
    records: tuple[dict, ...]
    stage: dict

    def select(self, task_id: str) -> tuple[str, ...]:
        """Every task id for ``"all"``, otherwise that one task."""
        if task_id == ALL_TASKS:
            return self.task_ids
        if task_id not in self.task_ids:
            raise FileNotFoundError(f"benchmark {self.root.name} has no task {task_id}")
        return (task_id,)

    def task(self, task_id: str) -> TaskLayout:
        """Layout of one task."""
        try:
            return self.layouts[self.task_ids.index(task_id)]
        except ValueError as exc:
            raise FileNotFoundError(f"benchmark {self.root.name} has no task {task_id}") from exc

    def record(self, task_id: str) -> dict:
        try:
            return self.records[self.task_ids.index(task_id)]
        except ValueError as exc:
            raise FileNotFoundError(f"benchmark {self.root.name} has no task {task_id}") from exc

    @functools.cached_property
    def library(self) -> AssetLibrary:
        """Parts, containers, trays, and tables that the tasks of this bundle use (``assets/catalog.json``)."""
        return load_library(self.root)

    def file(self, relative: str) -> Path:
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"benchmark is missing {relative} under {self.root}")
        return path


def load_benchmark(bundle: str | Path) -> Benchmark:
    """Parse ``benchmark.yaml``, every task file, and ``assets/stage.json``. Cached per directory.

    The index and the task files must name the same tasks, with matching ``layout`` and ``background``.
    """
    return _load_cached(str(Path(bundle).resolve()))


def task_ids_of(env_id: str, task_id: str) -> list[str] | None:
    """Task ids ``task_id`` runs, in env order. ``None`` when it is not a selector of this benchmark."""
    bundle = resolve_bundle(env_id)
    try:
        return list(load_benchmark(bundle).select(str(task_id)))
    except FileNotFoundError:
        return None


@functools.lru_cache(maxsize=8)
def _load_cached(bundle: str) -> Benchmark:
    root = Path(bundle)
    index_path = root / "benchmark.yaml"
    if not index_path.is_file():
        raise FileNotFoundError(f"benchmark has no benchmark.yaml under {root}")
    index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    entries = list(index.get("tasks") or [])
    files = sorted((root / "tasks").glob("*.json")) if (root / "tasks").is_dir() else []
    by_file = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(payload.get("task_id", path.stem))
        if task_id in by_file:
            raise ValueError(f"task {task_id} has more than one file under {root / 'tasks'}")
        if path.stem != task_id:
            raise ValueError(f"task file {path.name} must be named {task_id}.json")
        by_file[task_id] = payload
    listed = [str(entry["task_id"]) for entry in entries]
    if listed != list(dict.fromkeys(listed)):
        raise ValueError(f"benchmark.yaml repeats a task id under {root}")
    if set(listed) != set(by_file):
        missing = sorted(set(listed) - set(by_file))
        extra = sorted(set(by_file) - set(listed))
        raise ValueError(f"benchmark index and tasks/ disagree under {root}: missing {missing}, extra {extra}")
    stage = read_stage(root)
    time_limit_s = float(stage.get("time_limit_s", TASK_TIME_LIMIT_S))
    layouts = []
    records = []
    for entry in entries:
        task_id = str(entry["task_id"])
        payload = by_file[task_id]
        if str(payload.get("layout")) != str(entry.get("layout")):
            raise ValueError(f"task {task_id} layout does not match benchmark.yaml")
        if str(payload.get("background")) != str(entry.get("background")):
            raise ValueError(f"task {task_id} background does not match benchmark.yaml")
        layouts.append(layout_from_payload(payload, time_limit_s=time_limit_s))
        records.append(payload)
    return Benchmark(
        root=root,
        version=str(index.get("version", root.name)),
        task_ids=tuple(listed),
        layouts=tuple(layouts),
        records=tuple(records),
        stage=stage,
    )
