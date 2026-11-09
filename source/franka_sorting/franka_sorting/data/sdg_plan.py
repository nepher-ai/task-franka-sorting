# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shard plan and resume state of a data-generation run. Pure Python, no simulator.

A run is a list of shards. One shard is one Isaac Lab process on one stage: a set of tasks that share a background,
each with an episode quota. The state file records which shards finished, so a rerun with the same arguments
continues where the last one stopped. A shard that was cut short recomputes its remaining quota from the dataset
itself, which holds every episode that was saved before the stop.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .atomic import atomic_write_text

STATE_VERSION = 1
STATE_FILE = "sdg_state.json"
REPORT_FILE = "sdg_report.json"

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


@dataclass
class ShardSpec:
    """One stage worth of episodes."""

    index: int
    source: str
    """``benchmark`` or ``procedural``."""
    bundle: str
    """EnvHub id or directory of the bundle the shard's stage loads."""
    tasks: list[str]
    background: str
    episodes_per_task: int
    seed: int
    """Base seed of the shard's layouts and randomization. A retry changes it through ``attempt``."""
    randomize: bool = True
    attempt: int = 0
    status: str = PENDING
    saved: int = 0
    """Episodes of this shard in the dataset."""
    tried: int = 0
    """Episodes the shard finished, saved or not."""
    rejected: dict[str, int] = field(default_factory=dict)
    """Count per reason of the episodes that failed a gate, over all attempts."""
    elapsed_s: float = 0.0
    error: str = ""

    @property
    def target(self) -> int:
        return self.episodes_per_task * len(self.tasks)

    def layout_seed(self) -> int:
        """Seed that the env draws its layouts from, different for every shard and attempt."""
        return int(hashlib.sha256(f"{self.seed}:{self.index}:{self.attempt}".encode()).hexdigest()[:8], 16)


def plan_benchmark_shards(
    rows: Sequence[tuple[str, str]],
    env_id: str,
    episodes_per_task: int,
    tasks_per_shard: int,
    seed: int,
    randomize: bool = True,
) -> list[ShardSpec]:
    """Shards over ``rows`` of ``(task_id, background)``, one background per shard, in index order.

    A selection that spans several backgrounds splits into one or more shards per background, each with at most
    ``tasks_per_shard`` tasks.
    """
    if episodes_per_task < 1 or tasks_per_shard < 1:
        raise ValueError("episodes_per_task and tasks_per_shard must be at least 1")
    groups: dict[str, list[str]] = {}
    for task_id, background in rows:
        groups.setdefault(background, []).append(task_id)
    shards: list[ShardSpec] = []
    for background, tasks in groups.items():
        for first in range(0, len(tasks), tasks_per_shard):
            shards.append(
                ShardSpec(
                    index=len(shards),
                    source="benchmark",
                    bundle=env_id,
                    tasks=tasks[first : first + tasks_per_shard],
                    background=background,
                    episodes_per_task=episodes_per_task,
                    seed=seed,
                    randomize=randomize,
                )
            )
    return shards


def remaining_quota(episodes: Iterable[Mapping], shard: ShardSpec, count_failed: bool = False) -> dict[str, int]:
    """Episodes still missing per task of ``shard``, from the ``episodes.jsonl`` rows of the dataset.

    Args:
        episodes: Rows of ``meta/episodes.jsonl``.
        shard: Shard whose quota is measured.
        count_failed: Count episodes that were kept with ``success`` false too. Off by default, so a run that keeps
            failures still collects its quota of successes.
    """
    saved: Counter[str] = Counter()
    for row in episodes:
        if row.get("shard") != shard.index or row.get("task_id") not in shard.tasks:
            continue
        if count_failed or row.get("success", True):
            saved[row["task_id"]] += 1
    return {task: max(shard.episodes_per_task - saved[task], 0) for task in shard.tasks}


@dataclass
class SdgState:
    """Contents of ``meta/sdg_state.json``."""

    digest: str
    config: dict
    shards: list[ShardSpec] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def next_shard(self) -> ShardSpec | None:
        """First shard that is not done, or ``None``."""
        return next((shard for shard in self.shards if shard.status != DONE), None)

    def counts(self) -> dict[str, int]:
        counts = Counter(shard.status for shard in self.shards)
        return {key: counts.get(key, 0) for key in (PENDING, RUNNING, DONE, FAILED)}


def config_digest(config: Mapping) -> str:
    """Short hash of the arguments that decide what a run produces."""
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def save_state(directory: str | Path, state: SdgState) -> None:
    """Write ``sdg_state.json`` atomically."""
    state.updated = time.time()
    payload = {
        "version": STATE_VERSION,
        "digest": state.digest,
        "config": state.config,
        "created": state.created,
        "updated": state.updated,
        "shards": [asdict(shard) for shard in state.shards],
    }
    atomic_write_text(Path(directory) / STATE_FILE, json.dumps(payload, indent=2, default=str))


def load_state(directory: str | Path) -> SdgState | None:
    """The state saved in ``directory``, or ``None`` when there is none."""
    path = Path(directory) / STATE_FILE
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != STATE_VERSION:
        raise ValueError(f"{path} has state version {payload.get('version')}, expected {STATE_VERSION}")
    return SdgState(
        digest=payload["digest"],
        config=payload["config"],
        shards=[ShardSpec(**shard) for shard in payload["shards"]],
        created=payload.get("created", 0.0),
        updated=payload.get("updated", 0.0),
    )


def start_or_resume(directory: str | Path, config: Mapping, shards: list[ShardSpec]) -> SdgState:
    """Load the state of ``directory`` or create it.

    A shard left ``running`` by an interrupted run goes back to ``pending`` with its attempt raised, so the retry
    draws new layouts instead of repeating the ones that were cut short.

    Raises:
        ValueError: The saved run used other arguments.
    """
    digest = config_digest(config)
    state = load_state(directory)
    if state is None:
        state = SdgState(digest=digest, config=dict(config), shards=shards)
        save_state(directory, state)
        return state
    if state.digest != digest:
        raise ValueError(
            f"{Path(directory) / STATE_FILE} belongs to a run with other arguments. Rerun with the same arguments, "
            "use another --output, or pass --restart to discard the saved plan (episodes already saved stay)."
        )
    for shard in state.shards:
        if shard.status == RUNNING:
            shard.status = PENDING
            shard.attempt += 1
    save_state(directory, state)
    return state


def summarize(episodes: Sequence[Mapping], state: SdgState | None = None) -> dict:
    """Report of a dataset: counts per task, configuration, background, and rejected-episode reasons."""
    by = {key: Counter() for key in ("task_id", "container", "tray", "table", "background")}
    for row in episodes:
        for key, counter in by.items():
            counter[str(row.get(key, "?"))] += 1
    report = {
        "episodes": len(episodes),
        "frames": sum(int(row.get("length", 0)) for row in episodes),
        "success": sum(1 for row in episodes if row.get("success", True)),
        "mean_progress": round(sum(float(row.get("progress", 0.0)) for row in episodes) / max(len(episodes), 1), 4),
        "by_task": dict(sorted(by["task_id"].items())),
        "by_container": dict(sorted(by["container"].items())),
        "by_tray": dict(sorted(by["tray"].items())),
        "by_table": dict(sorted(by["table"].items())),
        "by_background": dict(sorted(by["background"].items())),
    }
    if state is not None:
        report["shards"] = state.counts()
        rejected: Counter[str] = Counter()
        for shard in state.shards:
            rejected.update(shard.rejected)
        report["tried"] = sum(shard.tried for shard in state.shards)
        report["rejected"] = dict(sorted(rejected.items()))
    return report
