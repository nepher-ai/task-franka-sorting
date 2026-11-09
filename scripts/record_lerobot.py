# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic data generation: record planner demonstrations as a LeRobot v2.1 dataset.

This script is the orchestrator. It never starts Isaac Sim itself. It splits the job into shards and runs
``record_shard.py`` for each one in its own process, so every stage is built fresh with its own assets and a crash
costs one shard. ``meta/sdg_state.json`` records the shards, so running the same command again resumes the run.

Sources:

``--source benchmark``    The tasks of an EnvHub bundle (``--task``, ``--task_index``). ``--randomize poses`` (the
                          default) draws a new overlap-free layout for every episode on the bodies of the task.
                          A selection that spans several backgrounds becomes one shard per background.
``--source procedural``   Fresh tasks for every shard: random container, tray, table, part types, cell map and count
                          on one background, built from the asset library bundle (``franka-sorting-parts-v2``).

Run it with the Isaac Lab python::

    isaaclab.bat -p source/task-franka-sorting/scripts/record_lerobot.py --env_id franka-sorting-parts-v2 ^
        --task_index 0~49 --episodes 4 --cameras --output datasets/sorting_v2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "source" / "franka_sorting" / "franka_sorting"

# The orchestrator only needs the pure modules: skip the package init, which loads Isaac Lab.
if "franka_sorting" not in sys.modules:
    _stub = types.ModuleType("franka_sorting")
    _stub.__path__ = [str(PACKAGE)]
    sys.modules["franka_sorting"] = _stub

from franka_sorting.benchmark.bundle import ALL_TASKS, DEFAULT_ENV_ID, load_benchmark, resolve_bundle  # noqa: E402
from franka_sorting.benchmark_select import parse_task_indices, select_tasks  # noqa: E402
from franka_sorting.data.atomic import atomic_write_text  # noqa: E402
from franka_sorting.data.procedural import procedural_tasks, write_shard_bundle  # noqa: E402
from franka_sorting.data.randomize import RandomizeConfig  # noqa: E402
from franka_sorting.data.sdg_plan import (  # noqa: E402
    DONE,
    FAILED,
    PENDING,
    REPORT_FILE,
    RUNNING,
    STATE_FILE,
    SdgState,
    ShardSpec,
    plan_benchmark_shards,
    save_state,
    start_or_resume,
    summarize,
)
from franka_sorting.scene.cameras import OBSERVATION_CAMERAS, selected_cameras  # noqa: E402

LIBRARY_ID = "franka-sorting-parts-v2"
WORKER = Path(__file__).resolve().with_name("record_shard.py")

parser = argparse.ArgumentParser(
    description="Record planner demonstrations as a LeRobot dataset, in resumable shards.",
    epilog="Other arguments (--device, --visualizer, ...) go to Isaac Lab in every shard. The shards run without a "
    "viewport unless --gui is given.",
)
parser.add_argument("--source", choices=("benchmark", "procedural"), default="benchmark")
parser.add_argument("--env_id", type=str, default=None, help="EnvHub bundle. Benchmark source: the package default.")
parser.add_argument("--task", type=str, default=None, help="Benchmark task id, or comma-separated ids. Default: all.")
parser.add_argument(
    "--task_index",
    type=str,
    nargs="+",
    default=None,
    metavar="SPEC",
    help="Tasks by benchmark.yaml index: 137, a range 16~31 (or 16-31), or a comma list 0,5,16~31. Not with --task.",
)
parser.add_argument("--list_tasks", action="store_true", help="Print index, task id and background, then exit.")
parser.add_argument("--output", type=str, default=None, help="Dataset directory. An existing dataset is extended.")
parser.add_argument("--episodes", type=int, default=8, help="Successful episodes to record per task.")
parser.add_argument("--tasks_per_shard", type=int, default=None, help="Tasks per stage. Default 16, procedural 8.")
parser.add_argument("--shards", type=int, default=4, help="Procedural source: number of shards (fresh stages).")
parser.add_argument(
    "--randomize",
    choices=("poses", "none"),
    default="poses",
    help="poses: a new random layout for every episode. none: the published layout every time.",
)
parser.add_argument("--seed", type=int, default=1, help="Base seed of the run.")
parser.add_argument(
    "--cameras",
    nargs="*",
    default=None,
    metavar="NAME",
    help=(
        f"Cameras to record: {', '.join(OBSERVATION_CAMERAS)}. "
        "Pass the flag with no names to record every camera. Omit the flag to record none."
    ),
)
parser.add_argument("--image_hw", type=int, nargs=2, default=(180, 320), help="Camera frame size [pixels]: H W.")
parser.add_argument("--num_envs", type=int, default=0, help="Parallel envs per shard. 0: 16 (8 with cameras).")
parser.add_argument("--max_steps", type=int, default=2600, help="Control steps after which an episode is cut.")
parser.add_argument(
    "--max_tries_factor", type=float, default=4.0, help="A shard stops after this many times its quota."
)
parser.add_argument("--min_steps", type=int, default=30, help="Shortest episode to keep [control steps].")
parser.add_argument("--keep_failed", action="store_true", help="Also save episodes that fail the task, labelled.")
parser.add_argument("--checkpoint", type=str, default=None, help="Tracker checkpoint file.")
parser.add_argument("--arm_noise", type=float, default=0.05, help="Half range [rad] of the arm start noise.")
parser.add_argument("--no_lighting", action="store_true", help="Keep the light intensities and poses of the stage.")
parser.add_argument("--camera_jitter", type=float, default=0.0, help="Scale of the camera extrinsics jitter (0 off).")
parser.add_argument("--retries", type=int, default=2, help="Extra attempts of a shard that stops short or crashes.")
parser.add_argument("--restart", action="store_true", help="Discard the saved plan. Saved episodes stay.")
parser.add_argument("--dry_run", action="store_true", help="Print the plan and exit.")
parser.add_argument("--gui", action="store_true", help="Open the Kit viewport in the shards (--visualizer kit).")
args, passthrough = parser.parse_known_args()


def die(message: str) -> None:
    parser.error(message)


def plan() -> tuple[list[ShardSpec], dict]:
    """Shards and the arguments that decide what the run produces."""
    if args.episodes < 1:
        die("--episodes must be at least 1")
    env_id = args.env_id or (LIBRARY_ID if args.source == "procedural" else DEFAULT_ENV_ID)
    randomize = args.randomize == "poses"
    config = {
        "source": args.source,
        "env_id": env_id,
        "episodes": args.episodes,
        "seed": args.seed,
        "randomize": randomize,
        "cameras": list(selected_cameras(OBSERVATION_CAMERAS if args.cameras == [] else args.cameras)),
        "image_hw": list(args.image_hw),
        "arm_noise": args.arm_noise,
        "lighting": not args.no_lighting,
        "camera_jitter": args.camera_jitter,
    }
    if args.source == "benchmark":
        benchmark = load_benchmark(resolve_bundle(env_id))
        if args.task is not None and args.task_index is not None:
            die("--task and --task_index cannot be used together")
        if args.task_index is not None:
            try:
                indices = parse_task_indices(args.task_index, len(benchmark.task_ids))
            except ValueError as error:
                die(str(error))
            selector = ",".join(benchmark.task_ids[index] for index in indices)
        else:
            selector = args.task or ALL_TASKS
        try:
            tasks = select_tasks(benchmark, selector)
        except FileNotFoundError as error:
            die(str(error))
        per_shard = args.tasks_per_shard or 16
        config.update(tasks=list(tasks), tasks_per_shard=per_shard)
        rows = [(task, benchmark.task(task).background) for task in tasks]
        shards = plan_benchmark_shards(rows, env_id, args.episodes, per_shard, args.seed, randomize)
        return shards, config
    master = resolve_bundle(env_id)
    benchmark = load_benchmark(master)
    per_shard = args.tasks_per_shard or 8
    config.update(shards=args.shards, tasks_per_shard=per_shard)
    names = [str(record["name"]) for record in benchmark.stage["backgrounds"]]
    order = [names[i] for i in np.random.default_rng(args.seed).permutation(len(names))]
    shards = []
    for index in range(args.shards):
        background = order[index % len(order)]
        payloads = procedural_tasks(
            benchmark.library, benchmark.stage, shard=index, count=per_shard, seed=args.seed, background=background
        )
        bundle = Path(args.output) / "_sdg" / "bundles" / f"shard_{index:03d}"
        write_shard_bundle(master, bundle, payloads, background)
        shards.append(
            ShardSpec(
                index=index,
                source="procedural",
                bundle=str(bundle.resolve()),
                tasks=[payload["task_id"] for payload in payloads],
                background=background,
                episodes_per_task=args.episodes,
                seed=args.seed,
                randomize=randomize,
            )
        )
    return shards, config


def episode_rows(output: Path) -> list[dict]:
    path = output / "meta" / "episodes.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def worker_options(config: dict) -> dict:
    return {
        "cameras": config["cameras"],
        "image_hw": config["image_hw"],
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "max_tries_factor": args.max_tries_factor,
        "keep_failed": args.keep_failed,
        "checkpoint": args.checkpoint,
        "arm_noise_rad": args.arm_noise,
        "randomize": RandomizeConfig(lighting=not args.no_lighting, camera_jitter=args.camera_jitter).as_dict(),
        "quality": {"min_steps": args.min_steps, "max_steps": args.max_steps},
    }


def run_shard(shard: ShardSpec, state: SdgState, output: Path, options: dict) -> None:
    """Run one shard until it completes or its attempts run out. Updates and saves the state."""
    work = output / "_sdg"
    work.mkdir(parents=True, exist_ok=True)
    for attempt_number in range(args.retries + 1):
        spec_path = work / f"shard_{shard.index:03d}.json"
        result_path = work / f"shard_{shard.index:03d}.result.json"
        result_path.unlink(missing_ok=True)
        atomic_write_text(
            spec_path, json.dumps({"shard": asdict(shard), "options": options, "output": str(output.resolve())})
        )
        shard.status = RUNNING
        save_state(output / "meta", state)
        command = [sys.executable, str(WORKER), "--spec", str(spec_path), *passthrough]
        if args.gui and not {"--visualizer", "--viz"} & set(passthrough):
            command += ["--visualizer", "kit"]
        print(f"[SDG] shard {shard.index} attempt {shard.attempt}: {len(shard.tasks)} tasks", flush=True)
        started = time.time()
        code = subprocess.call(command)
        shard.elapsed_s += time.time() - started
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        if result is not None:
            shard.tried += int(result["tried"])
            for reason, count in result["rejected"].items():
                shard.rejected[reason] = shard.rejected.get(reason, 0) + int(count)
        shard.saved = sum(1 for row in episode_rows(output) if row.get("shard") == shard.index and row.get("success"))
        if result is not None and result["complete"] and code == 0:
            shard.status, shard.error = DONE, ""
            save_state(output / "meta", state)
            return
        shard.error = f"exit code {code}" if result is None or code else "quota not reached"
        shard.status = FAILED
        shard.attempt += 1
        save_state(output / "meta", state)
        print(f"[SDG] shard {shard.index} stopped: {shard.error}", flush=True)


def main() -> None:
    if args.list_tasks:
        benchmark = load_benchmark(resolve_bundle(args.env_id or DEFAULT_ENV_ID))
        for index, layout in enumerate(benchmark.layouts):
            print(f"{index:4d}  {layout.task_id}  {layout.background}  {layout.instruction}")
        return
    if args.output is None:
        die("--output is required")
    output = Path(args.output)
    if args.restart:
        (output / "meta" / STATE_FILE).unlink(missing_ok=True)
        shutil.rmtree(output / "_sdg", ignore_errors=True)
    shards, config = plan()
    try:
        state = start_or_resume(output / "meta", config, shards)
    except ValueError as error:
        die(str(error))
    counts = state.counts()
    print(f"[SDG] {len(state.shards)} shards: {counts}", flush=True)
    for shard in state.shards:
        print(f"  shard {shard.index:3d} {shard.status:8s} {shard.background:34s} {len(shard.tasks)} tasks")
    if args.dry_run:
        return
    options = worker_options(config)
    try:
        for shard in state.shards:
            if shard.status == DONE:
                continue
            if shard.status == FAILED:
                shard.status = PENDING
            run_shard(shard, state, output, options)
    except KeyboardInterrupt:
        print("[SDG] interrupted; run the same command to resume", flush=True)
    rows = episode_rows(output)
    report = summarize(rows, state)
    atomic_write_text(output / "meta" / REPORT_FILE, json.dumps(report, indent=2))
    final = state.counts()
    print(
        f"[RESULT] {report['episodes']} episodes ({report['success']} successful), {report['frames']} frames, "
        f"shards {final}; report: {output / 'meta' / REPORT_FILE}",
        flush=True,
    )
    if final[DONE] != len(state.shards):
        sys.exit(1)


if __name__ == "__main__":
    main()
