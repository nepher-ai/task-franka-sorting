# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data generation without Isaac Sim: shard plan and resume, quality gates, procedural bundles, writer recovery."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from franka_sorting.benchmark.bundle import load_benchmark  # noqa: E402
from franka_sorting.data.procedural import procedural_tasks, unlink_directory, write_shard_bundle  # noqa: E402
from franka_sorting.data.quality import PANDA_JOINT_LOWER, PANDA_JOINT_UPPER, QualityLimits, check_episode  # noqa: E402
from franka_sorting.data.randomize import RandomizeConfig, draw_camera, draw_lighting  # noqa: E402
from franka_sorting.data.sdg_plan import (  # noqa: E402
    DONE,
    PENDING,
    RUNNING,
    ShardSpec,
    config_digest,
    load_state,
    plan_benchmark_shards,
    remaining_quota,
    save_state,
    start_or_resume,
    summarize,
)

import conftest  # noqa: E402,F401  (stubs the package init)

ENVHUB = Path(__file__).resolve().parents[2] / "envhub" / "environments"
MASTER = ENVHUB / "franka-sorting-parts-v2"


def _episode(steps: int = 60):
    mid = (np.asarray(PANDA_JOINT_LOWER) + np.asarray(PANDA_JOINT_UPPER)) / 2
    ramp = np.linspace(0.0, 0.2, steps)[:, None]
    joints = mid + ramp * np.linspace(-1, 1, 7)
    gripper = np.zeros((steps, 1))
    return np.hstack([joints, gripper]), np.hstack([joints, gripper])


# ------------------------------------------------------------------ quality gates


def test_a_smooth_episode_passes():
    state, action = _episode()
    assert check_episode(state, action) == []


def test_gates_name_each_failure():
    state, action = _episode()
    assert "too_short" in check_episode(state[:5], action[:5])
    assert "too_long" in check_episode(state, action, limits=QualityLimits(max_steps=10))
    bad = action.copy()
    bad[10, 2] = np.nan
    assert check_episode(state, bad) == ["nan"]
    over = state.copy()
    over[:, 0] = PANDA_JOINT_UPPER[0] + 0.5
    assert "joint_limit" in check_episode(over, action)
    jump = action.copy()
    jump[30:, 1] += 0.8
    assert "joint_jump" in check_episode(state, jump)
    grip = action.copy()
    grip[:, 7] = 0.5
    assert "gripper_range" in check_episode(state, grip)


def test_blank_video_is_rejected():
    state, action = _episode()
    blank = np.full((60, 8, 16, 3), 7, dtype=np.uint8)
    noisy = np.random.default_rng(0).integers(0, 255, (60, 8, 16, 3), dtype=np.uint8)
    assert check_episode(state, action, {"cam_wrist": noisy}) == []
    assert check_episode(state, action, {"cam_wrist": blank}) == ["blank_frames:cam_wrist"]


# ------------------------------------------------------------------ randomization


def test_visual_draws_follow_the_seed_and_the_ranges():
    config = RandomizeConfig(camera_jitter=1.0)
    first = draw_lighting(np.random.default_rng(3), config)
    assert first == draw_lighting(np.random.default_rng(3), config)
    assert first != draw_lighting(np.random.default_rng(4), config)
    assert config.dome_scale[0] <= first.dome_scale <= config.dome_scale[1]
    assert all(abs(v) <= h for v, h in zip(first.sphere_offset_m, config.sphere_offset_m))
    camera = draw_camera(np.random.default_rng(3), config)
    assert max(abs(v) for v in camera.shift_m) <= config.camera_pos_m
    assert max(abs(np.rad2deg(camera.rotvec_rad))) <= config.camera_rot_deg
    assert draw_camera(np.random.default_rng(3), RandomizeConfig()).shift_m == (0.0, 0.0, 0.0)
    off = draw_lighting(np.random.default_rng(3), RandomizeConfig(lighting=False))
    assert (off.dome_scale, off.sphere_scale) == (1.0, 1.0)


# ------------------------------------------------------------------ shard plan and resume


ROWS = [(f"t{i:02d}", "bg_a" if i < 5 else "bg_b") for i in range(8)]


def test_shards_split_by_background_and_size():
    shards = plan_benchmark_shards(ROWS, "env", episodes_per_task=2, tasks_per_shard=3, seed=1)
    assert [s.tasks for s in shards] == [["t00", "t01", "t02"], ["t03", "t04"], ["t05", "t06", "t07"]]
    assert [s.background for s in shards] == ["bg_a", "bg_a", "bg_b"]
    assert [s.index for s in shards] == [0, 1, 2]
    assert len({s.layout_seed() for s in shards}) == 3
    retry = ShardSpec(**{**shards[0].__dict__, "attempt": 1})
    assert retry.layout_seed() != shards[0].layout_seed()


def test_remaining_quota_counts_successes_of_the_shard_only():
    shard = plan_benchmark_shards(ROWS, "env", 3, 3, 1)[0]
    rows = [
        {"shard": 0, "task_id": "t00", "success": True},
        {"shard": 0, "task_id": "t00", "success": True},
        {"shard": 0, "task_id": "t01", "success": False},
        {"shard": 1, "task_id": "t02", "success": True},
        {"shard": 0, "task_id": "other", "success": True},
    ]
    assert remaining_quota(rows, shard) == {"t00": 1, "t01": 3, "t02": 3}
    assert remaining_quota(rows, shard, count_failed=True)["t01"] == 2


def test_state_resumes_and_reroutes_an_interrupted_shard(tmp_path):
    config = {"episodes": 2}
    shards = plan_benchmark_shards(ROWS, "env", 2, 3, 1)
    state = start_or_resume(tmp_path, config, shards)
    state.shards[0].status = DONE
    state.shards[1].status = RUNNING
    save_state(tmp_path, state)
    resumed = start_or_resume(tmp_path, config, plan_benchmark_shards(ROWS, "env", 2, 3, 1))
    assert [s.status for s in resumed.shards] == [DONE, PENDING, PENDING]
    assert [s.attempt for s in resumed.shards] == [0, 1, 0]
    assert resumed.next_shard().index == 1
    assert load_state(tmp_path).counts() == {"pending": 2, "running": 0, "done": 1, "failed": 0}
    with pytest.raises(ValueError, match="other arguments"):
        start_or_resume(tmp_path, {"episodes": 3}, shards)
    assert config_digest({"b": 1, "a": 2}) == config_digest({"a": 2, "b": 1})
    assert not list(tmp_path.glob(".*.tmp"))


def test_summary_counts_episodes_by_configuration():
    rows = [
        {"task_id": "a", "container": "bin", "tray": "t1", "length": 10, "success": True, "progress": 1.0},
        {"task_id": "a", "container": "bin", "tray": "t1", "length": 20, "success": False, "progress": 0.5},
    ]
    report = summarize(rows)
    assert (report["episodes"], report["frames"], report["success"]) == (2, 30, 1)
    assert report["by_container"] == {"bin": 2} and report["mean_progress"] == 0.75


# ------------------------------------------------------------------ procedural shards


@pytest.fixture(scope="module")
def library():
    benchmark = load_benchmark(MASTER)
    return benchmark.library, benchmark.stage


def test_procedural_tasks_are_valid_and_reproducible(library):
    lib, stage = library
    kwargs = dict(shard=2, count=6, seed=5, background="bg")
    first = procedural_tasks(lib, stage, **kwargs)
    again = procedural_tasks(lib, stage, **kwargs)
    assert [p["task_id"] for p in first] == [p["task_id"] for p in again]
    assert first[0]["placements"] == again[0]["placements"]
    assert len({p["task_id"] for p in first}) == 6
    assert all(p["background"] == "bg" for p in first)
    other_shard = procedural_tasks(lib, stage, **{**kwargs, "shard": 3})
    assert {p["layout"] for p in first} != {p["layout"] for p in other_shard}


def test_pairs_are_balanced_across_shards(library):
    lib, stage = library
    seen = [
        p["layout"]
        for shard in range(5)
        for p in procedural_tasks(lib, stage, shard=shard, count=10, seed=1, background="bg")
    ]
    assert len(seen) == 50
    assert len(set(seen)) >= 5


def test_shard_bundle_loads_like_any_benchmark(library, tmp_path):
    lib, stage = library
    payloads = procedural_tasks(lib, stage, shard=0, count=3, seed=1, background=stage["backgrounds"][0]["name"])
    bundle = write_shard_bundle(MASTER, tmp_path / "shard", payloads, stage["backgrounds"][0]["name"])
    benchmark = load_benchmark(bundle)
    assert benchmark.task_ids == tuple(p["task_id"] for p in payloads)
    assert (bundle / "assets" / "parts").is_dir() and (bundle / "assets" / "stage.json").is_file()
    assert all(layout.instruction for layout in benchmark.layouts)
    write_shard_bundle(MASTER, bundle, payloads, stage["backgrounds"][0]["name"])  # idempotent
    for link in (bundle / "assets").iterdir():
        if link.is_dir():
            unlink_directory(link)
    assert (MASTER / "assets" / "parts").is_dir()
