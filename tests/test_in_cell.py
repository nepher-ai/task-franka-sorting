# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ``in_cell`` success op: parsing, geometry, hands-off rule, latched progress, and the benchmark loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from franka_sorting.benchmark.bundle import load_benchmark, resolve_bundle
from franka_sorting.benchmark.predicates import in_cell, task_progress
from franka_sorting.benchmark.spec import spec_from_success
from franka_sorting.benchmark.stage_plan import (
    PARK_COLUMNS,
    PARK_Z_M,
    entity_pose,
    parking_position,
    plan_stage,
)
from franka_sorting.benchmark.success import task_satisfied

ENVHUB = Path(__file__).resolve().parents[2] / "envhub" / "environments"
CELLS = np.array([[-0.1, 0.05], [0.1, 0.05], [-0.1, -0.05], [0.1, -0.05]])  # a 2 x 2 tray, tray frame [m]
HALF = np.array([0.045, 0.04])
RIM = 0.04
QUARTER_TURN = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))


def _tray(quat=(1.0, 0.0, 0.0, 0.0), pos=(0.5, 0.1, 0.0)) -> dict:
    return {
        "pos": np.array([pos]),
        "quat": np.array([quat]),
        "half": np.array([0.2, 0.1, 0.02]),
        "cell_centers": CELLS,
        "cell_half": HALF,
        "rim_z": RIM,
    }


def _part(x: float, y: float, z: float = 0.012) -> dict:
    return {"pos": np.array([[x, y, z]]), "quat": np.array([[1.0, 0.0, 0.0, 0.0]]), "half": np.array([0.01] * 3)}


def _spec(cells=(0, 3)):
    checks = [{"op": "in_cell", "object": f"p{i}", "container": "tray", "cell": c} for i, c in enumerate(cells)]
    slots = {f"p{i}": f"p{i}" for i in range(len(cells))} | {"tray": "tray"}
    return spec_from_success("t", {"op": "all", "checks": checks}, 60.0, slots)


def test_in_cell_geometry_in_a_rotated_tray():
    # A quarter turn about +z maps tray (x, y) to world (-y, x).
    tray = _tray(QUARTER_TURN)
    origin = tray["pos"][0]
    for index, (cx, cy) in enumerate(CELLS):
        world = origin[:2] + np.array([-cy, cx])
        point = np.array([[*world, 0.012]])
        for other, cell in enumerate(CELLS):
            hit = in_cell(point, tray["pos"], tray["quat"], cell, HALF, RIM)
            assert bool(hit[0]) == (index == other)
    inside_edge = origin[:2] + np.array([-(CELLS[0][1] + HALF[1] - 1e-3), CELLS[0][0]])
    assert in_cell(np.array([[*inside_edge, 0.01]]), tray["pos"], tray["quat"], CELLS[0], HALF, RIM)[0]
    outside_edge = origin[:2] + np.array([-(CELLS[0][1] + HALF[1] + 1e-3), CELLS[0][0]])
    assert not in_cell(np.array([[*outside_edge, 0.01]]), tray["pos"], tray["quat"], CELLS[0], HALF, RIM)[0]


def test_in_cell_requires_the_center_below_the_rim_and_above_the_tray_bottom():
    tray = _tray()
    x, y = tray["pos"][0][:2] + CELLS[1]
    assert in_cell(np.array([[x, y, 0.02]]), tray["pos"], tray["quat"], CELLS[1], HALF, RIM)[0]
    assert not in_cell(np.array([[x, y, RIM + 0.01]]), tray["pos"], tray["quat"], CELLS[1], HALF, RIM)[0]
    assert not in_cell(np.array([[x, y, -0.01]]), tray["pos"], tray["quat"], CELLS[1], HALF, RIM)[0]


def test_spec_parses_cells_and_rejects_a_missing_cell():
    spec = _spec()
    goals = spec.check.children[0], spec.check.children[1]
    assert [g.goals[0].cell for g in goals] == [0, 3]
    assert set(spec.check.grasp) == {"p0", "p1"}  # a moved op: each part has to be grasped and released
    with pytest.raises(ValueError, match="in_cell"):
        spec_from_success(
            "t",
            {"op": "all", "checks": [{"op": "in_cell", "object": "p", "container": "tray"}]},
            60.0,
            {"p": "p", "tray": "tray"},
        )


def _poses(a: tuple[float, float], b: tuple[float, float]) -> dict:
    tray = _tray()
    origin = tray["pos"][0][:2]
    return {"tray": tray, "p0": _part(*(origin + a)), "p1": _part(*(origin + b))}


def test_at_least_quota_counts_parts_in_a_cell_of_their_type():
    """Two parts in one typed cell meet a quota of 2. A part left outside does not fail it."""
    checks = [
        {"op": "in_typed_cell", "object": f"p{i}", "cells": [{"container": "tray", "cell": 0}]}
        for i in range(3)
    ]
    slots = {f"p{i}": f"p{i}" for i in range(3)} | {"tray": "tray"}
    spec = spec_from_success("t", {"op": "at_least", "k": 2, "checks": checks}, 30.0, slots)
    assert spec.time_limit_s == 30.0
    assert spec.check.logical == "at_least" and spec.check.k == 2
    origin = _tray()["pos"][0][:2]
    inside = tuple(origin + CELLS[0])
    outside = (0.0, 0.4)

    def poses(*where: tuple[float, float]) -> dict:
        return {"tray": _tray(), **{f"p{i}": _part(*xy) for i, xy in enumerate(where)}}

    free = {f"p{i}": False for i in range(3)}
    assert not task_satisfied(spec, poses(inside, outside, outside), free, {})
    assert task_satisfied(spec, poses(inside, inside, outside), free, {})
    assert task_satisfied(spec, poses(inside, inside, inside), free, {})


def test_success_needs_every_part_in_its_own_cell_and_the_hand_away():
    spec = _spec((0, 3))
    right = _poses(tuple(CELLS[0]), tuple(CELLS[3]))
    free = {"p0": False, "p1": False}
    assert task_satisfied(spec, right, free, {})
    assert not task_satisfied(spec, right, {"p0": True, "p1": False}, {})  # still held
    swapped = _poses(tuple(CELLS[3]), tuple(CELLS[0]))
    assert not task_satisfied(spec, swapped, free, {})  # a part in the wrong cell does not count
    one = _poses(tuple(CELLS[0]), tuple(CELLS[1]))
    assert not task_satisfied(spec, one, free, {})


def _progress(spec, poses, held, contact, latched):
    return task_progress(spec, poses, held, contact, latched, [], gripper_open=True)


def test_progress_is_the_fraction_sorted_and_unlatches_when_a_part_leaves():
    spec = _spec((0, 3))
    sorted_one = _poses(tuple(CELLS[0]), (0.0, 0.3))
    score, latched, _ = _progress(spec, sorted_one, {"p0": True}, {"p0": False}, {})
    assert score == pytest.approx(0.5)  # p0 was held, let go in its cell
    score, latched, _ = _progress(spec, sorted_one, {}, {}, latched)
    assert score == pytest.approx(0.5)
    both = _poses(tuple(CELLS[0]), tuple(CELLS[3]))
    score, latched, _ = _progress(spec, both, {"p1": True}, {"p1": False}, latched)
    assert score == pytest.approx(1.0)
    knocked = _poses((0.0, 0.3), tuple(CELLS[3]))  # p0 left its cell
    score, latched, _ = _progress(spec, knocked, {}, {}, latched)
    assert score == pytest.approx(0.5)
    assert not any(key.startswith("in_cell:p0") for key, value in latched.items() if value)


def test_progress_ignores_a_part_that_was_never_grasped():
    spec = _spec((0,))
    placed = {"tray": _tray(), "p0": _part(*(_tray()["pos"][0][:2] + CELLS[0]))}
    score, _, _ = _progress(spec, placed, {}, {}, {})
    assert score == 0.0  # it has to be picked up and released, not merely found there


def _bundle(env_id: str) -> Path:
    path = ENVHUB / env_id
    if not (path / "benchmark.yaml").is_file():
        pytest.skip(f"{env_id} is not built")
    return path


def test_benchmark_loader_reads_sorting_fields_and_stage_time_limit():
    bundle = _bundle("franka-sorting-parts-v0")
    assert resolve_bundle(str(bundle)) == bundle.resolve()
    benchmark = load_benchmark(bundle)
    stage = json.loads((bundle / "assets" / "stage.json").read_text())
    assert len(benchmark.layouts) == 16
    for layout in benchmark.layouts:
        assert layout.spec.time_limit_s == stage["time_limit_s"]
        slots = layout.by_slot()
        assert slots["container"].asset.kind == "container"
        trays = [placement for placement in layout.placements if placement.asset.kind == "tray"]
        assert trays and all(placement.kinematic for placement in trays)
        assert slots["container"].kinematic
        goals = [child.goals[0] for child in layout.spec.check.children]
        assert layout.spec.check.logical == "at_least"
        assert all(goal.op == "in_typed_cell" for goal in goals)
        for goal in goals:
            assert slots[goal.slot].asset.variant in layout.cell_map
            assert goal.places


def test_stage_plan_keeps_container_and_tray_first_and_parks_on_distinct_slots():
    benchmark = load_benchmark(_bundle("franka-sorting-parts-v0"))
    plan = plan_stage(benchmark.layouts, benchmark.stage, num_envs=16)
    assert plan.entities[:2] == ("obj0", "obj1")
    assert all(plan.variants[0][choice].kinematic for choice in plan.choices[0])
    assert all(plan.variants[1][choice].kinematic for choice in plan.choices[1])
    slots = [parking_position(i) for i in range(len(plan.entities))]
    assert len(set(slots)) == len(slots)
    assert all(abs(z - PARK_Z_M) < 0.1 for _, _, z in slots)
    assert parking_position(PARK_COLUMNS)[1] != parking_position(0)[1]
    for env, layout in enumerate(benchmark.layouts):
        used = len(layout.placements)
        for index in range(used, len(plan.entities)):
            position, _ = entity_pose(plan, benchmark.layouts, env, index)
            assert position == parking_position(index)
        position, _ = entity_pose(plan, benchmark.layouts, env, 2)
        assert position == layout.placements[2].position_m


def test_a_tray_part_stays_in_its_cell_when_the_tray_turns():
    import math

    from franka_sorting.benchmark.episode import _follow_tray
    from franka_sorting.benchmark.task import ObjectAsset, Placement
    from franka_sorting.layout.sampler import quat_from_yaw

    tray = Placement(
        slot="tray_0",
        asset=ObjectAsset("tray_0", (0.2, 0.2, 0.07), "tray", "assets/t.usd", "parts_tray"),
        position_m=(0.5, 0.1, 0.0),
        role="tray_0",
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    local = CELLS[0]
    part = Placement(
        slot="part_20",
        asset=ObjectAsset("part_20", (0.04, 0.04, 0.04), "item", "assets/p.usd", "cap_screw"),
        position_m=(0.5 + float(local[0]), 0.1 + float(local[1]), 0.02),
        role="part_20",
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        home="tray",
        tray_slot="tray_0",
        cell=0,
    )
    position, _ = _follow_tray(part, tray, (0.2, -0.3, math.pi / 2), 0.0)
    assert in_cell(
        np.asarray(position, dtype=np.float64).reshape(1, 3),
        np.array([0.2, -0.3, 0.0]),
        np.asarray(quat_from_yaw(math.pi / 2)),
        local,
        HALF,
        RIM,
    )[0]
