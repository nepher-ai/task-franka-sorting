# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sorting decisions of the planner that need no simulator: goals, pick order, cell spots, and release yaw."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from franka_sorting.benchmark.bundle import load_benchmark

PROJECT = Path(__file__).resolve().parents[1]
ENVHUB = Path(os.environ.get("SORTING_ENVHUB", PROJECT.parent / "envhub" / "environments"))
CELLS = PROJECT / "source/franka_sorting/franka_sorting/tasks/hl_policy/planner/cells.py"

# The planner package imports Isaac Lab, so the pure module is loaded by path.
_spec = importlib.util.spec_from_file_location("planner_cells", CELLS)
cells = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cells
_spec.loader.exec_module(cells)


def _axis_error(a: float, b: float) -> float:
    """Angle [rad] between two undirected axes."""
    return abs(math.remainder(a - b, math.pi))


def test_wrap_half_pi_folds_axes_into_half_open_range():
    for angle in np.linspace(-7.0, 7.0, 57):
        folded = cells.wrap_half_pi(angle)
        assert -math.pi / 2 < folded <= math.pi / 2
        assert _axis_error(folded, angle) < 1e-9


@pytest.mark.parametrize("angle", [-1.2, -0.4, 0.0, 0.7, 1.4])
def test_principal_axes_find_a_turned_rectangle(angle):
    rng = np.random.default_rng(0)
    local = rng.uniform(-1.0, 1.0, size=(4000, 2)) * [0.04, 0.012]
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    found, long_half, short_half = cells.principal_axes(local @ rotation.T + [0.3, -0.1])
    assert _axis_error(found, angle) < 0.03
    assert long_half == pytest.approx(0.04, abs=0.002)
    assert short_half == pytest.approx(0.012, abs=0.002)


@pytest.mark.parametrize("part", np.linspace(-1.5, 1.5, 7))
@pytest.mark.parametrize("axis", [0.0, math.pi / 2, 0.3, -1.0])
def test_release_turn_lines_the_long_side_up_with_the_cell(part, axis):
    turn = cells.release_turn(part, 0.04, 0.012, axis)
    assert -math.pi / 2 < turn <= math.pi / 2
    assert _axis_error(part + turn, axis) < 1e-9


def test_round_parts_are_not_turned():
    assert cells.release_turn(1.0, 0.02, 0.0195, 0.0) == 0.0


def test_cell_axis_follows_the_tray_and_the_deeper_side_of_the_cell():
    assert _axis_error(cells.cell_axis(0.2, (0.05, 0.03)), 0.2) < 1e-9
    assert _axis_error(cells.cell_axis(0.2, (0.03, 0.05)), 0.2 + math.pi / 2) < 1e-9


def test_free_spot_is_central_when_empty_and_avoids_a_part_already_there():
    center, half, extent = np.array([0.5, 0.1]), (0.03, 0.03), (0.01, 0.01)
    spot, gap = cells.free_spot(center, 0.0, half, extent, [], 0.012)
    assert np.linalg.norm(spot - center) < 1e-9 and gap == np.inf
    taken = [(center.copy(), 0.012)]
    spot, gap = cells.free_spot(center, 0.0, half, extent, taken, 0.012)
    assert np.linalg.norm(spot - center) > 0.012
    assert np.all(np.abs(spot - center) <= np.array(half) - np.array(extent) + 1e-9)
    assert gap == pytest.approx(np.linalg.norm(spot - center) - 0.024)


def test_free_spot_reports_a_crowded_cell_with_a_negative_gap():
    center = np.array([0.0, 0.0])
    crowd = [(center + [dx, dy], 0.012) for dx in (-0.01, 0.01) for dy in (-0.01, 0.01)]
    _, gap = cells.free_spot(center, 0.0, (0.02, 0.02), (0.01, 0.01), crowd, 0.012)
    assert gap < 0.0


def test_free_spot_respects_the_tray_heading():
    center = np.array([0.4, -0.2])
    taken = [(center + [0.012, 0.0], 0.01)]
    spot, _ = cells.free_spot(center, math.pi / 2, (0.03, 0.01), (0.005, 0.005), taken, 0.01)
    # A tray turned a quarter has its long cell side along world y, so the spot moves along y.
    assert abs(spot[1] - center[1]) > abs(spot[0] - center[0])


def test_pick_score_takes_roomy_tall_near_fresh_parts_first():
    base = dict(travel=0.3, clearance=0.02, height=0.02, attempts=0)
    assert cells.pick_score(**{**base, "clearance": 0.05}) < cells.pick_score(**base)
    assert cells.pick_score(**{**base, "clearance": -0.01}) == cells.pick_score(**{**base, "clearance": 0.0})
    assert cells.pick_score(**{**base, "height": 0.04}) < cells.pick_score(**base)
    assert cells.pick_score(**{**base, "travel": 0.1}) < cells.pick_score(**base)
    assert cells.pick_score(**{**base, "attempts": 1}) > cells.pick_score(**base)
    # A part that keeps failing falls behind a roomier one, but does not outweigh the clearance cap.
    assert cells.pick_score(**{**base, "attempts": 2}) > cells.pick_score(**{**base, "clearance": 0.05})


@pytest.mark.parametrize("env_id", ["franka-sorting-parts-v0", "franka-sorting-parts-v1"])
def test_every_part_of_a_task_has_one_goal_for_the_cell_of_its_type(env_id):
    root = ENVHUB / env_id
    if not (root / "benchmark.yaml").is_file():
        pytest.skip(f"{env_id} is not built under {ENVHUB}")
    for layout in load_benchmark(root).layouts:
        goals = cells.cell_goals(layout)
        by_slot = layout.by_slot()
        parts = [p.slot for p in layout.placements if p.asset.kind == "item"]
        assert sorted(goal.slot for goal in goals) == sorted(parts)
        for goal in goals:
            assert goal.tray.startswith("tray")
            variant = by_slot[goal.slot].asset.variant
            assert variant in layout.cell_map
            assert goal.cells
