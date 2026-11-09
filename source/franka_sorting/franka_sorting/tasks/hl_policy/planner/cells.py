# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sorting decisions that need no simulator: goals, pick order, cell spots, and release yaw."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from franka_sorting.benchmark.spec import Check
from franka_sorting.benchmark.task import TaskLayout

# A part whose footprint is this elongated has a long axis that is worth aligning with the cell.
ALIGN_ASPECT = 1.15
# Spots in a cell are searched on a grid this fine, per axis.
SPOT_GRID = 9
# A spot this far [m] from every other part is as good as any other.
SPOT_ROOM_M = 0.02
# Pick score weights: lower is picked first.
TRAVEL_WEIGHT = 0.5
CLEARANCE_WEIGHT = 4.0
HEIGHT_WEIGHT = 2.0
ATTEMPT_PENALTY = 0.3
MAX_CLEARANCE_M = 0.05


@dataclass(frozen=True)
class CellGoal:
    """Put ``slot`` into ``cell`` of the tray at slot ``tray``.

    ``cells`` lists every ``(tray slot, cell)`` the part's type may use. The planner picks one that is free.
    """

    slot: str
    tray: str
    cell: int
    cells: tuple[tuple[str, int], ...] = ()


def cell_goals(layout: TaskLayout) -> list[CellGoal]:
    """The cell goals of a layout's success expression, in expression order."""
    goals: list[CellGoal] = []

    def walk(check: Check) -> None:
        for goal in check.goals:
            if goal.op not in ("in_cell", "in_typed_cell"):
                continue
            places = goal.places or ((goal.target, goal.cell),)
            goals.append(CellGoal(goal.slot, places[0][0], places[0][1], places))
        for child in check.children:
            walk(child)

    if layout.spec is not None and layout.spec.check is not None:
        walk(layout.spec.check)
    return goals


def wrap_half_pi(angle: float) -> float:
    """``angle`` [rad] folded into ``(-pi/2, pi/2]``, the range of an undirected axis or a symmetric tool."""
    folded = (float(angle) + math.pi / 2.0) % math.pi - math.pi / 2.0
    return math.pi / 2.0 if folded <= -math.pi / 2.0 + 1e-12 else folded


def rotation(angle: float) -> np.ndarray:
    """2-D rotation matrix of ``angle`` [rad]."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def yaw_of_wxyz(quat: np.ndarray) -> float:
    """Heading [rad] about +z of a wxyz quaternion."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def principal_axes(points_xy: np.ndarray) -> tuple[float, float, float]:
    """``(angle [rad], long half extent [m], short half extent [m])`` of horizontal points.

    The angle is the direction of the long axis in ``(-pi/2, pi/2]``. The extents are measured along the
    principal axes, so a round part has equal extents and an angle that carries no meaning.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    long_axis = vectors[:, 1]
    angle = wrap_half_pi(math.atan2(long_axis[1], long_axis[0]))
    along = centered @ long_axis
    across = centered @ vectors[:, 0]
    return angle, float(np.ptp(along)) / 2.0, float(np.ptp(across)) / 2.0


def release_turn(part_angle: float, long_half: float, short_half: float, cell_axis: float) -> float:
    """Turn [rad] about the vertical that lines the long axis of a part up with ``cell_axis``.

    A round part needs no turn. The result is in ``(-pi/2, pi/2]``, so it never exceeds a quarter turn.
    """
    if long_half < ALIGN_ASPECT * max(short_half, 1e-6):
        return 0.0
    return wrap_half_pi(cell_axis - part_angle)


def cell_axis(tray_yaw: float, cell_half: Sequence[float]) -> float:
    """Direction [rad] of the long side of a cell: the tray's own x axis, or its y axis for a cell that is deeper."""
    return wrap_half_pi(tray_yaw + (0.0 if cell_half[0] >= cell_half[1] else math.pi / 2.0))


def free_spot(
    cell_center: np.ndarray,
    tray_yaw: float,
    cell_half: Sequence[float],
    extent: Sequence[float],
    occupied: Sequence[tuple[np.ndarray, float]],
    radius: float,
) -> tuple[np.ndarray, float]:
    """Root position [m] in a cell that keeps the most room from the parts already there.

    Args:
        cell_center: Cell center [m], world xy.
        tray_yaw: Tray heading [rad].
        cell_half: Inner half sizes [m] of the cell, in the tray frame.
        extent: Half extents [m] of the part along the tray axes once it is aligned to the cell.
        occupied: ``(center xy [m], radius [m])`` of each part that already rests in the cell.
        radius: Footprint radius [m] of the part.

    Returns:
        The spot [m], world xy, and the gap [m] it keeps from the nearest part. The gap is negative when the cell is
        too crowded for another part to fit beside the others.
    """
    reach = [max(float(cell_half[axis]) - float(extent[axis]), 0.0) for axis in (0, 1)]
    ticks = np.linspace(-1.0, 1.0, SPOT_GRID)
    local = np.stack(np.meshgrid(ticks * reach[0], ticks * reach[1], indexing="ij"), axis=-1).reshape(-1, 2)
    world = np.asarray(cell_center, dtype=np.float64) + local @ rotation(tray_yaw).T
    gap = np.full(len(world), np.inf)
    for center, other_radius in occupied:
        gap = np.minimum(
            gap, np.linalg.norm(world - np.asarray(center, dtype=np.float64), axis=1) - radius - other_radius
        )
    score = np.minimum(gap, SPOT_ROOM_M) - 0.5 * np.linalg.norm(local, axis=1)
    best = int(np.argmax(score))
    return world[best], float(gap[best])


def pick_score(travel: float, clearance: float, height: float, attempts: int) -> float:
    """Pick priority of a part, lower first.

    Args:
        travel: Horizontal distance [m] from the tool to the part.
        clearance: Gap [m] from the part to the nearest other part that is still outside its cell.
        height: Height [m] of the part's top above the support under it.
        attempts: Grasps already tried on this part.

    Parts with room around them go first, so the crowded ones open up. A taller part goes first because it
    is the one that covers a neighbor.
    """
    return (
        TRAVEL_WEIGHT * travel
        - CLEARANCE_WEIGHT * min(max(clearance, 0.0), MAX_CLEARANCE_M)
        - HEIGHT_WEIGHT * height
        + ATTEMPT_PENALTY * attempts
    )
