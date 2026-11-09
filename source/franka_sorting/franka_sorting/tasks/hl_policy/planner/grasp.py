# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-down grasp search on surface points. The gripper points down and closes on one horizontal axis.

A grasp fits the opening, leaves a descent path for the open fingers, and clears the palm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 2F-85 pad and opening geometry [m].
PAD_HALF_WIDTH_M = 0.011
PAD_HALF_LENGTH_M = 0.019
OPEN_HALF_M = 0.0425
FINGER_OUTER_M = 0.060
MAX_WIDTH_M = 0.078
PALM_CLEARANCE_M = 0.045
PALM_HALF_M = 0.05
# The inner knuckles, which carry no contact sensor, reach the pads' contact plane 1.9-2.7 cm above the tool
# and come within 2 mm of it from 1.1 cm up (measured over the closure range, devkit/finger_geometry.py).
# A wall that rises higher than this [m] at the pad line takes the grip force on the knuckles instead of the
# pads: an 18 cm bottle gripped 2.4 cm below its top stalls the fingers with no pad contact.
LINK_CLEARANCE_M = 0.017
LINK_HALF_WIDTH_M = 0.025
LINK_REACH_M = 0.08
# Lowest tool height [m] above the table that keeps the fingertips off it. The pads end 1.9 cm below the tool
# (measured on the asset), so this leaves about 4 mm, and a 1 cm phone still meets the pads. Round objects
# are sensitive to it: the pads touch near their tips, and a bagel gripped 1 mm higher is squeezed out
# upward (devkit runs: 1 of 6 held at 2.4 cm, 4 of 6 at 2.3 cm).
TOOL_Z_MIN_M = 0.023
# Gap [m] the open fingers and the palm keep from objects other than the one grasped.
NEIGHBOR_MARGIN_M = 0.006
# Half width [m] of the gripper body across the closing axis, above the palm.
BODY_HALF_WIDTH_M = 0.045
_STEP_M = 0.008
_YAWS = np.deg2rad(np.arange(-90.0, 90.0, 15.0))
# A pad whose contact surface leans more than about 60 degrees off the closing axis squeezes the object out
# instead of holding it (a bagel gripped across a chord near its rim, a bottle on its tapering shoulder).
# Such grasps are dropped while others remain, and the lean is penalised by this much per unit of tangent.
MAX_CONTACT_SLOPE = 1.7
CONTACT_SLOPE_WEIGHT = 0.12


@dataclass(frozen=True)
class Grasp:
    """One feasible grasp in the object root frame.

    ``center_local`` is the tool position [m], ``axis_local`` the horizontal closing axis,
    ``width_m`` the thickness [m] between the pads, and ``score`` is larger when better. ``slope`` is the
    tangent of the larger angle between the closing axis and a pad's contact normal.
    """

    center_local: np.ndarray
    axis_local: np.ndarray
    width_m: float
    score: float
    slope: float = 0.0


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix of a wxyz quaternion."""
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def sample_prim_points(stage, prim_path: str, count: int = 6000, seed: int = 0) -> np.ndarray:
    """Surface points [m] under ``prim_path``, world frame, shape ``[count, 3]``, sampled by triangle area."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"no prim at {prim_path}")
    triangles = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).GetPurposeAttr().Get() == UsdGeom.Tokens.guide:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        matrix = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        local = np.asarray(points, dtype=np.float64)
        world = local @ matrix[:3, :3] + matrix[3, :3]
        corner = np.asarray(indices, dtype=np.int64)
        sizes = np.asarray(counts, dtype=np.int64)
        starts = np.cumsum(sizes) - sizes
        # Fan-triangulate each polygon around its first vertex.
        for extra in range(1, int(sizes.max()) - 1):
            valid = sizes > extra + 1
            first = corner[starts[valid]]
            second = corner[starts[valid] + extra]
            third = corner[starts[valid] + extra + 1]
            triangles.append(np.stack([world[first], world[second], world[third]], axis=1))
    if not triangles:
        raise ValueError(f"no mesh triangles under {prim_path}")
    tris = np.concatenate(triangles, axis=0)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=-1)
    if float(area.sum()) <= 0.0:
        raise ValueError(f"degenerate mesh under {prim_path}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tris), size=count, p=area / area.sum())
    bary = rng.random((count, 2))
    flip = bary.sum(axis=1) > 1.0
    bary[flip] = 1.0 - bary[flip]
    a, b, c = tris[pick, 0], tris[pick, 1], tris[pick, 2]
    return a + bary[:, :1] * (b - a) + bary[:, 1:] * (c - a)


def find_grasps(
    points: np.ndarray,
    root_z: float,
    limit: int = 12,
    table_z: float = 0.0,
    tool_z_min: float = TOOL_Z_MIN_M,
) -> list[Grasp]:
    """Feasible top-down grasps, best first.

    ``points`` are surface points [m] in the object root frame with world-aligned axes, ``[M, 3]``.
    ``root_z`` is the root height [m] and ``table_z`` the height [m] of what the part rests on, in the same
    frame. The tool stays ``tool_z_min`` [m] above ``table_z``; a thin part allows less than the default.
    """
    pts = np.asarray(points, dtype=np.float64)
    z_world = pts[:, 2] + root_z
    centroid = pts.mean(axis=0)
    # Grip tall objects above their middle: the fingertips then stay clear of the table and of the base.
    preferred_z = 0.5 * (float(centroid[2]) + float(pts[:, 2].max())) + root_z
    low = max(table_z + tool_z_min, float(z_world.min()) + 0.006)
    high = max(float(z_world.max()) - 0.004, low)
    levels = np.arange(low, high + 1e-6, 0.008)
    if len(levels) > 14:
        levels = np.linspace(low, high, 14)
    found: list[Grasp] = []
    for yaw in _YAWS:
        u_axis = np.array([-math.sin(yaw), math.cos(yaw)])
        v_axis = np.array([math.cos(yaw), math.sin(yaw)])
        pu = pts[:, :2] @ u_axis
        pv = pts[:, :2] @ v_axis
        for cv in np.arange(pv.min(), pv.max() + 1e-6, 0.012):
            strip = np.nonzero(np.abs(pv - cv) < PAD_HALF_WIDTH_M)[0]
            if len(strip) < 8:
                continue
            wide = np.nonzero(np.abs(pv - cv) < PALM_HALF_M)[0]
            wide_u, wide_v, wide_z = pu[wide], pv[wide], z_world[wide]
            for gz in levels:
                band = strip[np.abs(z_world[strip] - gz) < PAD_HALF_LENGTH_M]
                if len(band) < 8:
                    continue
                order = np.argsort(pu[band])
                band_u, band_z = pu[band][order], z_world[band][order]
                band_v = pv[band][order]
                for cu in np.arange(band_u[0], band_u[-1] + 1e-9, _STEP_M):
                    window = _window(band_u, cu)
                    if window is None:
                        continue
                    # Recenter the fingers on what they enclose, then require the same set.
                    cu = 0.5 * (band_u[window[0]] + band_u[window[1] - 1])
                    window = _window(band_u, cu)
                    width = float(band_u[window[1] - 1] - band_u[window[0]])
                    if width > MAX_WIDTH_M or abs(cu - 0.5 * (band_u[window[0]] + band_u[window[1] - 1])) > 1e-6:
                        continue
                    if not _feasible(wide_u, wide_v, wide_z, cu, cv, gz):
                        continue
                    if _linkage_blocked(wide_u, wide_v, wide_z, cu, cv, gz, width):
                        continue
                    xy = cu * u_axis + cv * v_axis
                    span = slice(window[0], window[1])
                    pads = band_z[span]
                    slope = _contact_slope(band_u[span], band_v[span] - cv, pads - gz)
                    score = (
                        -3.0 * float(np.hypot(*(xy - centroid[:2])))
                        - 2.0 * abs(gz - preferred_z)
                        + 0.5 * min(float(np.ptp(pads)) / (2 * PAD_HALF_LENGTH_M), 1.0)
                        + 0.1 * min((window[1] - window[0]) / 40.0, 1.0)
                        - 0.05 * abs(float(yaw))
                        - CONTACT_SLOPE_WEIGHT * min(slope, MAX_CONTACT_SLOPE)
                    )
                    center = np.array([xy[0], xy[1], gz - root_z])
                    found.append(Grasp(center, np.array([u_axis[0], u_axis[1], 0.0]), width, score, slope))
    found.sort(key=lambda grasp: -grasp.score)
    holding = [grasp for grasp in found if grasp.slope <= MAX_CONTACT_SLOPE]
    return _distinct(holding or found, limit)


def _contact_slope(u: np.ndarray, dv: np.ndarray, dz: np.ndarray) -> float:
    """Tangent of the larger angle between the closing axis and the contact normal of the two pads.

    ``u`` are the enclosed points along the closing axis [m], ``dv`` and ``dz`` their offsets [m] from the pad
    center across the pad and along it. Each pad meets the outermost points first: the envelope of those over
    the pad face, as a plane ``u = a + b dv + c dz``, gives the normal's lean ``sqrt(b^2 + c^2)``.
    """
    cells_v = np.clip(((dv + PAD_HALF_WIDTH_M) / (PAD_HALF_WIDTH_M / 1.5)).astype(int), 0, 2)
    cells_z = np.clip(((dz + PAD_HALF_LENGTH_M) / (PAD_HALF_LENGTH_M / 2.0)).astype(int), 0, 3)
    cell = cells_v * 4 + cells_z
    worst = 0.0
    for side in (-1.0, 1.0):
        rows = []
        for key in np.unique(cell):
            members = np.nonzero(cell == key)[0]
            k = members[np.argmax(side * u[members])]
            rows.append((dv[k], dz[k], side * u[k]))
        if len(rows) < 3:
            continue
        rows = np.asarray(rows)
        # Only the face near the first contact can touch the pad. A face that leaves the pad over one column of
        # cells is met by the pad's edge alone.
        rows = rows[rows[:, 2] > rows[:, 2].max() - 0.025]
        if len(np.unique(rows[:, 0].round(3))) < 2 or len(rows) < 3:
            worst = max(worst, 2.0 * MAX_CONTACT_SLOPE)
            continue
        design = np.column_stack([np.ones(len(rows)), rows[:, 0], rows[:, 1]])
        coeff, *_ = np.linalg.lstsq(design, rows[:, 2], rcond=None)
        worst = max(worst, float(np.hypot(coeff[1], coeff[2])))
    return worst


def _window(sorted_u: np.ndarray, center: float) -> tuple[int, int] | None:
    first = int(np.searchsorted(sorted_u, center - OPEN_HALF_M, side="left"))
    last = int(np.searchsorted(sorted_u, center + OPEN_HALF_M, side="right"))
    return (first, last) if last - first >= 4 else None


def _feasible(wide_u, wide_v, wide_z, cu: float, cv: float, gz: float) -> bool:
    offset = np.abs(wide_u - cu)
    in_path = (
        (np.abs(wide_v - cv) < PAD_HALF_WIDTH_M + 0.006)
        & (offset >= OPEN_HALF_M)
        & (offset <= FINGER_OUTER_M)
        & (wide_z > gz - PAD_HALF_LENGTH_M)
    )
    if bool(in_path.any()):
        return False
    return not bool(((offset < PALM_HALF_M) & (wide_z > gz + PALM_CLEARANCE_M)).any())


def _linkage_blocked(wide_u, wide_v, wide_z, cu: float, cv: float, gz: float, width: float) -> bool:
    """Whether the object rises at the pads' contact line high enough to stop the closing linkage."""
    offset = np.abs(wide_u - cu)
    blocked = (
        (np.abs(wide_v - cv) < LINK_HALF_WIDTH_M)
        & (offset > 0.5 * width - 0.005)
        & (offset < OPEN_HALF_M)
        & (wide_z > gz + LINK_CLEARANCE_M)
        & (wide_z < gz + LINK_REACH_M)
    )
    return bool(blocked.any())


def _distinct(grasps: list[Grasp], limit: int) -> list[Grasp]:
    """Drop a grasp within 2 cm and 20 degrees of a better one."""
    kept: list[Grasp] = []
    for grasp in grasps:
        if all(
            np.linalg.norm(grasp.center_local - other.center_local) > 0.02
            or abs(float(np.dot(grasp.axis_local, other.axis_local))) < math.cos(math.radians(20.0))
            for other in kept
        ):
            kept.append(grasp)
        if len(kept) >= limit:
            break
    return kept


def fallback_grasp(points: np.ndarray, root_z: float, table_z: float = 0.0, tool_z_min: float = TOOL_Z_MIN_M) -> Grasp:
    """Pinch the middle of the narrower horizontal side."""
    pts = np.asarray(points, dtype=np.float64)
    low, high = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (low + high)
    axis = np.array([1.0, 0.0, 0.0]) if (high[0] - low[0]) < (high[1] - low[1]) else np.array([0.0, 1.0, 0.0])
    height = max(table_z + tool_z_min, root_z + float(center[2]))
    return Grasp(
        center_local=np.array([center[0], center[1], height - root_z]),
        axis_local=axis,
        width_m=float(min(high[0] - low[0], high[1] - low[1])),
        score=-10.0,
    )


def neighbor_height(center: np.ndarray, axis: np.ndarray, neighbors: np.ndarray) -> float:
    """Lowest tool height [m] at which the open fingers and the palm miss every ``neighbors`` point.

    ``center`` is the tool position [m] and ``axis`` the closing axis, both world frame. ``neighbors`` are
    surface points [m] of the other objects, ``[M, 3]``, including whatever the object rests on.
    """
    u = np.asarray(axis, dtype=np.float64)[:2]
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    v = np.array([-u[1], u[0]])
    rel = np.asarray(neighbors, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    du, dv, dz = np.abs(rel[:, :2] @ u), np.abs(rel[:, :2] @ v), rel[:, 2]
    floor = -np.inf
    fingers = (du < FINGER_OUTER_M + NEIGHBOR_MARGIN_M) & (dv < PAD_HALF_WIDTH_M + NEIGHBOR_MARGIN_M + 0.004)
    if fingers.any():
        floor = float(dz[fingers].max()) + PAD_HALF_LENGTH_M + NEIGHBOR_MARGIN_M
    body = (du < PALM_HALF_M + NEIGHBOR_MARGIN_M) & (dv < BODY_HALF_WIDTH_M + NEIGHBOR_MARGIN_M)
    if body.any():
        floor = max(floor, float(dz[body].max()) - PALM_CLEARANCE_M + NEIGHBOR_MARGIN_M)
    return float(center[2]) + floor


def tool_yaw(axis_world: np.ndarray) -> float:
    """Yaw [rad] about +Z for fingers closing along ``axis_world``, in ``(-pi/2, pi/2]``."""
    yaw = math.atan2(-float(axis_world[0]), float(axis_world[1]))
    if yaw > math.pi / 2 + 1e-9:
        yaw -= math.pi
    elif yaw <= -math.pi / 2 + 1e-9:
        yaw += math.pi
    return yaw
