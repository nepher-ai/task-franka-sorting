# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""2D geometry for the layout sampler: convex polygons, rectangles, and separation tests. Lengths are [m].

Polygons are ``[N, 2]`` arrays of vertices in counter-clockwise order.
"""

from __future__ import annotations

import numpy as np


def rotation(yaw: float) -> np.ndarray:
    """2x2 rotation matrix of ``yaw`` [rad]."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def transform(polygon: np.ndarray, yaw: float, offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Rotate ``polygon`` by ``yaw`` [rad] about the origin, then translate it by ``offset`` [m]."""
    return np.asarray(polygon, dtype=np.float64) @ rotation(yaw).T + np.asarray(offset, dtype=np.float64)


def rectangle(half_x: float, half_y: float, yaw: float = 0.0, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Corners of a rectangle with half sizes [m], rotated by ``yaw`` [rad] about its center."""
    corners = np.array([[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y]])
    return transform(corners, yaw, center)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2D points (monotone chain), without collinear vertices."""
    pts = np.unique(np.round(np.asarray(points, dtype=np.float64), 12), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def half(sequence: np.ndarray) -> list[np.ndarray]:
        chain: list[np.ndarray] = []
        for point in sequence:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], point) <= 1e-15:
                chain.pop()
            chain.append(point)
        return chain

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def area(polygon: np.ndarray) -> float:
    """Area [m^2] of a simple polygon."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def enclosing_polygon(polygon: np.ndarray, max_vertices: int) -> np.ndarray:
    """Convex polygon with at most ``max_vertices`` that contains ``polygon``.

    The greedy step drops the vertex whose removal adds the least area. The two neighbouring edges are
    extended until they meet, so the result always contains the input.
    """
    poly = np.asarray(polygon, dtype=np.float64)
    while len(poly) > max(max_vertices, 3):
        best = None
        count = len(poly)
        for i in range(count):
            a, b = poly[(i - 1) % count], poly[i]
            c, d = poly[(i + 1) % count], poly[(i + 2) % count]
            before = poly[(i - 2) % count]
            point = _line_intersection(before, a, c, d)
            if point is None:
                continue
            # The new vertex has to lie outside the removed corner (a, b, c), on the far side of a-c.
            if _cross(a, c, point) >= 0.0 or _cross(a, c, b) >= 0.0:
                continue
            added = 0.5 * abs(_cross(a, point, c)) - 0.5 * abs(_cross(a, b, c))
            if best is None or added < best[0]:
                best = (added, i, point)
        if best is None:
            break
        _, i, point = best
        keep = [k for k in range(count) if k not in (i, (i - 1) % count)]
        poly = np.array([poly[k] for k in keep] + [point])
        poly = convex_hull(poly)
    return poly


def _line_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray | None:
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-14:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def bounding_radius(polygon: np.ndarray, center: tuple[float, float] = (0.0, 0.0)) -> float:
    """Largest distance [m] from ``center`` to a vertex."""
    return float(np.max(np.linalg.norm(np.asarray(polygon) - np.asarray(center), axis=1)))


def _axes(polygon: np.ndarray) -> np.ndarray:
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-15)


def separation(first: np.ndarray, second: np.ndarray) -> float:
    """Separation [m] of two convex polygons.

    A positive value is the exact distance between disjoint polygons. A value of 0 or below means they
    touch or overlap (it is the largest edge-normal gap, not a penetration depth).
    """
    best = -np.inf
    for axes in (_axes(first), _axes(second)):
        a = first @ axes.T
        b = second @ axes.T
        # Gap along each axis: positive when the projections are disjoint.
        gap = np.maximum(a.min(axis=0) - b.max(axis=0), b.min(axis=0) - a.max(axis=0))
        best = max(best, float(gap.max()))
    if best <= 0.0:
        return best
    # Disjoint: the exact distance is between a vertex of one and an edge of the other.
    return min(_vertex_edge_distance(first, second), _vertex_edge_distance(second, first))


def _vertex_edge_distance(points_of: np.ndarray, edges_of: np.ndarray) -> float:
    start = edges_of
    end = np.roll(edges_of, -1, axis=0)
    edge = end - start
    length_sq = np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-18)
    rel = points_of[:, None, :] - start[None, :, :]
    t = np.clip(np.einsum("pej,ej->pe", rel, edge) / length_sq, 0.0, 1.0)
    closest = start[None, :, :] + t[..., None] * edge[None, :, :]
    return float(np.linalg.norm(points_of[:, None, :] - closest, axis=-1).min())


def polygons_clear(first: np.ndarray, second: np.ndarray, margin: float) -> bool:
    """Whether the polygons are at least ``margin`` [m] apart."""
    return separation(first, second) >= margin


def contains_polygon(outer_half: tuple[float, float], polygon: np.ndarray, clearance: float = 0.0) -> bool:
    """Whether ``polygon`` lies inside the axis-aligned rectangle ``outer_half`` [m] shrunk by ``clearance`` [m]."""
    hx, hy = outer_half[0] - clearance, outer_half[1] - clearance
    if hx <= 0.0 or hy <= 0.0:
        return False
    return bool(np.all(np.abs(polygon[:, 0]) <= hx + 1e-12) and np.all(np.abs(polygon[:, 1]) <= hy + 1e-12))


def min_width(polygon: np.ndarray, steps: int = 180) -> float:
    """Smallest extent [m] of a polygon over all directions (the closing width of a parallel gripper)."""
    angles = np.linspace(0.0, np.pi, steps, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    projected = polygon @ directions.T
    return float((projected.max(axis=0) - projected.min(axis=0)).min())


def min_area_rectangle(polygon: np.ndarray, steps: int = 180) -> tuple[float, float, float]:
    """Minimum-area bounding rectangle: ``(long side, short side, angle of the long side)`` [m, m, rad]."""
    best = None
    for angle in np.linspace(0.0, np.pi / 2.0, steps, endpoint=False):
        rotated = polygon @ rotation(-angle).T
        extent = rotated.max(axis=0) - rotated.min(axis=0)
        value = float(extent[0] * extent[1])
        if best is None or value < best[0]:
            best = (value, extent, angle)
    _, extent, angle = best
    if extent[0] >= extent[1]:
        return float(extent[0]), float(extent[1]), float(angle)
    return float(extent[1]), float(extent[0]), float(angle + np.pi / 2.0)


def point_in_polygon(polygon: np.ndarray, point: tuple[float, float]) -> bool:
    """Whether ``point`` is inside a convex counter-clockwise polygon."""
    p = np.asarray(point, dtype=np.float64)
    edges = np.roll(polygon, -1, axis=0) - polygon
    rel = p - polygon
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    return bool(np.all(cross >= -1e-12))


def wrap_angle(angle: float) -> float:
    """Wrap ``angle`` [rad] to ``(-pi, pi]``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
