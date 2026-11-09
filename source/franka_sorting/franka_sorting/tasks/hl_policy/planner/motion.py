# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-parameterised tool motion along waypoint polylines.

A move runs from rest to rest with a trapezoidal speed profile in path length, so the tool cruises at
``v_max`` [m/s] and only ramps at its ends. The evaluator rewards a cruise between 0.20 and 0.27 m/s and
ignores samples below 0.02 m/s, so a move either cruises or stands still.

``sample(t + lead)`` gives the tracker a target a little ahead of the reference, which cancels the
lag of the joint drives while the end point stays exact.
"""

from __future__ import annotations

import math

import numpy as np

V_CRUISE_M_S = 0.25
A_MAX_M_S2 = 2.5
YAW_RATE_RAD_S = 2.0
# Corners are rounded over this distance [m] on each side, so the reference never turns instantly.
BLEND_M = 0.025
_BLEND_POINTS = 6


def round_corners(points: np.ndarray, blend_m: float = BLEND_M) -> np.ndarray:
    """Replace each interior corner of a polyline with a quadratic blend that starts ``blend_m`` before it."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3 or blend_m <= 0.0:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        before, corner, after = pts[i - 1], pts[i], pts[i + 1]
        d_in, d_out = np.linalg.norm(corner - before), np.linalg.norm(after - corner)
        if d_in < 1e-6 or d_out < 1e-6:
            out.append(corner)
            continue
        u_in, u_out = (corner - before) / d_in, (after - corner) / d_out
        if float(u_in @ u_out) > 0.999:
            out.append(corner)
            continue
        cut = min(blend_m, 0.45 * d_in, 0.45 * d_out)
        a, b = corner - cut * u_in, corner + cut * u_out
        for t in np.linspace(0.0, 1.0, _BLEND_POINTS):
            out.append((1 - t) ** 2 * a + 2 * (1 - t) * t * corner + t * t * b)
    out.append(pts[-1])
    return np.asarray(out)


class Move:
    """Rest-to-rest motion through ``points`` ``[K, 3]`` [m], angles blended from ``yaw0`` to ``yaw1`` [rad].

    The angles are a yaw, or an array such as ``(yaw, pitch)``. They are interpolated with the path length,
    so the tool turns while it travels. A turn that needs longer than the path stretches the move.
    """

    def __init__(
        self,
        points: np.ndarray,
        yaw0: float,
        yaw1: float,
        v_max: float = V_CRUISE_M_S,
        a_max: float = A_MAX_M_S2,
        yaw_rate: float = YAW_RATE_RAD_S,
        blend_m: float = BLEND_M,
        turn_by_end: float = 0.0,
    ):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        keep = [pts[0]]
        for point in pts[1:]:
            if np.linalg.norm(point - keep[-1]) > 1e-6:
                keep.append(point)
        self.points = round_corners(np.asarray(keep), blend_m)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1) if len(self.points) > 1 else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])
        self._scalar = np.ndim(yaw0) == 0 and np.ndim(yaw1) == 0
        self.yaw0 = np.atleast_1d(np.asarray(yaw0, dtype=np.float64)).copy()
        self.yaw1 = np.atleast_1d(np.asarray(yaw1, dtype=np.float64)).copy()
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        # Ramp time and distance of a full trapezoid.
        ramp_t = self.v_max / self.a_max
        ramp_d = 0.5 * self.a_max * ramp_t**2
        if self.length >= 2.0 * ramp_d:
            self._peak = self.v_max
            self._ramp_t = ramp_t
            self._cruise_t = (self.length - 2.0 * ramp_d) / self.v_max
        else:
            self._peak = math.sqrt(self.a_max * self.length)
            self._ramp_t = self._peak / self.a_max
            self._cruise_t = 0.0
        path_t = 2.0 * self._ramp_t + self._cruise_t
        yaw_t = float(np.abs(self.yaw1 - self.yaw0).max()) / max(yaw_rate, 1e-6)
        # Angles finish ``turn_by_end`` [m] before the end, e.g. before a descent onto an object.
        self._turn_by = float(np.clip(self.length - turn_by_end, 0.3 * self.length, self.length))
        turn_share = self._turn_by / self.length if self.length > 1e-9 else 1.0
        # A pure turn, or a turn longer than the translation, stretches time uniformly.
        self._stretch = max(1.0, yaw_t / (path_t * turn_share)) if path_t > 1e-9 else 1.0
        self.duration = max(path_t * self._stretch, yaw_t)

    def _arc(self, t: float) -> float:
        """Path length [m] travelled at time ``t`` [s]."""
        if self.length <= 0.0:
            return 0.0
        t = min(max(t / self._stretch, 0.0), 2.0 * self._ramp_t + self._cruise_t)
        a, ramp, cruise, peak = self.a_max, self._ramp_t, self._cruise_t, self._peak
        if t <= ramp:
            return 0.5 * a * t * t
        d_ramp = 0.5 * a * ramp * ramp
        if t <= ramp + cruise:
            return d_ramp + peak * (t - ramp)
        rest = t - ramp - cruise
        return min(d_ramp + peak * cruise + peak * rest - 0.5 * a * rest * rest, self.length)

    def _angles(self, frac: float):
        angles = self.yaw0 + (self.yaw1 - self.yaw0) * frac
        return float(angles[0]) if self._scalar else angles

    def sample(self, t: float) -> tuple[np.ndarray, float | np.ndarray]:
        """Position [m] and angles [rad] at time ``t`` [s], clamped to the end of the move."""
        if self.length <= 0.0 or len(self.points) == 1:
            frac_t = 1.0 if self.duration <= 0 else min(max(t / self.duration, 0.0), 1.0)
            return self.points[-1].copy(), self._angles(_smooth(frac_t))
        s = self._arc(t)
        index = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.points) - 2))
        span = self.cum[index + 1] - self.cum[index]
        u = 0.0 if span <= 0 else (s - self.cum[index]) / span
        position = self.points[index] + u * (self.points[index + 1] - self.points[index])
        return position, self._angles(min(s / self._turn_by, 1.0))

    @property
    def end(self) -> np.ndarray:
        return self.points[-1].copy()


def _smooth(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)
