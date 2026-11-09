# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded visual randomization draws for data generation. Pure numpy, no simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RandomizeConfig:
    """Ranges of the per-episode visual randomization."""

    lighting: bool = True
    """Randomize the dome and the sphere light."""
    dome_scale: tuple[float, float] = (0.6, 1.6)
    """Range of the factor on the dome intensity of the stage."""
    dome_yaw_deg: tuple[float, float] = (-180.0, 180.0)
    """Range of the dome rotation about the vertical axis [deg]."""
    sphere_scale: tuple[float, float] = (0.5, 1.8)
    """Range of the factor on the sphere light intensity of the stage."""
    sphere_offset_m: tuple[float, float, float] = (0.25, 0.25, 0.15)
    """Half range of the sphere light shift [m] along x, y and z."""
    camera_jitter: float = 0.0
    """Scale of the camera extrinsics jitter. Zero keeps the cameras where the stage puts them."""
    camera_pos_m: float = 0.02
    """Half range of the camera shift [m] per axis at ``camera_jitter`` 1."""
    camera_rot_deg: float = 2.0
    """Half range of the camera rotation [deg] per axis at ``camera_jitter`` 1."""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LightingDraw:
    """Light parameters of one episode."""

    dome_scale: float
    dome_yaw_deg: float
    sphere_scale: float
    sphere_offset_m: tuple[float, float, float]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CameraDraw:
    """Camera extrinsics offset of one episode."""

    shift_m: tuple[float, float, float]
    rotvec_rad: tuple[float, float, float]

    def as_dict(self) -> dict:
        return asdict(self)


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def draw_lighting(rng: np.random.Generator, config: RandomizeConfig) -> LightingDraw:
    """Light draw of one episode. Identity when ``config.lighting`` is off."""
    if not config.lighting:
        return LightingDraw(1.0, 0.0, 1.0, (0.0, 0.0, 0.0))
    offset = tuple(float(rng.uniform(-half, half)) for half in config.sphere_offset_m)
    return LightingDraw(
        _uniform(rng, config.dome_scale),
        _uniform(rng, config.dome_yaw_deg),
        _uniform(rng, config.sphere_scale),
        offset,  # type: ignore[arg-type]
    )


def draw_camera(rng: np.random.Generator, config: RandomizeConfig) -> CameraDraw:
    """Camera offset of one episode, zero when ``config.camera_jitter`` is zero."""
    scale = float(config.camera_jitter)
    if scale <= 0.0:
        return CameraDraw((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    shift = rng.uniform(-1.0, 1.0, 3) * config.camera_pos_m * scale
    rotvec = np.deg2rad(rng.uniform(-1.0, 1.0, 3) * config.camera_rot_deg * scale)
    return CameraDraw(tuple(float(v) for v in shift), tuple(float(v) for v in rotvec))  # type: ignore[arg-type]
