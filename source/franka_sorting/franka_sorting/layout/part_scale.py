# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scale rules that make a part suitable for the Robotiq 2F-85 and a 150 x 200 mm cell. Lengths are [m].

The catalog stores one scale per variant. A rebuild multiplies that scale by 3, or by 2.5 for the wide
variants in ``WIDE_VARIANTS``. A rest pose is usable when the fingers can close on either its horizontal pinch
or its thickness, it is not too thin to clear the floor, it is not too tall for a cell, and its footprint fits
the cell. A variant is usable when the usable poses carry most of the rest-pose probability. Weight does not
remove a variant: the Franka payload is ``MASS_MAX_KG``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

GRIP_WIDTH_MIN_M = 0.010
"""Narrowest closing width of a pose [m]: below it the pads grip too little of the part."""

THICKNESS_MIN_M = 0.005
"""Thinnest pose [m]: the fingertips need room to close above the floor."""

HEIGHT_MAX_M = 0.050
"""Tallest pose [m]: it keeps a resting part inside a 6 cm tray (3 mm floor, about 57 mm of interior)."""

CLOSING_WIDTH_MAX_M = 0.070
"""Widest closing width [m]: the 2F-85 opens to 85 mm, and the fingers need clearance on both sides."""

CELL_MARGIN_M = 0.008
"""Margin [m] between a part footprint and the cell walls, on each side. Packing clearance, not a hard fit."""

REFERENCE_CELL_M = (0.200, 0.150)
"""Inner (long, short) size [m] of a sorting cell."""

WIDE_VARIANTS = frozenset(
    {
        "bracket_square_window_v1",
        "bracket_square_window_v2",
        "bracket_square_window_v3",
        "flat_washer_v2",
        "flat_washer_v3",
        "plate_cutout_large_v1",
        "plate_cutout_large_v2",
        "plate_cutout_large_v3",
        "union_body_v3",
    }
)
"""Variants rebuilt at 2.5x. Every other part is rebuilt at 3x. Their footprint may use the full cell."""

WIDE_SCALE = 2.5
"""Scale multiplier of :data:`WIDE_VARIANTS`."""

DEFAULT_SCALE = 3.0
"""Scale multiplier of every part that is not in :data:`WIDE_VARIANTS`."""

TALL_VARIANT = "union_body_v3"
"""Variant whose 2.5x height is about 60 mm. It is seated only in a tray deeper than the part."""

TALL_HEIGHT_MAX_M = 0.080
"""Height cap [m] of :data:`TALL_VARIANT`. Other poses stay under :data:`HEIGHT_MAX_M`."""

MIN_POSE_PROB = 0.01
"""Rest poses below this probability are not kept."""

MIN_USABLE_MASS = 0.75
"""Share of the rest-pose probability that the usable poses must carry."""

SCALE_STEP = 0.05
SCALE_MIN = 1.0
SCALE_MAX = 16.0
"""Largest stored scale. A 3x rebuild of a gripper-sized part lands above the old cap of 4."""

MASS_MAX_KG = 3.0
"""Franka payload [kg], including the part. A variant is not removed for weight."""


@dataclass(frozen=True)
class NativePose:
    """Rest pose of a part at its native size."""

    prob: float
    height: float
    grip_width: float
    rect_long: float
    rect_short: float


def scale_factor(name: str) -> float:
    """Multiplier applied to the stored scale of ``name`` when the catalog is rebuilt."""
    return WIDE_SCALE if name in WIDE_VARIANTS else DEFAULT_SCALE


def pose_limits(name: str) -> tuple[float, float]:
    """``(cell margin [m], height cap [m])`` used to accept a pose of ``name``."""
    margin = 0.0 if name in WIDE_VARIANTS else CELL_MARGIN_M
    height = TALL_HEIGHT_MAX_M if name == TALL_VARIANT else HEIGHT_MAX_M
    return margin, height


def _in_pinch(width: float) -> bool:
    return GRIP_WIDTH_MIN_M <= width <= CLOSING_WIDTH_MAX_M


def pose_usable(
    pose: NativePose, scale: float, *, margin: float | None = None, height_max: float | None = None
) -> bool:
    """Whether a rest pose passes every rule at ``scale``.

    The fingers may close on the horizontal pinch or on the thickness. A plate stays graspable when its
    thickness is inside the pinch range even if its long side is not. ``margin`` is the cell clearance on
    each side. The wide variants pass with no margin, because their footprint fills the 150 x 200 mm cell.
    """
    cell_long, cell_short = REFERENCE_CELL_M
    if margin is None:
        margin = CELL_MARGIN_M
    if height_max is None:
        height_max = HEIGHT_MAX_M
    height = pose.height * scale
    width = pose.grip_width * scale
    return (
        (_in_pinch(width) or _in_pinch(height))
        and height >= THICKNESS_MIN_M
        and height <= height_max
        and pose.rect_long * scale <= cell_long - 2.0 * margin
        and pose.rect_short * scale <= cell_short - 2.0 * margin
    )


def usable_mass(
    poses: Sequence[NativePose], scale: float, *, margin: float | None = None, height_max: float | None = None
) -> float:
    """Share of the rest-pose probability that is usable at ``scale``."""
    total = sum(pose.prob for pose in poses)
    if total <= 0.0:
        return 0.0
    return sum(pose.prob for pose in poses if pose_usable(pose, scale, margin=margin, height_max=height_max)) / total


def variant_usable(
    poses: Sequence[NativePose], scale: float, native_mass_kg: float = 0.0, *, name: str = ""
) -> bool:
    """Whether a variant is usable at ``scale``.

    Enough of the rest-pose probability has to pass. ``native_mass_kg`` is not a reason to drop the variant:
    the scaled mass is kept up to the Franka payload. A negative mass is rejected.
    """
    if native_mass_kg < 0.0:
        return False
    margin, height_max = pose_limits(name) if name else (None, None)
    return usable_mass(poses, scale, margin=margin, height_max=height_max) >= MIN_USABLE_MASS


def choose_family_scale(
    variants: Mapping[str, Sequence[NativePose]], native_mass_kg: Mapping[str, float] | None = None
) -> tuple[float, dict[str, bool]]:
    """One scale for a family, and which variants are usable at it.

    The scale is the smallest one on the grid that keeps the most variants usable. When no scale keeps any
    variant usable, the scale is ``SCALE_MIN`` and every variant is unusable.
    """
    masses = native_mass_kg or {}
    best_scale, best_flags, best_count = SCALE_MIN, {name: False for name in variants}, 0
    steps = int(round((SCALE_MAX - SCALE_MIN) / SCALE_STEP))
    for index in range(steps + 1):
        scale = round(SCALE_MIN + index * SCALE_STEP, 6)
        flags = {
            name: variant_usable(poses, scale, masses.get(name, 0.0), name=name) for name, poses in variants.items()
        }
        count = sum(flags.values())
        if count > best_count:
            best_scale, best_flags, best_count = scale, flags, count
    return best_scale, best_flags
