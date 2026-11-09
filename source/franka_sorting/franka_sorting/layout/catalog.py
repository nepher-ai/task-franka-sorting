# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The asset library of a sorting benchmark: parts, containers, and trays, read from ``assets/catalog.json``.

All lengths are [m]. Part poses are rest poses on a flat support: ``quat_wxyz`` rotates the part from its mesh
frame, ``rest_z`` is the height [m] of the part origin (its center of mass) above the support, and ``footprint``
is a convex polygon [m] around the origin's projection that contains the part's silhouette.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import area, bounding_radius


@dataclass(frozen=True)
class PartPose:
    quat_wxyz: tuple[float, float, float, float]
    prob: float
    rest_z: float
    height: float
    grip_width: float
    footprint: np.ndarray = field(compare=False)
    usable: bool = True

    @property
    def radius(self) -> float:
        return bounding_radius(self.footprint)

    @property
    def footprint_area(self) -> float:
        return area(self.footprint)


@dataclass(frozen=True)
class PartAsset:
    """One part variant."""

    name: str
    cls: str
    usd: str
    size_m: tuple[float, float, float]
    mass_kg: float
    scale: float
    poses: tuple[PartPose, ...]
    usable: bool = True

    @property
    def usable_poses(self) -> tuple[PartPose, ...]:
        return tuple(pose for pose in self.poses if pose.usable)

    @property
    def footprint_area_m2(self) -> float:
        """Probability-weighted footprint area [m^2] of the usable rest poses, from the bounding circle."""
        poses = self.usable_poses
        weights = np.array([pose.prob for pose in poses])
        areas = np.array([pose.footprint_area for pose in poses])
        return float(np.dot(weights, areas) / weights.sum())


@dataclass(frozen=True)
class ContainerAsset:
    """A bin the parts start in. The origin is the center of its flat inner floor at table level, z up.

    ``outer_bounds_m`` is ``(x0, y0, x1, y1)`` [m] of the whole bin, which is not centered on the origin when the
    flat floor is off-center. ``floor_half_m`` is the half size [m] of the flat floor rectangle around the origin.
    """

    name: str
    usd: str
    size_m: tuple[float, float, float]
    scale: float
    outer_bounds_m: tuple[float, float, float, float]
    floor_half_m: tuple[float, float]
    floor_z: float
    rim_z: float

    @property
    def outer_polygon(self) -> np.ndarray:
        x0, y0, x1, y1 = self.outer_bounds_m
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


@dataclass(frozen=True)
class TrayAsset:
    """A parts tray. The origin is the tray center at its bottom. Cell ``(row, col)`` has index ``row * cols + col``."""

    name: str
    usd: str
    cols: int
    rows: int
    size_m: tuple[float, float, float]
    cell_centers: np.ndarray = field(compare=False)
    cell_half_m: tuple[float, float] = (0.0, 0.0)
    floor_z: float = 0.0
    rim_z: float = 0.0
    kinematic: bool = True

    @property
    def cells(self) -> int:
        return self.cols * self.rows

    @property
    def outer_half_m(self) -> tuple[float, float]:
        return (self.size_m[0] / 2.0, self.size_m[1] / 2.0)

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        x, y = self.cell_centers[row * self.cols + col]
        return float(x), float(y)


@dataclass
class AssetLibrary:
    """Parts by variant name, containers, trays, and the work tables (name to USD file under ``assets/``)."""

    parts: dict[str, PartAsset]
    containers: dict[str, ContainerAsset]
    trays: dict[str, TrayAsset]
    tables: dict[str, str] = field(default_factory=dict)

    def families(self) -> dict[str, list[PartAsset]]:
        """Usable part variants grouped by class, in variant order."""
        groups: dict[str, list[PartAsset]] = {}
        for asset in self.parts.values():
            if asset.usable:
                groups.setdefault(asset.cls, []).append(asset)
        return {cls: sorted(items, key=lambda item: item.name) for cls, items in sorted(groups.items())}


def _pose(record: dict) -> PartPose:
    return PartPose(
        quat_wxyz=tuple(float(v) for v in record["quat_wxyz"]),
        prob=float(record["prob"]),
        rest_z=float(record["rest_z"]),
        height=float(record["height"]),
        grip_width=float(record["grip_width"]),
        footprint=np.asarray(record["footprint"], dtype=np.float64),
        usable=bool(record.get("usable", True)),
    )


def library_from_catalog(catalog: dict) -> AssetLibrary:
    """Parse the parts, containers, trays, and tables of a ``catalog.json`` payload. Other kinds are ignored."""
    parts, containers, trays, tables = {}, {}, {}, {}
    for name, record in catalog.items():
        kind = record.get("kind")
        if kind == "part":
            parts[name] = PartAsset(
                name=name,
                cls=str(record["cls"]),
                usd=str(record["file"]),
                size_m=tuple(float(v) for v in record["size_m"]),
                mass_kg=float(record["mass_kg"]),
                scale=float(record["scale"]),
                poses=tuple(_pose(item) for item in record.get("poses", [])),
                usable=bool(record.get("usable", True)) and any(item.get("usable", True) for item in record["poses"]),
            )
        elif kind == "container":
            containers[name] = ContainerAsset(
                name=name,
                usd=str(record["file"]),
                size_m=tuple(float(v) for v in record["size_m"]),
                scale=float(record["scale"]),
                outer_bounds_m=tuple(float(v) for v in record["outer_bounds_m"]),
                floor_half_m=tuple(float(v) for v in record["floor_half_m"]),
                floor_z=float(record["floor_z"]),
                rim_z=float(record["rim_z"]),
            )
        elif kind == "tray":
            cells = record["cells"]
            trays[name] = TrayAsset(
                name=name,
                usd=str(record["file"]),
                cols=int(record["grid"][0]),
                rows=int(record["grid"][1]),
                size_m=tuple(float(v) for v in record["size_m"]),
                cell_centers=np.asarray(cells["centers"], dtype=np.float64),
                cell_half_m=tuple(float(v) for v in cells["half_m"]),
                floor_z=float(cells["floor_z"]),
                rim_z=float(cells["rim_z"]),
            )
        elif kind == "table":
            tables[name] = str(record["file"])
    return AssetLibrary(parts=parts, containers=containers, trays=trays, tables=tables)


def load_library(bundle: str | Path) -> AssetLibrary:
    """Library of the bundle directory ``bundle`` (reads ``assets/catalog.json``)."""
    path = Path(bundle) / "assets" / "catalog.json"
    if not path.is_file():
        raise FileNotFoundError(f"bundle has no assets/catalog.json under {bundle}")
    return library_from_catalog(json.loads(path.read_text(encoding="utf-8")))
