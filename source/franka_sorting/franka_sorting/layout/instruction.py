# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Names and the instruction text of a sorting task. Cells are named as the robot sees the tray."""

from __future__ import annotations

import numpy as np

from . import geometry as geo
from .catalog import TrayAsset
from .sampler import SortingLayout


def display_name(variant: str) -> str:
    """Spoken name of a part type: ``ball_stud_v1`` is ``ball stud v1``."""
    return variant.replace("_", " ")


def container_name(container: str) -> str:
    return container.replace("_", " ")


def view_cells(tray: TrayAsset, tray_yaw: float) -> np.ndarray:
    """``[cells, 2]`` (row, column) of every cell, 1-based, as seen from the robot.

    Rows count from the side of the tray nearest the robot (small x) to the far side. Columns count from the
    robot's left (large y) to its right. The tray's own yaw is snapped to a quarter turn.
    """
    quarter = int(np.round(tray_yaw / (np.pi / 2.0))) % 4
    centers = geo.transform(tray.cell_centers, quarter * np.pi / 2.0)
    rows = _rank(centers[:, 0]) + 1
    columns = _rank(-centers[:, 1]) + 1
    return np.stack([rows, columns], axis=1)


def _rank(values: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
    """Rank of each value among the distinct ones, where values within ``tolerance`` [m] are the same."""
    order = np.argsort(values)
    ranks = np.zeros(len(values), dtype=int)
    rank = 0
    for previous, current in zip(order[:-1], order[1:]):
        if values[current] - values[previous] > tolerance:
            rank += 1
        ranks[current] = rank
    return ranks


def view_shape(tray: TrayAsset, tray_yaw: float) -> tuple[int, int]:
    """``(rows, columns)`` of the tray as seen from the robot."""
    cells = view_cells(tray, tray_yaw)
    return int(cells[:, 0].max()), int(cells[:, 1].max())


def cell_label(tray: TrayAsset, tray_yaw: float, cell: int) -> str:
    row, column = view_cells(tray, tray_yaw)[cell]
    return f"row {row}, column {column}"


def build_instruction(layout: SortingLayout, tray: TrayAsset, library=None) -> str:
    """Instruction text with the full type to cell mapping, in reading order of the cells."""
    if layout.trays and library is not None:
        return _multi_instruction(layout, library)
    cells = view_cells(tray, layout.tray_pose[2])
    rows, columns = view_shape(tray, layout.tray_pose[2])
    order = sorted(range(tray.cells), key=lambda c: (cells[c, 0], cells[c, 1]))
    mapping = "; ".join(
        f"{display_name(layout.cell_types[c])} in row {cells[c, 0]}, column {cells[c, 1]}" for c in order if layout.cell_types[c]
    )
    quota = f" Sort at least {layout.quota} parts." if layout.quota else ""
    return (
        f"Sort the parts from the {container_name(layout.container)} into the parts tray, which has {rows} rows and "
        f"{columns} columns. Rows are counted from the side of the tray nearest the robot, columns from the "
        f"robot's left. Put every part into its own cell: {mapping}.{quota}"
    )


def _multi_instruction(layout: SortingLayout, library) -> str:
    """Name every tray, the cell of each type, and the quota."""
    sentences = [
        f"Sort at least {layout.quota} parts from the {container_name(layout.container)} into the trays. "
        "Rows are counted from the side of a tray nearest the robot, columns from the robot's left."
    ]
    offset = 0
    for name, pose in zip(layout.tray_list(), layout.tray_pose_list()):
        tray = library.trays[name]
        cells = view_cells(tray, pose[2])
        rows, columns = view_shape(tray, pose[2])
        order = sorted(range(tray.cells), key=lambda c: (cells[c, 0], cells[c, 1]))
        mapping = "; ".join(
            f"{display_name(layout.cell_types[offset + c])} in row {cells[c, 0]}, column {cells[c, 1]}"
            for c in order
            if layout.cell_types[offset + c]
        )
        label = display_name(name.removeprefix("parts_tray_"))
        body = f"The {label} tray has {rows} rows and {columns} columns."
        if mapping:
            body += f" Put {mapping}."
        sentences.append(body)
        offset += tray.cells
    return " ".join(sentences)
