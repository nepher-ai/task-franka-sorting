# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pq = pytest.importorskip("pyarrow.parquet")

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_sorting"
sys.path.insert(0, str(PACKAGE))

from franka_sorting.data.lerobot_writer import LeRobotWriter  # noqa: E402


def _episode(length: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(length, 8)),
        rng.normal(size=(length, 8)),
        rng.normal(size=(length, 7)),
    )


def test_episodes_are_indexed_globally_and_tasks_are_shared(tmp_path):
    writer = LeRobotWriter(tmp_path, fps=15)
    writer.add_episode(*_episode(5), task="put the apple in the pail")
    writer.add_episode(*_episode(3, 1), task="put the cubes in the bin", extra={"success": True})
    writer.add_episode(*_episode(4, 2), task="put the apple in the pail")
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"], info["fps"]) == (3, 12, 2, 15)
    assert info["codebase_version"] == "v2.1" and info["splits"] == {"train": "0:3"}
    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000002.parquet").to_pydict()
    assert table["index"] == [8, 9, 10, 11]
    assert table["frame_index"] == [0, 1, 2, 3] and set(table["episode_index"]) == {2}
    assert set(table["task_index"]) == {0}
    assert len(table["action"][0]) == 8 and len(table["observation.eef_pose"][0]) == 7
    assert table["timestamp"][1] == pytest.approx(1 / 15)
    episodes = [json.loads(line) for line in (tmp_path / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [e["length"] for e in episodes] == [5, 3, 4] and episodes[1]["success"] is True
    stats = [json.loads(line) for line in (tmp_path / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert len(stats) == 3 and stats[0]["stats"]["action"]["count"] == [5]


def test_an_existing_dataset_is_extended_not_overwritten(tmp_path):
    LeRobotWriter(tmp_path, fps=15).add_episode(*_episode(5), task="a")
    writer = LeRobotWriter(tmp_path, fps=15)
    assert writer.add_episode(*_episode(2), task="b") == 1
    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000001.parquet").to_pydict()
    assert table["index"] == [5, 6] and table["task_index"] == [1, 1]
    with pytest.raises(ValueError):
        LeRobotWriter(tmp_path, fps=30)


def test_modality_file_describes_the_gr00t_embodiment(tmp_path):
    writer = LeRobotWriter(tmp_path, fps=15, camera_hw={"cam_wrist": (8, 16)})
    writer._flush_meta()
    modality = json.loads((tmp_path / "meta" / "modality.json").read_text())
    assert modality["embodiment"] == "franka_2f85_jointpos"
    assert modality["action"]["absolute_joint_position"] == {"start": 0, "end": 7}
    assert modality["video"]["cam_wrist"] == {"original_key": "observation.images.cam_wrist"}


def test_frames_must_match_the_declared_cameras(tmp_path):
    writer = LeRobotWriter(tmp_path, fps=15, camera_hw={"cam_wrist": (8, 16)})
    with pytest.raises(ValueError):
        writer.add_episode(*_episode(2), task="a")


def test_videos_are_written_per_camera_and_episode(tmp_path):
    pytest.importorskip("av")
    writer = LeRobotWriter(tmp_path, fps=15, camera_hw={"cam_wrist": (32, 64)})
    frames = {"cam_wrist": np.random.default_rng(0).integers(0, 255, (6, 32, 64, 3), dtype=np.uint8)}
    writer.add_episode(*_episode(6), task="a", frames=frames)
    assert (
        tmp_path / "videos" / "chunk-000" / "observation.images.cam_wrist" / "episode_000000.mp4"
    ).stat().st_size > 0


def test_phase_column_and_names_are_stored(tmp_path):
    writer = LeRobotWriter(tmp_path, fps=15, phase_names=["init", "approach"])
    writer.add_episode(*_episode(3), task="a", phase=np.array([0, 1, 1]))
    writer.add_episode(*_episode(2, 1), task="a")
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["phase_names"] == ["init", "approach"] and info["features"]["annotation.phase"]["dtype"] == "int64"
    assert pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet").to_pydict()[
        "annotation.phase"
    ] == [0, 1, 1]
    assert pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000001.parquet").to_pydict()[
        "annotation.phase"
    ] == [-1, -1]
    with pytest.raises(ValueError):
        writer.add_episode(*_episode(2), task="a", phase=np.zeros(5))
    with pytest.raises(ValueError):
        LeRobotWriter(tmp_path, fps=15, phase_names=["other"])


def test_no_temp_files_remain_and_a_half_flushed_dataset_reopens(tmp_path):
    writer = LeRobotWriter(tmp_path, fps=15)
    for seed in range(3):
        writer.add_episode(*_episode(4, seed), task="a")
    assert not list(tmp_path.rglob(".*.tmp"))
    # A stop between two row files leaves one of them a row short: the reopened dataset keeps three episodes' worth.
    stats = tmp_path / "meta" / "episodes_stats.jsonl"
    stats.write_text("".join(stats.read_text().splitlines(keepends=True)[:2]))
    reopened = LeRobotWriter(tmp_path, fps=15)
    assert len(reopened.episodes) == 2 and reopened.frames == 8
    assert reopened.add_episode(*_episode(5), task="a") == 2
    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000002.parquet").to_pydict()
    assert table["index"] == [8, 9, 10, 11, 12] and len(table["episode_index"]) == 5
