# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeRobot v2.1 writer. A row is one control step: joints, gripper, tool pose, cameras, and the action.

The layout matches LeRobot, openpi, and Isaac-GR00T. Parquet needs ``pyarrow`` and video needs ``av``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .atomic import atomic_write, atomic_write_text

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
IMAGE_PREFIX = "observation.images."

STATE_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["gripper"]
ACTION_NAMES = [f"panda_joint{i}_target" for i in range(1, 8)] + ["gripper_close"]
EEF_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
PHASE_KEY = "annotation.phase"
NO_PHASE = -1
"""Phase value of a frame whose episode came without phases."""
_VIDEO_STAT_STRIDE = 10


class LeRobotWriter:
    """Append episodes to a LeRobot v2.1 directory. An existing dataset is extended.

    ``fps`` is the control rate [Hz]. ``camera_hw`` is ``{name: (height, width)}`` [pixels]. ``phase_names`` names
    the values of the per-frame ``annotation.phase`` column. Every file is written next to its target and renamed, so
    an interrupted run leaves whole episodes only; a reopened dataset drops metadata rows of an episode that
    never finished.
    """

    def __init__(
        self,
        root: str | Path,
        fps: int,
        camera_hw: dict[str, tuple[int, int]] | None = None,
        robot_type: str = "franka_2f85",
        phase_names: Sequence[str] | None = None,
    ):
        self.root = Path(root)
        self.fps = int(fps)
        self.camera_hw = dict(camera_hw or {})
        self.robot_type = robot_type
        self.phase_names = list(phase_names or [])
        self.episodes: list[dict] = []
        self.tasks: dict[str, int] = {}
        self.frames = 0
        self._stats: list[dict] = []
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        if (self.root / "meta" / "info.json").is_file():
            self._resume()

    # ------------------------------------------------------------------ writing

    def add_episode(
        self,
        state: np.ndarray,
        action: np.ndarray,
        eef_pose: np.ndarray,
        task: str,
        frames: dict[str, np.ndarray] | None = None,
        extra: dict | None = None,
        phase: np.ndarray | None = None,
    ) -> int:
        """Write one episode and return its index.

        ``state`` and ``action`` are ``[T, 8]`` (joints [rad], then gripper in 0..1). ``eef_pose``
        is ``[T, 7]`` (position [m], then wxyz). ``frames`` is ``{camera: uint8 [T, H, W, 3]}``. ``phase`` is
        ``[T]`` ints, the planner stage that produced each step; omitted it is stored as ``-1``.
        """
        length = len(action)
        state, action, eef_pose = (np.asarray(x, dtype=np.float32) for x in (state, action, eef_pose))
        if state.shape != (length, 8) or action.shape != (length, 8) or eef_pose.shape != (length, 7):
            raise ValueError(
                f"expected state/action [T, 8] and eef_pose [T, 7], got {state.shape}, {action.shape}, {eef_pose.shape}"
            )
        if length < 1:
            raise ValueError("an episode needs at least one step")
        phase = np.full(length, NO_PHASE, dtype=np.int64) if phase is None else np.asarray(phase, dtype=np.int64)
        if phase.shape != (length,):
            raise ValueError(f"phase must be [T], got {phase.shape}")
        frames = frames or {}
        if set(frames) != set(self.camera_hw):
            raise ValueError(f"frames for {sorted(frames)} do not match cameras {sorted(self.camera_hw)}")
        index = len(self.episodes)
        chunk = index // CHUNK_SIZE
        task_index = self.tasks.setdefault(task, len(self.tasks))
        self._write_parquet(index, chunk, task_index, state, action, eef_pose, phase)
        for name, images in frames.items():
            self._write_video(name, index, chunk, images)
        record = {"episode_index": index, "tasks": [task], "length": length}
        record.update(extra or {})
        self.episodes.append(record)
        self._stats.append(self._episode_stats(index, state, action, eef_pose, frames, phase))
        self.frames += length
        self._flush_meta()
        return index

    def close(self) -> None:
        """Write metadata. Also called after every episode, so a crash keeps what was recorded."""
        self._flush_meta()

    # ------------------------------------------------------------------ files

    def _write_parquet(self, index, chunk, task_index, state, action, eef_pose, phase) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        length = len(action)
        offset = self.frames
        vector = pa.list_(pa.float32())
        table = pa.table(
            {
                "observation.state": pa.array(state.tolist(), type=vector),
                "observation.eef_pose": pa.array(eef_pose.tolist(), type=vector),
                "action": pa.array(action.tolist(), type=vector),
                "timestamp": pa.array((np.arange(length) / self.fps).astype(np.float32)),
                "frame_index": pa.array(np.arange(length, dtype=np.int64)),
                "episode_index": pa.array(np.full(length, index, dtype=np.int64)),
                "index": pa.array(np.arange(offset, offset + length, dtype=np.int64)),
                "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
                PHASE_KEY: pa.array(phase),
            }
        )
        path = self.root / DATA_PATH.format(episode_chunk=chunk, episode_index=index)
        atomic_write(path, lambda temp: pq.write_table(table, temp))

    def _write_video(self, camera: str, index: int, chunk: int, images: np.ndarray) -> None:
        try:
            import av
        except ImportError as error:
            raise ImportError("writing videos needs PyAV: install the `record` extra (`pip install av`)") from error
        height, width = self.camera_hw[camera]
        if images.shape[1:] != (height, width, 3) or images.dtype != np.uint8:
            raise ValueError(
                f"{camera} frames must be uint8 [T, {height}, {width}, 3], got {images.dtype} {images.shape}"
            )
        path = self.root / VIDEO_PATH.format(episode_chunk=chunk, video_key=IMAGE_PREFIX + camera, episode_index=index)

        def encode(temp: Path) -> None:
            with av.open(str(temp), "w", format="mp4") as container:
                stream = container.add_stream("libx264", rate=self.fps)
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                stream.options = {"crf": "23", "preset": "fast"}
                for image in images:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

        atomic_write(path, encode)

    # ------------------------------------------------------------------ metadata

    def _flush_meta(self) -> None:
        # ``episodes.jsonl`` goes last among the row files and ``info.json`` after it: a reopened dataset trusts them
        # in that order.
        meta = self.root / "meta"
        _write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": t} for t, i in self.tasks.items()])
        _write_jsonl(meta / "episodes_stats.jsonl", self._stats)
        _write_jsonl(meta / "episodes.jsonl", self.episodes)
        atomic_write_text(meta / "modality.json", json.dumps(self._modality(), indent=2))
        atomic_write_text(meta / "info.json", json.dumps(self._info(), indent=2))

    def _info(self) -> dict:
        features = {
            "observation.state": {"dtype": "float32", "shape": [8], "names": STATE_NAMES},
            "observation.eef_pose": {"dtype": "float32", "shape": [7], "names": EEF_NAMES},
            "action": {"dtype": "float32", "shape": [8], "names": ACTION_NAMES},
        }
        for name, (height, width) in self.camera_hw.items():
            features[IMAGE_PREFIX + name] = {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": float(self.fps),
                    "video.height": height,
                    "video.width": width,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        for name, dtype in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64")):
            features[name] = {"dtype": dtype, "shape": [1], "names": None}
        features["index"] = {"dtype": "int64", "shape": [1], "names": None}
        features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
        features[PHASE_KEY] = {"dtype": "int64", "shape": [1], "names": None}
        episodes = len(self.episodes)
        return {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": episodes,
            "total_frames": self.frames,
            "total_tasks": len(self.tasks),
            "total_videos": episodes * len(self.camera_hw),
            "total_chunks": (episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
            "chunks_size": CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{episodes}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self.camera_hw else None,
            "features": features,
            "phase_names": self.phase_names,
        }

    def _modality(self) -> dict:
        return {
            "embodiment": "franka_2f85_jointpos",
            "state": {"joint_position": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}},
            "action": {"absolute_joint_position": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}},
            "video": {name: {"original_key": IMAGE_PREFIX + name} for name in self.camera_hw},
            "annotation": {
                "human.task_description": {"original_key": "task_index"},
                "phase": {"original_key": PHASE_KEY},
            },
        }

    def _episode_stats(self, index: int, state, action, eef_pose, frames, phase) -> dict:
        stats = {
            "observation.state": _vector_stats(state),
            "observation.eef_pose": _vector_stats(eef_pose),
            "action": _vector_stats(action),
        }
        for name, images in frames.items():
            sample = images[::_VIDEO_STAT_STRIDE].astype(np.float64) / 255.0
            stats[IMAGE_PREFIX + name] = {
                "min": sample.min(axis=(0, 1, 2)).reshape(3, 1, 1).tolist(),
                "max": sample.max(axis=(0, 1, 2)).reshape(3, 1, 1).tolist(),
                "mean": sample.mean(axis=(0, 1, 2)).reshape(3, 1, 1).tolist(),
                "std": sample.std(axis=(0, 1, 2)).reshape(3, 1, 1).tolist(),
                "count": [len(sample)],
            }
        length = len(action)
        for key, values in (
            ("timestamp", np.arange(length) / self.fps),
            ("frame_index", np.arange(length)),
            ("episode_index", np.full(length, index)),
            ("index", np.arange(self.frames, self.frames + length)),
            (PHASE_KEY, phase),
        ):
            stats[key] = _vector_stats(np.asarray(values, dtype=np.float64).reshape(-1, 1))
        return {"episode_index": index, "stats": stats}

    def _resume(self) -> None:
        meta = self.root / "meta"
        info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
        if int(info["fps"]) != self.fps:
            raise ValueError(f"dataset runs at {info['fps']} Hz, not {self.fps} Hz")
        cameras = {
            key[len(IMAGE_PREFIX) :]: tuple(value["shape"][:2])
            for key, value in info["features"].items()
            if key.startswith(IMAGE_PREFIX)
        }
        if cameras != self.camera_hw:
            raise ValueError(f"dataset cameras {cameras} differ from {self.camera_hw}")
        stored_phases = list(info.get("phase_names") or [])
        if self.phase_names and stored_phases and stored_phases != self.phase_names:
            raise ValueError(f"dataset phases {stored_phases} differ from {self.phase_names}")
        self.phase_names = self.phase_names or stored_phases
        # An interrupted flush can leave one row file a row ahead of another: keep the episodes all of them know.
        episodes = _read_jsonl(meta / "episodes.jsonl")
        stats = _read_jsonl(meta / "episodes_stats.jsonl")
        keep = min(len(episodes), len(stats))
        self.episodes, self._stats = episodes[:keep], stats[:keep]
        self.tasks = {row["task"]: int(row["task_index"]) for row in _read_jsonl(meta / "tasks.jsonl")}
        self.frames = sum(int(row["length"]) for row in self.episodes)


def _vector_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
