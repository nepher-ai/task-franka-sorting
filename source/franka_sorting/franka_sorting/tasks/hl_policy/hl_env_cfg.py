# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical env: the planner writes tracker targets, and the tracker checkpoint is the policy.

``task_id`` is ``"all"``, one task id, or comma-separated task ids. The dome is the one background the
selected tasks share.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

from franka_sorting.benchmark.bundle import ALL_TASKS, DEFAULT_ENV_ID, load_benchmark, resolve_bundle
from franka_sorting.benchmark_select import select_tasks, shared_background
from franka_sorting.frames import EE_BODY
from franka_sorting.scene.cameras import OBSERVATION_CAMERAS, selected_cameras
from franka_sorting.scene.scene_cfg import SortingSceneCfg, apply_sorting_physics, apply_stage_assets

from ..ll_policy.ll_env_cfg import ActionsCfg, ObservationsCfg
from . import mdp
from .highlight import HighlightCfg, HighlightPreset, rendering_camera


@configclass
class CommandsCfg:
    """Planner targets under the names the tracker was trained on."""

    ee_pose = mdp.HLPoseCommandCfg(asset_name="robot", body_name=EE_BODY, resampling_time_range=(1.0e6, 1.0e6))
    grip_cmd = mdp.HLGripCommandCfg(resampling_time_range=(1.0e6, 1.0e6))


@configclass
class EventCfg:
    """Each env's layout on the stage before physics starts. On reset: default robot pose, arm noise, layout."""

    place_layout = EventTerm(func=mdp.place_planned_layout, mode="prestartup")
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_arm = EventTerm(func=mdp.reset_arm_start_noise, mode="reset")
    reset_layout = EventTerm(func=mdp.reset_sorting_layout, mode="reset")


@configclass
class TerminationsCfg:
    """Success held for 0.5 s, a failure-only stop for a lost object, and the episode limit."""

    success = DoneTerm(func=mdp.task_success)
    object_lost = DoneTerm(func=mdp.object_lost)
    time_out = DoneTerm(func=mdp.episode_timeout, time_out=True)


@configclass
class RewardsCfg:
    """No reward: success is scored by ``benchmark.success``."""


@configclass
class HLEnvCfg(ManagerBasedRLEnvCfg):
    """Planner plus tracker on the benchmark stage. ``scene.num_envs`` must be a multiple of the task count."""

    scene: SortingSceneCfg = SortingSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    env_id: str = DEFAULT_ENV_ID
    task_id: str = ALL_TASKS
    randomize_layout: bool = False
    """Draw a fresh random layout at every episode start, on the bodies of the selected task."""
    arm_start_noise_rad: float = 0.05
    """Half range [rad] of the uniform offset on each arm joint at reset. Zero keeps the ready pose."""
    cameras: tuple[str, ...] = ()
    """Observation cameras to spawn. Empty spawns none."""
    layout_seed: int = 1
    """Seed of the random layouts."""
    highlight: HighlightPreset = HighlightPreset()
    highlight_tasks: tuple[int, ...] = ()
    """Task indices or an inclusive range such as ``0-15``. Empty samples up to 16 tasks."""
    task_ids: tuple[str, ...] = ()
    """Tasks of the stage, filled from ``task_id``."""
    stage_plan: object | None = None
    """Stage plan for ``scene.num_envs``. ``HLSortingEnv`` fills it after that count is final."""

    def __post_init__(self):
        apply_sorting_physics(self)
        # Kit otherwise paints a solid sky over the background dome. None keeps the dome texture.
        self.sim.default_visualizer_cfg = VisualizerCfg(background_color=None)
        self.observations.policy.enable_corruption = False
        bundle = resolve_bundle(self.env_id)
        benchmark = load_benchmark(bundle)
        self.task_ids = select_tasks(benchmark, str(self.task_id))
        # Bodies are mounted once num_envs is final. The dome is the background of the selected tasks. A mixed
        # selection keeps the stage default here and is rejected when the env syncs its selection.
        try:
            dome = shared_background(benchmark, self.task_ids)
        except ValueError:
            dome = None
        apply_stage_assets(self, bundle, benchmark.stage, background=dome)
        self.scene.num_envs = len(self.task_ids)
        keep = selected_cameras(self.cameras)
        self.cameras = keep
        for name in OBSERVATION_CAMERAS:
            if name not in keep:
                setattr(self.scene, name, None)
        command = self.commands.ee_pose
        command.env_id = self.env_id
        command.task_ids = self.task_ids
        command.body_name = EE_BODY
        command.randomize_layout = self.randomize_layout
        command.layout_seed = self.layout_seed

    def sync_task_selection(self) -> None:
        """Apply ``task_id`` after play overrides.

        ``__post_init__`` runs before ``env.task_id`` and ``--num_envs``. A scene count still equal
        to the previous task count follows the new selection. Any other count is kept.
        """
        bundle = resolve_bundle(self.env_id)
        benchmark = load_benchmark(bundle)
        selected = select_tasks(benchmark, str(self.task_id))
        # One run renders one dome, so tasks of different backgrounds cannot share a stage.
        dome = shared_background(benchmark, selected)
        previous = tuple(self.task_ids)
        if previous == selected:
            return
        apply_stage_assets(self, bundle, benchmark.stage, background=dome)
        if int(self.scene.num_envs) == len(previous):
            self.scene.num_envs = len(selected)
        self.task_ids = selected
        command = self.commands.ee_pose
        command.task_ids = selected
        command.env_id = self.env_id
        command.randomize_layout = self.randomize_layout
        command.layout_seed = self.layout_seed

    def play_mode(self):
        """Play defaults. Highlight adds an unspawned camera so Kit loads its rendering app.

        The base play mode caps the scene at 50 envs; the task count is restored before ``--num_envs``.
        """
        task_count = len(self.task_ids)
        super().play_mode()
        self.scene.num_envs = task_count
        if isinstance(self.highlight, HighlightCfg):
            self.rendering_camera = rendering_camera()
        else:
            self.highlight = None


@configclass
class HLEnvCfg_PLAY(HLEnvCfg):
    """Evaluation layout: one env per task, or a whole number of extra copies through ``scene.num_envs``."""
