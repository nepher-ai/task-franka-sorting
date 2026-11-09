# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from franka_sorting.benchmark.bundle import load_benchmark, resolve_bundle
from franka_sorting.benchmark.stage import xyzw_from_wxyz
from franka_sorting.benchmark.stage_plan import entity_pose, surface_body_pose
from franka_sorting.data.randomize import RandomizeConfig, draw_camera, draw_lighting
from franka_sorting.scene.robot import ARM_JOINTS

from ..start_noise import arm_start_offsets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_sorting_layout(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pose_command_name: str = "ee_pose") -> None:
    """Write this scene's poses over the spawn pose. Run after ``reset_scene_to_default``."""
    env.command_manager.get_term(pose_command_name).tracker.begin(env_ids)


def reset_arm_start_noise(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice) -> None:
    """Offset the arm from the ready pose by seeded noise. Run after ``reset_scene_to_default``.

    The half range is ``env.cfg.arm_start_noise_rad`` [rad]. The offsets follow the seed, the reset round and the
    env index, so every full reset of an evaluation starts from a different but reproducible pose.
    """
    half_range = float(env.cfg.arm_start_noise_rad)
    if half_range <= 0.0:
        return
    robot = env.scene["robot"]
    if isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        ids = torch.as_tensor(env_ids, device=env.device).long()
    joint_ids = robot.find_joints(ARM_JOINTS)[0]
    index = ids.tolist()
    offsets = arm_start_offsets(
        env.start_noise_seed,
        env.start_noise_round,
        index,
        half_range,
        len(joint_ids),
        episode_counts=env.start_noise_resets[index],
    )
    env.start_noise_resets[index] += 1
    rows = ids[:, None]
    position = robot.data.default_joint_pos.torch[rows, joint_ids].clone()
    position += torch.as_tensor(offsets, device=env.device, dtype=position.dtype)
    limits = robot.data.soft_joint_pos_limits.torch[rows, joint_ids]
    position = position.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_position_target_index(target=position, joint_ids=joint_ids, env_ids=env_ids)


def place_planned_layout(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Spawn every env's bodies at that env's own layout, before physics starts.

    A body is created once for all envs, at the pose of the first task that uses it. Without this the
    warm-up step and the kinematic setup see the wrong layout, and the first reset then teleports the
    kinematic table and shelves under the objects of the other envs.
    """
    plan = env.cfg.stage_plan
    benchmark = load_benchmark(resolve_bundle(env.cfg.env_id))
    env_layouts = [benchmark.task(plan.task_ids[task]) for task in plan.assignment[0]]
    stage = env.sim.stage
    for index in range(plan.num_envs):
        root = f"{env.scene.env_ns}/env_{index}"
        for entity_index, name in enumerate(plan.entities):
            _place(stage, f"{root}/{name}", *entity_pose(plan, env_layouts, index, entity_index))
        for surface_index, name in enumerate(plan.surface_entities):
            leaf = env.scene[name].cfg.prim_path.rsplit("/", 1)[-1]
            _place(stage, f"{root}/{leaf}", *surface_body_pose(plan, index, surface_index))


def _place(stage, path: str, position, quat_wxyz) -> None:
    sim_utils.standardize_xform_ops(
        stage.GetPrimAtPath(path),
        translation=tuple(float(value) for value in position),
        orientation=xyzw_from_wxyz(tuple(float(value) for value in quat_wxyz)),
    )


DOME_PATH = "/World/background"
RANDOMIZED_CAMERAS = ("cam_exterior_left", "cam_exterior_right", "cam_front")


def randomize_visuals(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, config: dict, seed: int) -> None:
    """Draw the lights and camera extrinsics of the episode that starts in ``env_ids``.

    The sphere light and the exterior cameras are per env and change at every reset. The dome is one prim shared
    by every env, so it is drawn once, at the first reset, and stays for the run. The draws follow ``seed``, the
    env, and its reset count; the latest draw of each env is in ``env.visual_draws``.

    Args:
        config: :meth:`RandomizeConfig.as_dict` of the run.
        seed: Base seed of the draws.
    """
    cfg = RandomizeConfig(**config)
    if isinstance(env_ids, slice):
        ids = list(range(env.num_envs))[env_ids]
    else:
        ids = [int(i) for i in torch.as_tensor(env_ids).reshape(-1).tolist()]
    if not hasattr(env, "visual_draws"):
        env.visual_draws = [{} for _ in range(env.num_envs)]
        env._visual_resets = np.zeros(env.num_envs, dtype=int)
        env._sphere_nominal = {}
        env._dome_draw = _apply_dome(env, cfg, seed)
    stage = env.sim.stage
    cameras = [name for name in RANDOMIZED_CAMERAS if name in env.scene.keys() and cfg.camera_jitter > 0.0]
    for index in ids:
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, int(env._visual_resets[index])]))
        env._visual_resets[index] += 1
        light = draw_lighting(rng, cfg)
        draw = {"lighting": light.as_dict(), "dome": env._dome_draw}
        if cfg.lighting:
            _apply_sphere(env, stage, index, light)
        shifts = {}
        for name in cameras:
            camera = draw_camera(rng, cfg)
            _apply_camera(env, name, index, camera)
            shifts[name] = camera.as_dict()
        draw["cameras"] = shifts
        env.visual_draws[index] = draw


def _apply_dome(env, cfg: RandomizeConfig, seed: int) -> dict:
    light = draw_lighting(np.random.default_rng(np.random.SeedSequence([seed, 1 << 20])), cfg)
    if not cfg.lighting:
        return {"dome_scale": 1.0, "dome_yaw_deg": 0.0}
    prim = env.sim.stage.GetPrimAtPath(DOME_PATH)
    attribute = prim.GetAttribute("inputs:intensity")
    attribute.Set(float(attribute.Get()) * light.dome_scale)
    half = math.radians(light.dome_yaw_deg) / 2.0
    sim_utils.standardize_xform_ops(prim, orientation=(0.0, 0.0, math.sin(half), math.cos(half)))
    return {"dome_scale": light.dome_scale, "dome_yaw_deg": light.dome_yaw_deg}


def _apply_sphere(env, stage, index: int, light) -> None:
    prim = stage.GetPrimAtPath(f"{env.scene.env_ns}/env_{index}/sphere")
    if index not in env._sphere_nominal:
        translate = prim.GetAttribute("xformOp:translate").Get()
        env._sphere_nominal[index] = (
            float(prim.GetAttribute("inputs:intensity").Get()),
            tuple(float(v) for v in translate),
        )
    intensity, position = env._sphere_nominal[index]
    prim.GetAttribute("inputs:intensity").Set(intensity * light.sphere_scale)
    shifted = tuple(base + shift for base, shift in zip(position, light.sphere_offset_m))
    sim_utils.standardize_xform_ops(prim, translation=shifted)


def _apply_camera(env, name: str, index: int, draw) -> None:
    """Move one env's camera from its stage pose by ``draw`` (a shift in the env frame and a world rotation)."""
    camera = env.scene[name]
    offset = camera.cfg.offset
    origin = env.scene.env_origins[index]
    device = origin.device
    position = origin + torch.tensor(offset.pos, device=device, dtype=origin.dtype)
    nominal = torch.tensor(offset.rot, device=device, dtype=origin.dtype)
    rotvec = torch.tensor(draw.rotvec_rad, device=device, dtype=origin.dtype)
    angle = torch.linalg.norm(rotvec)
    axis = rotvec / angle if float(angle) > 0.0 else torch.tensor([0.0, 0.0, 1.0], device=device, dtype=origin.dtype)
    small = torch.cat([axis * torch.sin(angle / 2), torch.cos(angle / 2).reshape(1)])
    rotated = math_utils.quat_mul(small.unsqueeze(0), nominal.unsqueeze(0))
    shifted = position + torch.tensor(draw.shift_m, device=device, dtype=origin.dtype)
    camera.set_world_poses(shifted.unsqueeze(0), rotated, env_ids=[index], convention=offset.convention)
