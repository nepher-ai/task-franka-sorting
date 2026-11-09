# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One data-generation shard: one Isaac Lab process on one stage. Started by ``record_lerobot.py``.

The shard spec names the tasks and their episode quotas. Episodes that pass every gate are appended to the dataset
with their metadata; the quota is measured against what the dataset already holds, so a rerun only records what is
missing. The result of the run is written to ``--result`` for the parent.

Run through the parent. To run a shard by hand::

    isaaclab.bat -p scripts/record_shard.py --spec <output>/_sdg/shard_000.json --visualizer none
"""

# Launch Isaac Sim first. Modules that import pxr must load after the app starts.

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from franka_sorting.data.atomic import atomic_write_text
from franka_sorting.data.quality import QualityLimits
from franka_sorting.data.randomize import RandomizeConfig
from franka_sorting.data.sdg_plan import ShardSpec, remaining_quota

parser = argparse.ArgumentParser(description="Record one shard of planner demonstrations.")
parser.add_argument("--spec", type=str, required=True, help="Shard spec written by record_lerobot.py.")
parser.add_argument("--result", type=str, default=None, help="Where to write the result. Next to the spec by default.")

args_cli, _ = parser.parse_known_args()
SPEC = json.loads(Path(args_cli.spec).read_text(encoding="utf-8"))
SHARD = ShardSpec(**SPEC["shard"])
OPTIONS = SPEC["options"]
OUTPUT = Path(SPEC["output"])
RESULT = Path(args_cli.result or Path(args_cli.spec).with_suffix(".result.json"))


def read_episodes() -> list[dict]:
    path = OUTPUT / "meta" / "episodes.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_result(payload: dict) -> None:
    atomic_write_text(RESULT, json.dumps(payload, indent=2))


QUOTA = remaining_quota(read_episodes(), SHARD, count_failed=False)
ACTIVE = [task for task in SHARD.tasks if QUOTA[task] > 0]
if not ACTIVE:
    write_result({"complete": True, "tried": 0, "saved": 0, "rejected": {}, "elapsed_s": 0.0})
    print(f"[SHARD {SHARD.index}] nothing left to record", flush=True)
    sys.exit(0)

# The remaining imports start Isaac Sim.
from isaaclab.app import AppLauncher  # noqa: E402

# Other arguments (--headless, --device, ...) belong to the AppLauncher.
launcher_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(launcher_parser)
launcher_args, _ = launcher_parser.parse_known_args([a for a in sys.argv[1:] if a not in ("--spec", args_cli.spec)])
if OPTIONS["cameras"]:
    launcher_args.enable_cameras = True
app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app

import franka_sorting.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from franka_sorting.data.lerobot_writer import LeRobotWriter  # noqa: E402
from franka_sorting.data.quality import check_episode  # noqa: E402
from franka_sorting.paths import sync_best_policy  # noqa: E402
from franka_sorting.scene.observations import camera_rgb, eef_pose, gripper_pos, joint_pos, resize_rgb  # noqa: E402
from franka_sorting.scene.robot import GRIPPER_CLOSED_RAD  # noqa: E402
from franka_sorting.tasks.hl_policy import mdp  # noqa: E402
from franka_sorting.tasks.hl_policy.hl_env_cfg import HLEnvCfg_PLAY  # noqa: E402
from franka_sorting.tasks.hl_policy.planner import Stage  # noqa: E402

from isaaclab.managers import EventTermCfg  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

GYM_ID = "Nepher-FrankaSorting-HL-Play-v0"
# Runner of the EnvhubPlay entry: resolves best_policy/best_policy.pt, never the latest training log.
AGENT_GYM_ID = "Nepher-FrankaSorting-HL-EnvhubPlay-v0"
PHASE_NAMES = [stage.name.lower() for stage in Stage]
# A failed task is a normal outcome of --keep_failed. Anything else (a bad start, a jump, blank video) never is.
TASK_FAILURES = {"not_success", "lost_part"}


def _host(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def layout_record(layout) -> dict:
    """Full description of an episode's layout: tables, containers, types, and every part pose."""
    parts = [
        {
            "slot": p.slot,
            "type": p.asset.variant,
            "position_m": [round(float(v), 5) for v in p.position_m],
            "quat_wxyz": [round(float(v), 5) for v in p.quat_wxyz],
            "scale": None if p.scale is None else [round(float(v), 4) for v in p.scale],
        }
        for p in layout.placements
        if p.asset.kind == "item"
    ]
    anchors = {
        p.slot: {
            "variant": p.asset.variant,
            "position_m": [round(float(v), 5) for v in p.position_m],
            "quat_wxyz": [round(float(v), 5) for v in p.quat_wxyz],
        }
        for p in layout.placements
        if p.asset.kind in ("container", "tray")
    }
    return {"parts": parts, "fixtures": anchors}


def main() -> None:
    started = time.time()
    installed_version = check_rsl_rl_version()
    agent_cfg = load_cfg_from_registry(AGENT_GYM_ID, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    checkpoint = sync_best_policy(
        agent_cfg.experiment_name,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
        explicit_checkpoint=OPTIONS["checkpoint"],
    )
    randomize = RandomizeConfig(**OPTIONS["randomize"])
    limits = QualityLimits(**OPTIONS["quality"])
    cameras = tuple(OPTIONS["cameras"])
    size = tuple(OPTIONS["image_hw"])
    seed = SHARD.layout_seed()

    cfg = HLEnvCfg_PLAY(
        env_id=SHARD.bundle,
        task_id=",".join(ACTIVE),
        randomize_layout=SHARD.randomize,
        layout_seed=seed,
        arm_start_noise_rad=float(OPTIONS["arm_noise_rad"]),
        cameras=cameras,
    )
    cfg.seed = seed % 2**31
    if randomize.lighting or randomize.camera_jitter > 0.0:
        cfg.events.randomize_visuals = EventTermCfg(
            func=mdp.randomize_visuals, mode="reset", params={"config": randomize.as_dict(), "seed": seed}
        )
    task_count = len(ACTIVE)
    wanted = int(OPTIONS["num_envs"]) or (8 if cameras else 16)
    cfg.scene.num_envs = -(-wanted // task_count) * task_count
    print(
        f"[SHARD {SHARD.index}] {task_count} tasks, {cfg.scene.num_envs} envs, background {SHARD.background}",
        flush=True,
    )
    env = gym.make(GYM_ID, cfg=cfg)
    unwrapped = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=unwrapped.device)

    command = unwrapped.command_manager.get_term("ee_pose")
    tracker, planner = command.tracker, command.planner
    arm_action = unwrapped.action_manager.get_term("arm_action")
    fps = round(1.0 / unwrapped.step_dt)
    writer = LeRobotWriter(OUTPUT, fps=fps, camera_hw={name: size for name in cameras}, phase_names=PHASE_NAMES)

    count = unwrapped.num_envs
    keys = ("state", "action", "eef", "phase", *cameras)
    buffers = [{key: [] for key in keys} for _ in range(count)]
    retries = [0] * count
    draws = [dict(getattr(unwrapped, "visual_draws", [{}] * count)[i]) for i in range(count)]
    need = dict(QUOTA)
    tried = 0
    saved_total = 0
    rejected: Counter = Counter()
    budget = int(OPTIONS["max_tries_factor"] * sum(need.values())) + count
    obs = env.get_observations()
    for step in range(int(OPTIONS["max_steps"]) * budget):
        if step % 300 == 0:
            print(
                f"[INFO] shard {SHARD.index} step {step}: {tried} tried, {saved_total} saved, "
                f"{time.time() - started:.0f} s",
                flush=True,
            )
        state = torch.cat([joint_pos(unwrapped), gripper_pos(unwrapped)], dim=-1)
        pose = eef_pose(unwrapped)
        images = {name: resize_rgb(camera_rgb(unwrapped, name), size) for name in cameras}
        phase_h = planner.stage_ids()
        for index in range(count):
            retries[index] = planner.retry_count(index)
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        joint_command = arm_action.joint_command
        action = torch.cat([joint_command[:, :7], (joint_command[:, 7:8] > 0.5 * GRIPPER_CLOSED_RAD).float()], dim=-1)
        state_h, pose_h, action_h = _host(state), _host(pose), _host(action)
        images_h = {name: _host(frames) for name, frames in images.items()}
        done_h = _host(dones).astype(bool)
        for index in range(count):
            buffer = buffers[index]
            buffer["state"].append(state_h[index])
            buffer["eef"].append(pose_h[index])
            buffer["action"].append(action_h[index])
            buffer["phase"].append(phase_h[index])
            for name in cameras:
                buffer[name].append(images_h[name][index])
            if not done_h[index]:
                continue
            tried += 1
            task_id = tracker.last_task_ids[index]
            layout = tracker.last_layouts[index]
            success = bool(tracker.last_success[index])
            arrays = {key: np.stack(buffer[key]) for key in keys}
            frames = {name: arrays[name] for name in cameras}
            reasons = check_episode(arrays["state"], arrays["action"], frames, limits)
            if tracker.last_invalid_start[index]:
                reasons.append("invalid_start")
            if tracker.last_lost[index]:
                reasons.append("lost_part")
            if not success:
                reasons.append("not_success")
            hard = [reason for reason in reasons if reason not in TASK_FAILURES]
            keep = not hard and (not reasons or (OPTIONS["keep_failed"] and success is False))
            if need.get(task_id, 0) <= 0 and success:
                keep = False
                rejected["quota_full"] += 1
            elif not keep:
                rejected.update(reasons or ["rejected"])
                print(f"[INFO] shard {SHARD.index} {task_id}: episode rejected {reasons}", flush=True)
            if keep:
                extra = {
                    "success": success,
                    "task_id": task_id,
                    "shard": SHARD.index,
                    "source": SHARD.source,
                    "bundle": Path(SHARD.bundle).name,
                    "seed": seed,
                    "episode_key": tracker.last_episode_keys[index],
                    "background": layout.background,
                    "table": layout.surface,
                    "container": layout.container,
                    "tray": layout.tray,
                    "cell_map": list(layout.cell_map),
                    "n_parts": sum(1 for p in layout.placements if p.asset.kind == "item"),
                    "progress": float(tracker.last_progress[index]),
                    "steps": len(arrays["action"]),
                    "planner_retries": int(retries[index]),
                    "reject_reasons": reasons,
                    "randomization": draws[index],
                    "layout": layout_record(layout),
                }
                writer.add_episode(
                    arrays["state"],
                    arrays["action"],
                    arrays["eef"],
                    task=layout.instruction,
                    frames=frames,
                    extra=extra,
                    phase=arrays["phase"],
                )
                if success:
                    need[task_id] -= 1
                    saved_total += 1
                print(
                    f"[INFO] shard {SHARD.index} {task_id}: episode saved "
                    f"({extra['steps']} steps, {success=}, {saved_total}/{sum(QUOTA.values())})",
                    flush=True,
                )
            draws[index] = dict(getattr(unwrapped, "visual_draws", [{}] * count)[index])
            retries[index] = 0
            for key in buffer:
                buffer[key] = []
        if all(value <= 0 for value in need.values()):
            break
    complete = all(value <= 0 for value in need.values())
    env.close()
    write_result(
        {
            "complete": complete,
            "tried": tried,
            "saved": saved_total,
            "rejected": dict(rejected),
            "remaining": need,
            "elapsed_s": time.time() - started,
        }
    )
    print(f"[RESULT] shard {SHARD.index}: {saved_total} saved, {tried} tried, complete={complete}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
