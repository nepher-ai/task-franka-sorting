# Franka parts sorting in NVIDIA Isaac Lab

**sorting planner · learned tool-pose tracker · damped least-squares IK · 8-D joint targets**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![rsl_rl](https://img.shields.io/badge/rsl__rl-PPO-orange.svg)](https://github.com/leggedrobotics/rsl_rl)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

![Franka parts sorting in Isaac Sim: bins, crates, and compartment trays](images/landing.png)

**task-franka-sorting** is a [Nepher Robotics](https://github.com/nepher-ai) extension for [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab) / [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html). It runs a **Franka** arm with a **Robotiq 2F-85** gripper that picks industrial parts out of a bin or crate and sorts them into the cells of a parts tray. Each cell takes one part type. The mapping is random per task and stated in the instruction. A sorting planner chooses each grasp and place. An **RSL-RL PPO** policy tracks that tool pose. **Damped least-squares IK** turns the tracker action into **8-D absolute joint targets**.

Use this repository when you need Isaac Lab Gym environments for Franka **parts sorting** (`Nepher-FrankaSorting-LL-*`, `Nepher-FrankaSorting-HL-*`), the EnvHub benchmark `franka-sorting-parts-v0`, or sharded **LeRobot v2.1** synthetic data.

| | |
|---|---|
| **Robot** | Franka with a Robotiq 2F-85 gripper |
| **Simulator** | NVIDIA Isaac Sim 6.1.0 + Isaac Lab 3.0 |
| **Algorithm** | rsl_rl PPO tool-pose tracker |
| **Stack** | Sorting planner → PPO tracker → damped least-squares IK → 8-D joint targets |
| **Rates** | Physics 240 Hz. Commands every 16 steps (15 Hz) |
| **Tracker observation** | 43 values: arm position and velocity, gripper, tool pose, pose error, pose target, grip target, last action |
| **Tracker action** | 7-D tool-pose delta plus gripper, integrated to absolute joint targets |
| **Joint action** | 8 values: 7 absolute joint targets [rad], gripper 0 (open) or 1 (closed) |
| **Episode** | 40 s. The bin holds 20-40 parts. A task sorts 3-6 part types |
| **Success** | At least 4, 5, or 6 parts in the instructed cells, held for 0.5 s. Parts left in the container are allowed |
| **Eval** | EnvHub `franka-sorting-parts-v0` (16 tasks) |
| **Cameras** | `cam_exterior_left`, `cam_exterior_right`, `cam_wrist`, `cam_front` |
| **Dataset** | LeRobot v2.1, sharded and resumable, written by `scripts/record_lerobot.py` |
| **License** | BSD 3-Clause |

## Who this is for

- Researchers training a **Franka tool-pose tracker** in Isaac Lab, then running it under a classical sorting planner.
- Tournament evaluation on the EnvHub bundle **`franka-sorting-parts-v0`**. The high-level env puts every task of the bundle in one stage.
- Synthetic data for imitation learning. This repository records **LeRobot v2.1** episodes itself. `sdg-gen` can run the same Gym task from `configs/task-franka-sorting.yaml`.

The package ships the robot USD. Containers, tables, parts, trays, and `*.hdr` backgrounds live in the EnvHub bundles. Those bundles are written by the separate `build_sorting_benchmark` project. `best_policy/best_policy.pt` is the published tracker checkpoint (rsl_rl, weights only), fine-tuned from the tabletop tracker.

## How it runs

```
sorting planner (SortingPlanner) -> learned tool-pose tracker (ll_policy) -> damped-least-squares IK -> 8-D joint targets
```

Train the tracker first, then play the planner on top of that checkpoint.

- **High level.** `SortingPlanner` picks one part at a time out of the container and sets it into its tray cell. Motion is a rest-to-rest polyline, so the tool cruises at the speed the evaluator rewards and stops at the grasp and at the release. Every placement is checked against the benchmark `in_cell` predicate. A part that leaves its cell is picked up again.
- **Low level.** The PPO policy tracks the planner's tool pose and gripper command. Its 7-D action is a tool-pose delta plus the gripper. `IkJointCommandAction` turns that into absolute joint targets, the same 8-D action space a downstream policy writes. Training uses 4096 envs, observation noise, and an 8 s episode. About 35% of the pose targets are level, leaning, or pitched, so the tracker keeps the poses the planner uses at the edge of the workspace.
- **Scene.** Layout sampling and validation live in `franka_sorting/layout/`. Benchmark loading and success live in `franka_sorting/benchmark/`. The stage lives in `franka_sorting/scene/`. `scripts/link_cache.py` links the EnvHub bundles into the nepher cache, so no download is needed.

### Episode success

The success expression has to hold on the current state and stay true for 0.5 s. A published task asks for at least 4, 5, or 6 parts in the cells named by the instruction, inside the 40 s episode. Parts still in the bin do not fail the task. A lost object, one that falls off the table or leaves the workspace, ends the episode as a failure and never counts as success. The checks are in `franka_sorting/benchmark/success.py`.

## Gym environments

| Id | Use |
| --- | --- |
| `Nepher-FrankaSorting-LL-v0` | Train the tracker (4096 envs, observation noise) |
| `Nepher-FrankaSorting-LL-Play-v0` | Play the tracker (32 envs, no noise) |
| `Nepher-FrankaSorting-HL-Play-v0` | Planner plus tracker on every selected task in one stage. Loads the latest tracker checkpoint |
| `Nepher-FrankaSorting-HL-EnvhubPlay-v0` | The same layout, for eval-nav. Loads `best_policy/best_policy.pt` |

`scripts/list_envs.py` lists them. `./isaaclab.sh train` and `play` discover these through the `isaaclab.tasks` entry point. The Gym registration names rsl_rl, so `--rl_library` is not passed.

Both high-level tasks load the EnvHub benchmark `franka-sorting-parts-v0`, 16 tasks.

**One stage, all selected tasks.** The high-level env builds one stage for the selected tasks (`env.task_id=all`, or one task id). `scene.num_envs` defaults to one env per task. `--num_envs` is that count or a whole multiple of it (`16`, `32`, `64`, ...). Env `i` runs task `i % T` and resets on its own, and every selected task is in the stage together.

## Benchmarks

| Bundle | Tasks | Content |
| --- | --- | --- |
| `franka-sorting-parts-v0` | 16 | Eval benchmark. Shared tray library, four tables, copies only the assets it references |

Containers are `grey_bin`, `purple_crate`, `container_f24`, `bin_b04`, and `bin_a06`. Tasks use the oak, maple, bamboo, and black tables.

Trays are 1x1, 1x2, 2x2, and 2x3. Each cell is 150 mm by 200 mm and 6-8 cm deep, in eight finishes (32 trays). A task sorts 3-6 part types into a random combination of those trays. Each bin and tray is seated on a quarter turn, then turned by about 15° when that heading still fits on the table. The bin holds 20-40 parts, dropped in a pile.

Parts are rebuilt at 3x, except nine wide variants at 2.5x. The published task keeps the parts whose centers land inside the walls. Deep bins are flattened to 0.165 m so the wrist link clears the rim. Their colliders are invisible boxes (a floor slab and four walls), because the source meshes pop parts out.

## Install

Isaac Sim has to run with its bundled Python, so every command goes through `./isaaclab.sh` from the Isaac Lab root. On Windows the same commands start with `isaaclab.bat`. Inside that tree this package is `source/task-franka-sorting/source/franka_sorting`.

```bash
./isaaclab.sh -p -m pip install -e "source/task-franka-sorting/source/franka_sorting[record]"
```

The `record` extra adds `pyarrow` and `av`. The `layout` extra adds `trimesh` and `shapely` for the layout checks. TensorBoard logs are under this project, at `logs/rsl_rl/franka_sorting_ll_ee_tracking`, not the Isaac Lab `logs` directory.

## Train and play

Isaac Lab 3 does not have a `--headless` flag. `--viz none` runs without a window. `--num_envs` for a high-level task defaults to one env per task, and a larger value has to be a multiple of that count. One task is `env.task_id=<task_id>`.

```bash
./isaaclab.sh train --task Nepher-FrankaSorting-LL-v0 --num_envs 4096 --max_iterations 1500 --viz none
./isaaclab.sh play --task Nepher-FrankaSorting-LL-Play-v0 --num_envs 32 --viz none
./isaaclab.sh play --task Nepher-FrankaSorting-HL-EnvhubPlay-v0 --num_envs 16 --viz none
./isaaclab.sh play --task Nepher-FrankaSorting-HL-Play-v0 --num_envs 16 --viz none
```

`Nepher-FrankaSorting-LL-Play-v0` and `Nepher-FrankaSorting-HL-Play-v0` load the latest `model_*.pt` under `logs/rsl_rl/franka_sorting_ll_ee_tracking` unless you pass `--checkpoint`. `Nepher-FrankaSorting-HL-EnvhubPlay-v0` loads `best_policy/best_policy.pt`.

### Highlight videos

`env.highlight=on` on `Nepher-FrankaSorting-HL-Play-v0` or `Nepher-FrankaSorting-HL-EnvhubPlay-v0` writes `best_policy/videos/highlight/<camera>.mp4` for `cam_exterior_left`, `cam_exterior_right`, `cam_wrist`, and `cam_front`, then exits. Each file is 1280x720, a 4x4 grid of 320x180 cells, and lasts one 40 s episode at the 15 Hz control rate. The grid shows up to 16 tasks of the stage in benchmark order. A stage with fewer leaves the remaining cells black. `env.highlight_tasks=0-15` films that inclusive range, and `0,5,12` films those indices, in that order instead of the sample. Only those envs are rendered, at the cell size.

```bash
./isaaclab.sh play --task Nepher-FrankaSorting-HL-EnvhubPlay-v0 --num_envs 16 --viz none \
    env.highlight=on "env.highlight_tasks=0-15"
```

## LeRobot synthetic data

`scripts/record_lerobot.py` is a sharded, resumable orchestrator. The parent process starts no Kit. It runs `scripts/record_shard.py` once per shard, and a shard that stops short or crashes is retried (`--retries`). Its state is in `meta/sdg_state.json`, and `sdg_report.json` summarises the run.

- `--source benchmark` records the tasks of `--env_id` (or `--task`). `--source procedural` samples fresh layouts per shard (`--shards`).
- `--episodes` is the number of successful episodes kept per task. An existing dataset is extended, and an interrupted run resumes (`--restart` discards the saved plan, not the saved episodes).
- Randomization: `--randomize poses` (default) draws a new layout every episode, and `--randomize none` replays the published layout. Lighting is randomized unless `--no_lighting`, and `--camera_jitter` and `--arm_noise` set the camera extrinsics jitter and the arm start noise.
- Quality gates drop an episode that is too short or too long, has NaNs, leaves the joint limits, jumps more than 0.30 rad per step, leaves the gripper range, has blank frames, starts invalid, loses a part, or does not succeed. `--keep_failed` saves failed episodes with a label.
- Output is a LeRobot v2.1 dataset. Files are written atomically (`info.json` last), and `annotation.phase` holds the planner stage of each frame.

```bash
./isaaclab.sh -p source/task-franka-sorting/scripts/record_lerobot.py --env_id franka-sorting-parts-v0 --list_tasks
./isaaclab.sh -p source/task-franka-sorting/scripts/record_lerobot.py --env_id franka-sorting-parts-v0 \
    --episodes 8 --output datasets/sorting
./isaaclab.sh -p source/task-franka-sorting/scripts/record_lerobot.py --dry_run --source procedural --shards 4
```

Without `--cameras` only the state is recorded. With cameras use `--num_envs 8` or fewer per shard.

The separate `sdg-gen` project can record the same task. Its `configs/task-franka-sorting.yaml` captures 8-D joint state, tool pose, the four cameras, and the absolute joint-target action.

## FAQ

**What robot and simulator does this target?**
A Franka arm with a Robotiq 2F-85 gripper, in NVIDIA Isaac Sim 6.1.0 with Isaac Lab 3.0. Physics steps at 240 Hz. The policy command lands every 16 steps, at 15 Hz.

**What does the tracker observe and output?**
The observation is 43 values: 7 arm positions, 7 arm velocities, gripper, 7-D tool pose, 6-D pose error, 7-D pose target, grip target, and the last 7-D action. The action is a 7-D tool-pose delta plus the gripper. `IkJointCommandAction` integrates it with damped least-squares IK into absolute joint targets.

**How is an episode a success?**
At least 4, 5, or 6 parts have to sit in the cells named by the instruction, and that predicate has to stay true for 0.5 s, within the 40 s episode. Parts left in the container are allowed. A lost object ends the episode as a failure.

**Which Gym ID should I start with?**
Train `Nepher-FrankaSorting-LL-v0`, then play the planner with `Nepher-FrankaSorting-HL-Play-v0`. For eval-nav, use `Nepher-FrankaSorting-HL-EnvhubPlay-v0`, which loads `best_policy/best_policy.pt`.

**Which EnvHub bundle loads by default?**
`franka-sorting-parts-v0` (16 tasks). `scripts/link_cache.py` links the bundle into the nepher cache.

**Where does the tracker checkpoint load from?**
`Nepher-FrankaSorting-LL-Play-v0` and `Nepher-FrankaSorting-HL-Play-v0` load the latest `model_*.pt` under `logs/rsl_rl/franka_sorting_ll_ee_tracking` unless you pass `--checkpoint`. `Nepher-FrankaSorting-HL-EnvhubPlay-v0` loads `best_policy/best_policy.pt`.

**How do I record a LeRobot dataset?**
Run `scripts/record_lerobot.py`. `--source benchmark` records a published bundle. `--source procedural` samples new layouts. Successful episodes are kept, and a stopped run resumes.

## License

Copyright (c) 2026, [Nepher Robotics](https://github.com/nepher-ai). All rights reserved.
Distributed under the [BSD 3-Clause License](LICENSE).

**Keywords:** Franka, Franka Panda, Robotiq 2F-85, bin picking, parts sorting, industrial parts, reinforcement learning, PPO, hierarchical policy, inverse kinematics, NVIDIA Isaac Lab, Isaac Sim, rsl_rl, LeRobot, synthetic data, imitation learning, EnvHub, franka-sorting-parts-v0, Nepher.
