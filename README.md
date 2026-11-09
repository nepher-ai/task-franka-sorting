# Franka parts sorting

A Franka arm with a Robotiq 2F-85 gripper picks randomly placed industrial parts out of a container and sorts them
into the cells of a parts tray. Every tray cell takes one part type, and the mapping is random per task and stated in
the instruction text. The project is built for tournament evaluation and synthetic data generation (SDG) for LeRobot.

```
planner (SortingPlanner) -> learned tool-pose tracker (rsl_rl PPO) -> damped-least-squares IK -> 8-D joint targets
```

Physics runs at 180 Hz and control at 15 Hz.

Agents: start at [`llms.txt`](llms.txt) (index) or [`llms-full.txt`](llms-full.txt) (single-file context).

- `source/franka_sorting/` is the task package: layout sampler and validator (`layout/`), benchmark loading
  (`benchmark/`), scene (`scene/`), planner, tracker and gym registrations (`tasks/`), and the SDG code (`data/`).
- The package ships only the robot USD. Containers, tables, parts, trays and `*.hdr` backgrounds live in the EnvHub
  bundles under `source/envhub/environments/`.
- `best_policy/best_policy.pt` is the tracker checkpoint (rsl_rl, weights only), fine-tuned from the tabletop tracker.
- `scripts/` holds the synthetic-data tools. The EnvHub bundles are written by the separate
  `build_sorting_benchmark` project.

## Install

From the Isaac Lab root:

```bash
./isaaclab.sh -p -m pip install -e "source/task-franka-sorting/source/franka_sorting[record]"
```

The `record` extra adds `pyarrow` and `av`. The `layout` extra adds `trimesh` and `shapely` for the layout checks.
`scripts/link_cache.py` links the bundles into the nepher cache, so no download is needed.

## Benchmarks

| Bundle                    | Tasks | Content                                                                          |
| ------------------------- | ----- | -------------------------------------------------------------------------------- |
| `franka-sorting-parts-v0` | 16    | Eval benchmark. Shared tray library, four tables, copies only the assets it references. |
| `franka-sorting-parts-v1` | 16    | Eval benchmark. Same tray library, a different seed.                                    |
| `franka-sorting-parts-v2` | 160   | Rich library. Blocks of 16 tasks share one background. Task `i` never moves.            |

- Containers are `grey_bin`, `purple_crate`, `container_f24`, `bin_b04` and `bin_a06`. Tasks use the oak, maple,
  bamboo, and black tables.
- Trays are 1x1, 1x2, 2x2 and 2x3. Each cell is 150 mm by 200 mm and 6-8 cm deep, in eight finishes (32 trays).
  A task sorts 3-6 part types into a random combination of those trays. The bin holds 20-40 parts, dropped in a
  pile. Success is sorting at least 4, 5, or 6 of them within 30 s. Parts left in the bin do not fail the task.
- Parts are rebuilt at 3x, except nine wide variants at 2.5x. The bin holds a settled pile: the published task
  keeps the parts whose centers land inside the walls.
- Deep bins are flattened to 0.165 m so the wrist link clears the rim. Their colliders are invisible boxes (floor slab
  and four walls), because the source meshes pop parts out.

## Environments

`scripts/list_envs.py` lists them.

- `Nepher-FrankaSorting-HL-Play-v0`: every task of a bundle in one stage.
- `Nepher-FrankaSorting-HL-EnvhubPlay-v0`: planner plus the checkpoint in `best_policy/`, as the evaluator runs it.
- `Nepher-FrankaSorting-LL-v0` and `-LL-Play-v0`: train and play the tracker.

Train and play go through the Isaac Lab CLI. The Gym registration names rsl_rl, so `--rl_library` is not passed.

```bash
./isaaclab.sh train --task Nepher-FrankaSorting-LL-v0 --num_envs 4096 --max_iterations 1500 --viz none
./isaaclab.sh play --task Nepher-FrankaSorting-HL-EnvhubPlay-v0 --num_envs 16 --viz none
./isaaclab.sh play --task Nepher-FrankaSorting-HL-EnvhubPlay-v0 --num_envs 16 --viz none \
    env.highlight=on "env.highlight_tasks=0-15"
```

`env.highlight=on` writes one 30 s clip per camera to `best_policy/videos/highlight/` and then exits. On Windows the same commands start with `isaaclab.bat`.

## Synthetic data generation

`scripts/record_lerobot.py` is a sharded, resumable orchestrator. The parent process starts no Kit. It runs
`scripts/record_shard.py` once per shard, and a shard that stops short or crashes is retried (`--retries`). Its state
is in `meta/sdg_state.json`, and `sdg_report.json` summarises the run.

- `--source benchmark` records the tasks of `--env_id` (or `--task`). `--source procedural` samples fresh layouts per
  shard (`--shards`).
- `--episodes` is the number of successful episodes kept per task. An existing dataset is extended, and an
  interrupted run resumes (`--restart` discards the saved plan, not the saved episodes).
- Randomization: `--randomize poses` (default) draws a new layout every episode, and `--randomize none` replays the
  published layout. Lighting is randomized unless `--no_lighting`, and `--camera_jitter` and `--arm_noise` set the
  camera extrinsics jitter and the arm start noise.
- Quality gates drop an episode that is too short or too long, has NaNs, leaves the joint limits, jumps more than
  0.30 rad per step, leaves the gripper range, has blank frames, starts invalid, loses a part or does not succeed.
  `--keep_failed` saves failed episodes with a label.
- Output is a LeRobot v2.1 dataset. Files are written atomically (`info.json` last), and `annotation.phase` holds the
  planner stage of each frame.

```bash
./isaaclab.sh -p source/task-franka-sorting/scripts/record_lerobot.py --env_id franka-sorting-parts-v0 --list_tasks
./isaaclab.sh -p source/task-franka-sorting/scripts/record_lerobot.py --env_id franka-sorting-parts-v2 \
    --episodes 8 --output datasets/sorting
./isaaclab.sh -p source/task-franka-sorting/scripts/record_lerobot.py --dry_run --source procedural --shards 4
```

Without `--cameras` only the state is recorded. With cameras use `--num_envs 8` or fewer per shard.

## Tests

```bash
_isaac_sim\python.bat -m pytest source/task-franka-sorting/tests
```

The tests need no simulator.
