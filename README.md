# sable_isaacgym

Sable robot authoring, Isaac Gym control, training, and evaluation.

Project instructions are in [AGENTS.md](AGENTS.md). The current migration plan is
[16-command / 6-frame-history policy](planning/sable_cmd16_history6_migration.md).
Its requested acceptance includes translation velocity tracking, yaw velocity
tracking, and commanded gait-frequency transitions.

The pre-migration source is `deployment/legged_gym_stride80_all_20261027`.
The designated teacher is the final saved result of
`runs/stand5_20261027_v44_stride80_all`, after completion and archive verification.
Runs, checkpoints, videos, and pending-deletion historical files remain on the
server and are excluded from Git. The NVIDIA Isaac Gym SDK is not redistributed.

Use the existing server environment with `source deployment/runtime.sh`.
The two open-source dependencies are pinned Git submodules; fresh checkouts use
`git submodule update --init --recursive`.
Training runs only on the remote GPU server; retain each run's frozen runtime,
physical assets, configuration, deadline, evaluation, and archive. See
[deployment notes](deployment/README.md) for environment provenance.
