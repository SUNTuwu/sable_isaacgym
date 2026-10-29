# sable_isaacgym

Sable robot authoring, Isaac Gym control, training, and evaluation.

Project instructions are in [AGENTS.md](AGENTS.md). For the latest saved training
and moving to another server, start with the
[2026-10-29 migration guide](deployment/migration_20261029/README.md).

The active basic-locomotion baseline is
`baselines/user_accepted_20261029_120250`: eight reference-guided teacher policies,
iteration 2084, 94 observations and 15 outputs. All eight survived the final
four-environment-per-policy, 24-second evaluation; strict tracking acceptance
remains false. The separate yaw stance policy at iteration 1274 also needs work.
User acceptance of basic locomotion excludes this yaw task.

The [16-command / 6-frame-history plan](planning/sable_cmd16_history6_migration.md)
remains a separate student-policy objective. The current teacher bank does not
complete its full-target, reference-free acceptance requirements. The v44-9470
teacher in that plan is historical, not the latest basic-locomotion baseline.

Code, current frozen runtime, physical assets, reference arrays, hashes and small
evaluation reports are in Git. Checkpoints (including Adam state), bulk videos,
caches and pending-deletion historical files stay outside Git. A verified private
transfer package is prepared separately; cloning Git alone cannot resume weights.
The NVIDIA Isaac Gym SDK is installed or privately transferred separately.

Use the existing server environment with `source deployment/runtime.sh`.
The two open-source dependencies are pinned Git submodules; fresh checkouts use
`git submodule update --init --recursive`.
Training runs only on the remote GPU server; retain each run's frozen runtime,
physical assets, configuration, deadline, evaluation, and archive. See
[deployment notes](deployment/README.md) for environment provenance.
