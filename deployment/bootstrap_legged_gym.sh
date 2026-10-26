#!/usr/bin/env bash
set -euo pipefail
source /root/gpufree-data/projects/sable-isaacgym/deployment/runtime.sh
cd "$SABLE_GYM_ROOT"
mkdir -p vendor logs runs assets deployment/legged_gym
test -d vendor/legged_gym/.git || git clone --depth 1 https://github.com/leggedrobotics/legged_gym.git vendor/legged_gym
test -d vendor/rsl_rl/.git || git clone --depth 1 --branch v1.0.2 https://github.com/leggedrobotics/rsl_rl.git vendor/rsl_rl
python -m pip install --no-deps -e vendor/rsl_rl -e vendor/legged_gym
python -m pip check
git -C vendor/legged_gym rev-parse HEAD
git -C vendor/rsl_rl rev-parse HEAD
