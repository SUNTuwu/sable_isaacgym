"""Persistent operator pause, independent of any frozen run's completion state."""
import json
from pathlib import Path

HOLD = Path('/root/gpufree-data/projects/sable-isaacgym/results/cmd16_hist6/TRAINING_PAUSED.json')


def require_training_enabled(path=HOLD):
    if not path.exists():
        return
    # Corrupt pause metadata must not silently authorize a launch.
    state = json.loads(path.read_text(encoding='utf8'))
    if state.get('paused') is not False:
        raise RuntimeError('Training paused by user; a new explicit resume instruction is required. ' + str(path))
