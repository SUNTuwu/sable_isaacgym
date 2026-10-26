"""Apply the shoulder-roll lowering through the shared mount revision script."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name('lower_hip_roll.py')),
              init_globals={'ROLL_MOUNT':'shoulder'}, run_name='__main__')
