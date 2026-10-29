"""Authorized recovery always reloads the selected per-teacher baselines."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name('launch_selected.py')),run_name='__main__')
