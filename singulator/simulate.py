"""Run one batch through the line.

    from singulator.config import default_config
    from singulator.simulate import run
    result = run(default_config(seed=392, video=False))

run(cfg, save=True) is the package's entry point (plough.py is its command line); the run itself is
singulator/sim/run.py.
"""
from .sim.run import run

__all__ = ['run']
