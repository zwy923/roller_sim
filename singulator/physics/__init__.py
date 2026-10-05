"""The simulated plant: what moves in the MuJoCo model, and the true state of the lumps.

    drives      every conveyor behind its force-limited motor; the load bookkeeping
    actuators   the plough face's hinge servo and the separator plate's cylinder
    lumps       each lump's true pose and outline, sampled every 10 ms: the truth the sensors are formed from and
                the record is written from
    numerics    numerical screening of the run (penetration, solver iterations, warnings)

The controllers (control/) never read anything here: they get sensor signals (sensing/) and set drive targets.
"""
