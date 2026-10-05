"""The controllers of the line. Each reads sensor signals (sensing/) and sets targets for drives and actuators.

    feeder      the feed belt: lets the batch go lump by lump
    face        the plough face's retract action (the phase machine on its servo)
    station     buffer belt, measuring belt and separator plate: items, void rules, routes, alarms
    supervisor  what ties the three together: drive targets, the feed interlock, stall -> retract, the stops

Nothing here imports MuJoCo or reads the simulated plant (physics/).
"""
