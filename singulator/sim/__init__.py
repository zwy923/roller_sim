"""Running a batch: everything put together and stepped, the scene's set-up, the outputs.

    layouts     how the batch lies on the feed belt at t = 0 (--layout)
    scenarios   acceptance scenarios (--scenario) and taking a held item off the line
    line        Line: everything built for one batch, and what happens in a physics step and in a 10 ms sample
    record      what a run records of the true state (per lump, at the measuring plane, the trajectory)
    results     the result (result.json) put together; the output files
    video       the follow camera and the plan view with their text overlay

singulator/simulate.py holds run(cfg), the loop over Line.advance().
"""
