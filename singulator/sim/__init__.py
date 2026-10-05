"""Running a batch: everything put together and stepped, the scene's set-up, the outputs.

    layouts     how the batch lies on the feed belt at t = 0 (--layout)
    scenarios   acceptance scenarios (--scenario) and taking a held item off the line
    run         run(cfg): one batch through the line; result.json, model.xml, trajectory.npz
    video       the follow camera and the plan view with their text overlay
"""
