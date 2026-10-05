"""Verification against the true state: instrumentation a real line would not have.

    feeder      which true lumps went over the feed belt's head edge in which release
    station     the station's items against the truth: false valid, false void, landed unmeasured
    transfer    the feed head's transfer record (tip, hang, early contact, double releases, stop distances)

These read the truth (physics/) next to what the controllers saw and decided, and write what result.json reports
about it. They feed nothing back into the run.
"""
