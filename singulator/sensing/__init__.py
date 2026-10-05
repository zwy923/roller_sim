"""What the controllers can know: the sensors of the line as models.

    vision      the overhead cameras: tracked objects (outline, centroid, speed, confidence), 50 ms late
    beams       through-beams: debounce, faults, the three diagnoses
    volume      the volume scanner over the measuring belt: the volume and whether the scan is usable
    weigher     the load cells under the measuring belt
    suite       the line's set of sensors, fed with the scene every 10 ms; injected faults

The true state (physics/) enters here and only here; what comes out is what control/ reads. Every object and
verdict keeps what it was formed from under keys starting with '_', for the verification (verify/): no controller
reads those.
"""
