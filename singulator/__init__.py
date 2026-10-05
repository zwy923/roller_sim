"""Small-batch coal/gangue line: assembly and uncalibrated rigid-contact screening in MuJoCo.

    feed belt -> main belt -> plough face -> lane -> buffer belt -> measuring belt -> flip separator

Modules, in the order a run uses them:
  config      command-line configuration (every default is the line)
  lumps       shape families, material draws, feed layer, placed-lump outline geometry
  machine     dimensions and derived geometry (feed belt, plough face, lane, belts, station, motion windows)
  assembly    the MuJoCo model, and the moving-part clearance sweep
  devices     beams, cameras and the volume scanner as hardware
  drives      virtual motors, the conveyors, load bookkeeping
  face        retract action of the hinged plough face
  feeder      the step-down feed belt releasing the batch lump by lump
  station     buffer belt, measuring belt (weight + volume) and the separator's control
  separator   the flip separator's geometry, shared with the standalone model in designs/flip_separator/
  perception  what the controllers read: camera objects, debounced beams, the scanner's verdict
  line        sensors, held items taken off, injected faults, acceptance scenarios
  trial       bench layouts of the batch and the feed head transfer record
  tracking    per-lump observation and the stall rule
  audit       shared per-lump measurement audit and result/CLI summary (no control feedback)
  video       follow camera + plan view with a text overlay
  simulate    run(): one batch through the machine, and its outputs
"""
