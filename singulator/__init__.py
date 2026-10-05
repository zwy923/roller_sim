"""Small-batch coal/gangue line: assembly and uncalibrated rigid-contact screening in MuJoCo.

    feed belt -> main belt -> plough face -> lane -> buffer belt -> measuring belt -> flip separator

The package in layers; a module imports only from the layers above its own:

  config, tuning    every parameter of the line, defined once; changing a tunable constant for one run (--set)
  lumps, geom2d, series, audit
                    what a batch is made of; plane geometry; load summaries; the per-lump measurement audit
  machine/          the hardware: where every section stands (pure geometry) and its MuJoCo model
  physics/          the simulated plant: drives, actuators, the lumps' true state, numerical screening
  sensing/          what the controllers can know: cameras, beams, volume scanner, load cells as models
  control/          the controllers: feed belt, plough face, station
  verify/           verification against the true state (feeds nothing back)
  sim/              a batch put together and advanced: layouts, scenarios, the line, record, results, video
  simulate          run(cfg): the entry point (plough.py is its command line)

docs/ARCHITECTURE.md describes the layers and how to replace a device model.
"""
