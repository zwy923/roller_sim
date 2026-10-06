"""Every check of the project, one after the other.

    python checks/run_all.py            the quick ones: one to three minutes
    python checks/run_all.py --full     also the physics acceptance runs of station_checks.py: 10 to 15 minutes more

Each script still runs on its own (python checks/NAME.py); this only saves remembering the list. timestep_checks.py
is not here: it is a sensitivity screen with its own arguments and output folder, not a pass/fail check.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATION_QUICK = ['Geometry', 'Devices', 'CompiledModel', 'Weighing', 'Tracker', 'Control', 'Rules', 'Baseline']
QUICK = [['checks/architecture_checks.py'],
         ['checks/geometry_regression_checks.py'],
         ['checks/material_checks.py'],
         ['checks/physics_regression_checks.py'],
         ['checks/feeder_regression_checks.py'],
         ['checks/flip_separator_checks.py'],
         ['experiments/analyze_checks.py'],
         ['checks/station_checks.py'] + STATION_QUICK,
         ['checks/plough_checks.py']]
FULL = [['checks/station_checks.py', 'Acceptance']]


def main():
    failed = []
    for cmd in QUICK + (FULL if '--full' in sys.argv[1:] else []):
        t0 = time.time()
        done = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
        ok = done.returncode == 0
        print('%s %-60s %5.0f s' % ('PASS' if ok else 'FAIL', ' '.join(cmd), time.time() - t0), flush=True)
        if not ok:
            failed.append(cmd)
            print((done.stdout + done.stderr)[-3000:])
    print('all checks passed' if not failed else '%d failed: %s' % (len(failed), [c[0] for c in failed]))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
