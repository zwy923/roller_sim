"""The package keeps its layering (no physics runtime needed; a second).

Run: python checks/architecture_checks.py
1. Layers: a module imports only from its own layer or the ones above it in singulator/__init__.py's list
   (base < machine < physics < sensing < control < verify < sim < simulate).
2. Controllers stay on sensor signals: nothing in control/ imports MuJoCo or the simulated plant (physics/), and no
   controller source mentions the truth keys ('_truth', 'truth_').
3. Tunables stay settable: no module imports another module's tunable constant by name (from x import NAME copies
   the value, and --set would no longer reach that use); it reads module.NAME where it uses it.
4. machine/ is importable without MuJoCo (geometry is pure): MuJoCo is imported inside the functions that compile.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'singulator'
sys.path.insert(0, str(ROOT))
LAYERS = ['base', 'machine', 'physics', 'sensing', 'control', 'verify', 'sim', 'simulate']
# constants that are part of an interface, not tunables: fine to import by name
NOT_TUNABLES = {'SAMPLE_S', 'ROOT', 'CATS', 'IN_LINE', 'END_STATES', 'LUMP_BODY', 'LUMP_GEOM', 'LUMP_JOINT', 'SKIRT',
                'FAMILIES', 'DENSITY_RANGE', 'PARAMS', 'NAMES', 'LAYOUTS', 'SCENARIOS'}


def layer(module):
    """'singulator.control.station' -> 'control'; top-level modules are 'base', except simulate."""
    parts = module.split('.')
    if len(parts) == 2:
        return 'simulate' if parts[1] == 'simulate' else 'base'
    return parts[1]


def imports(path):
    """(imported module, names, at module level?) for every import of the package or of mujoco in a file."""
    module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
    package = module.rsplit('.', 1)[0] if path.name != '__init__.py' else module.rsplit('.', 1)[0]
    tree = ast.parse(path.read_text(encoding='utf-8'))
    top = set(map(id, tree.body))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, [], id(node) in top) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                up = package.split('.')[:len(package.split('.')) - node.level + 1]
                base = '.'.join(up + ([node.module] if node.module else []))
            out.append((base, [a.name for a in node.names], id(node) in top))
    return module, out


def main():
    bad = []
    for path in sorted(PKG.rglob('*.py')):
        module, found = imports(path)
        mine = layer(module) if path.name != '__init__.py' or path.parent != PKG else 'base'
        for target, names, at_top in found:
            if target == 'mujoco' or target.startswith('mujoco.'):
                if mine == 'control':
                    bad.append('%s imports mujoco: a controller reads sensors, not the simulator' % module)
                if mine == 'machine' and at_top:
                    bad.append('%s imports mujoco at module level: geometry must stay importable without it' % module)
                continue
            if not target.startswith('singulator'):
                continue
            # "from ..machine import station" names a module of the layer, "from ..config import X" a name in it
            targets = [target] if (PKG / Path(*target.split('.')[1:])).with_suffix('.py').exists() else \
                ['%s.%s' % (target, n) for n in names] if target != 'singulator' else ['singulator.%s' % n for n in names]
            for t in targets:
                theirs = layer(t) if t.count('.') else 'base'
                if LAYERS.index(theirs) > LAYERS.index(mine):
                    bad.append('%s (%s) imports %s (%s): against the layering' % (module, mine, t, theirs))
                if mine == 'control' and theirs == 'physics':
                    bad.append('%s imports %s: a controller must not read the simulated plant' % (module, t))
            if len(targets) == 1 and targets[0] == target:
                copied = [n for n in names if n.isupper() and n not in NOT_TUNABLES]
                if copied:
                    bad.append('%s copies %s from %s: read it as module.NAME so that --set reaches it'
                               % (module, ', '.join(copied), target))
    for path in sorted((PKG / 'control').glob('*.py')):
        text = path.read_text(encoding='utf-8')
        for token in ("'_truth'", '"_truth"', 'truth_'):
            if token in text:
                bad.append('%s mentions %s: the truth belongs to verify/' % (path.relative_to(ROOT), token))
    for line in bad:
        print('FAIL ' + line)
    if bad:
        sys.exit(1)
    from singulator import tuning
    print('PASS layers, controllers on sensor signals, tunables read where they are used (%d modules, %d tunables)'
          % (len(list(PKG.rglob('*.py'))), len(tuning.listing())))


if __name__ == '__main__':
    main()
