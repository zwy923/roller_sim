"""Tunables: changing a module constant of the package for one run, on the record.

The margins, thresholds and sensor-model numbers of the line are constants next to the code they belong to, each
with the reason for its value (control/station.py: APPROACH, ISO_MARGIN ...; sensing/vision.py: LATENCY_S ...).
They are not command-line parameters: there are about 150 of them and a run normally changes none. When an
experiment does change one, it says so here instead of patching the module behind the model's back:

    python plough.py --set control.station.APPROACH=.65 --set sensing.weigher.WEIGH_MAX_S=5
    cfg = default_config(set=['machine.station.STOP_BACK=.37'])

simulate.run() applies cfg['set'] for the run and restores the values afterwards; the names and values are in
result.json (config.set). A name is MODULE.NAME below the package; the constant must exist and be a number, None or
a tuple of numbers, and the value must be of the same kind.

For an override to reach every use, code reads another module's tunable as module.NAME at the point of use; it
does not import the name or derive a constant from it at import time (checks/architecture_checks.py).
"""
import importlib
from contextlib import contextmanager

PACKAGE = __name__.rsplit('.', 1)[0]


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _tunable(v):
    return v is None or _number(v) or (isinstance(v, tuple) and len(v) > 0 and all(_number(x) for x in v))


def resolve(name):
    """'control.station.APPROACH' -> (module, 'APPROACH'); refuses a name that is not a tunable."""
    mod, _, attr = name.rpartition('.')
    if not mod or not attr.isupper():
        raise ValueError('--set %s: write MODULE.NAME, e.g. control.station.APPROACH' % name)
    try:
        module = importlib.import_module('%s.%s' % (PACKAGE, mod))
    except ImportError:
        raise ValueError('--set %s: no module %s.%s' % (name, PACKAGE, mod))
    if not hasattr(module, attr) or not _tunable(getattr(module, attr)):
        raise ValueError('--set %s: %s.%s has no tunable %s' % (name, PACKAGE, mod, attr))
    return module, attr


def parse(text):
    """'control.station.APPROACH=.65' -> ('control.station.APPROACH', 0.65). Values: a number, 'none', or numbers
    separated by commas (a tuple)."""
    name, eq, raw = text.partition('=')
    if not eq or not raw.strip():
        raise ValueError('--set %s: write MODULE.NAME=VALUE' % text)

    def number(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
    raw = raw.strip()
    try:
        value = (None if raw.lower() == 'none' else tuple(number(s) for s in raw.split(',')) if ',' in raw
                 else number(raw))
    except ValueError:
        raise ValueError('--set %s: %r is not a number, a list of numbers or none' % (text, raw))
    return name.strip(), value


@contextmanager
def applied(settings):
    """Apply the settings ('MODULE.NAME=VALUE' strings, or (name, value) pairs) for the block, then restore."""
    pairs = [parse(s) if isinstance(s, str) else tuple(s) for s in settings or ()]
    old = []
    try:
        for name, value in pairs:
            module, attr = resolve(name)
            was = getattr(module, attr)
            if value is not None and was is not None and isinstance(value, tuple) != isinstance(was, tuple):
                raise ValueError('--set %s: the value must be %s' % (name, 'a tuple' if isinstance(was, tuple)
                                                                    else 'a number'))
            old.append((module, attr, was))
            setattr(module, attr, value)
        yield dict(pairs)
    finally:
        for module, attr, was in reversed(old):
            setattr(module, attr, was)


def listing():
    """Every tunable of the package: {'control.station.APPROACH': 0.45, ...} (python -m singulator.tuning)."""
    import pkgutil
    out = {}
    package = importlib.import_module(PACKAGE)
    for info in pkgutil.walk_packages(package.__path__, PACKAGE + '.'):
        module = importlib.import_module(info.name)
        for attr, value in vars(module).items():
            if attr.isupper() and not attr.startswith('_') and _tunable(value) \
                    and getattr(module, '__name__', '') == info.name:
                out['%s.%s' % (info.name[len(PACKAGE) + 1:], attr)] = value
    return out


if __name__ == '__main__':
    for key, value in sorted(listing().items()):
        print('%-44s %r' % (key, value))
