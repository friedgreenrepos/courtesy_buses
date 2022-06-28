# pub node
import sys


PUB = 0

EPSILON = 1e-3

verbose = False


def node_to_string(v):
    return 'PUB' if v == 0 else f"({v})"


def eprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs, file=sys.stderr)


def vprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def abort(msg):
    print(f"ERROR: {msg}")
    exit(-1)

