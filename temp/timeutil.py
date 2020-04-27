import time
import sys

_time = None


def start():
    global _time
    _time = time.time()


def end(dest=sys.stderr):
    val = time.time() - _time
    if dest is None:
        return val
    else:
        dest.write(f"elapsed: {val}")
