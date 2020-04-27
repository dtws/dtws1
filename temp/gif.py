import subprocess as sp
import sys


def make_gif(files, output, interval=10):
    cmd = [
        "convert",
        "-layers",
        "optimize",
        "-loop",
        "0",
        "-delay",
        str(interval)
    ] + list(files) + [
        output
    ]
    r = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    o = r.stdout.decode("utf-8")
    e = r.stderr.decode("utf-8")
    print(o)
    if len(e) > 0:
        msg = f"{e}\n{' '.join(cmd)}\n"
        sys.stderr.write(msg)
