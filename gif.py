import subprocess as sp


def make_gif(files, output):
    cmd = [
        "convert",
        "-layers",
        "optimize",
        "-loop",
        "0",
        "-delay",
        "40"
    ] + list(files) + [
        output
    ]
    r = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    return r.stdout.decode("utf-8"), r.stderr.decode("utf-8")
