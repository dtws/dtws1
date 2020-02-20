import subprocess


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
    subprocess.call(cmd)
