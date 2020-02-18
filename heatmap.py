import matplotlib.pyplot as plt
from h3 import h3
import pandas as pd
import numpy as np
import functools as ft
import tilemapbase as tmb
from colormath.color_objects import sRGBColor, HSVColor
from colormath.color_conversions import convert_color


def _flatten(lss):
    return ft.reduce(lambda x, y: x+y, lss)


def _swap(ls):
    return [
        (p[1], p[0])
        for p in ls
    ]


def extent(df, id_col):
    ids = df[id_col].unique()
    latlngs = _flatten(map(h3.h3_to_geo_boundary, ids))
    lats = map(lambda x: x[0], latlngs)
    lngs = map(lambda x: x[1], latlngs)
    extent = tmb.Extent.from_lonlat(
        min(lngs), max(lngs),
        min(lats), max(lats)
    )
    extent.to_aspect(1)
    return extent


def color_selector(df, val_col, n, hue=0.9):
    values = df[val_col]
    cols = [
        convert_color(
            HSVColor(hue, i/n, 1),
            sRGBColor
        ).get_rgb_hex()
        for i in range(n)
    ]
    m, M = min(values), max(values)
    width = M - m
    tick = width / n

    def selector(v):
        q = min(v - m // tick, n - 1)
        return cols[q]

    return selector


def draw(df, id_col, val_col, extent, color_selector, figsize=(8, 8), dpi=100, width=600):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    t = tmb.tiles.build_OSM()
    plotter = tmb.Plotter(extent, t, width=width)
    plotter.plot(ax, t)

    n = length(draw)
    for i in range(n):
        id = df[id_col][i]
        val = df[val_col][i]
        color = color_selector(val)
        vts = _swap(h3.h3_to_geo_boundary(id))
        xys = map(lambda x: tmb.project(*x), vtx)
        poly = plt.Polygon(xys, fc=color)
        ax.add_patch(poly)
