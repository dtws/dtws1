import itertools as it
import matplotlib.pyplot as plt
from h3 import h3
import pandas as pd
import numpy as np
import functools as ft
import tilemapbase as tmb
from colormath.color_objects import sRGBColor, HSVColor
from colormath.color_conversions import convert_color
import folium
from collections import namedtuple


def _flatten(lss):
    return ft.reduce(lambda x, y: x+y, lss)


def _swap(ls):
    return [
        (p[1], p[0])
        for p in ls
    ]


def _extent(ids):
    RetExtent = namedtuple(
        "RetExtent", "extent min_lng max_lng min_lat max_lat")

    ids_u = ids.unique()
    latlngs = _flatten(map(h3.h3_to_geo_boundary, ids_u))
    lats = [x[0] for x in latlngs]
    lngs = [x[1] for x in latlngs]
    extent = tmb.Extent.from_lonlat(
        min(lngs), max(lngs),
        min(lats), max(lats)
    )
    extent.to_aspect(1)
    return RetExtent(extent, min(lngs), max(lngs), min(lats), max(lats))


def extent(ids):
    return _extent(ids).extent


def n_extent(ids, n):
    extent_ = _extent(ids)
    lng_width = (extent_.max_lng - extent_.min_lng) / n
    lat_width = (extent_.max_lat - extent_.min_lat) / n
    lng_ticks = [
        extent_.min_lng + i * lng_width
        for i in range(n+1)
    ]
    lat_ticks = [
        extent_.min_lat + i * lat_width
        for i in range(n+1)
    ]
    MinMax = namedtuple("MinMax", "min max")
    lng_bounds = map(lambda x: MinMax(*x), zip(lng_ticks[:-1], lng_ticks[1:]))
    lat_bounds = map(lambda x: MinMax(*x), zip(lat_ticks[:-1], lat_ticks[1:]))

    Bound = namedtuple("Bound", "longitude latitude")
    bounds = list(map(lambda x: Bound(*x), it.product(lng_bounds, lat_bounds)))
    extents = [
        tmb.Extent.from_lonlat(
            bound.longitude.min, bound.longitude.max,
            bound.latitude.min, bound.latitude.max
        ) for bound in bounds
    ]

    def search_position(id):
        lat, lng = h3.h3_to_geo(id)
        xpos = int((lng - extent_.min_lng) // lng_width)
        ypos = int((lat - extent_.min_lat) // lat_width)
        # import pdb
        # pdb.set_trace()
        return n*xpos + ypos
    search_position_v = np.vectorize(search_position)
    positions = search_position_v(ids)

    Return = namedtuple("Return", "extents positions")
    return Return(extents, positions)


def color_selector(values, n, hue=0.9, max_value=None):
    cols = [
        convert_color(
            HSVColor(hue, i/(n-1), 1),
            sRGBColor
        ).get_rgb_hex()
        for i in range(n)
    ]
    m, M = min(values), max(values)
    if max_value is not None:
        M = min(M, max_value)
    width = M - m
    tick = width / n

    def selector(v):
        v = max(m, v)
        if max_value is not None:
            v = min(v, max_value)
        pos = min(int((v - m)//tick), n - 1)
        return cols[int(pos)]

    return selector


def color_selector_p(values, n, hue=0.9):
    cols = [
        convert_color(
            HSVColor(hue, i/(n-1), 1),
            sRGBColor
        ).get_rgb_hex()
        for i in range(n)
    ]
    ticks = np.percentile(values, np.linspace(0, 100, (n+1)))[1:]

    def selector(v):
        pos = 0
        while ticks[pos] < v and pos < n:
            pos += 1
        return cols[pos]

    return selector


def draw(df, id_col, val_col, extent, color_selector, figsize=(8, 8), dpi=100, width=600, alpha=0.8):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    t = tmb.tiles.build_OSM()
    plotter = tmb.Plotter(extent, t, width=width)
    plotter.plot(ax, t)

    n = len(df)
    for i in range(n):
        id = df[id_col].iloc[i]
        val = df[val_col].iloc[i]
        color = color_selector(val)
        vts = _swap(h3.h3_to_geo_boundary(id))
        xys = [tmb.project(*x) for x in vts]
        poly = plt.Polygon(xys, fc=color, alpha=alpha)
        ax.add_patch(poly)

    return fig, ax


def draw_folium(df, id_col, val_col, color_selector, zoom_start):
    n = len(df)
    lats, lngs = 0, 0
    polys = {}
    for i in range(n):
        id = df[id_col].iloc[i]
        val = df[val_col].iloc[i]
        color = color_selector(val)
        vts = h3.h3_to_geo_boundary(id)
        for v in vts:
            lats += v[0]
            lngs += v[1]
        polys[i] = folium.Polygon(
            locations=vts,
            color=color,
            fill=True,
            fill_opacity=0.8,
            weight=0
        )
        polys[i].add_child(folium.Popup(str(val)))
    lat, lng = lats/(6*n), lngs/(6*n)
    fmap = folium.Map(
        location=[lat, lng],
        zoom_start=zoom_start
    )
    for k, v in polys.items():
        v.add_to(fmap)
    return fmap
