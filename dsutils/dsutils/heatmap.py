import itertools as it
import matplotlib.pyplot as plt
from h3 import h3
import numpy as np
import functools as ft
import tilemapbase as tmb
import folium
from collections import namedtuple
import geojson as gj


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


def extent_minmax(lat, lng):
    ret = tmb.Extent.from_lonlat(
        lng[0], lng[1],
        lat[0], lat[1]
    )
    ret.to_aspect(1)
    return ret


LatLng = namedtuple("LatLng", "lat lng")


def extent_corner(ll, ur):
    return extent_minmax(
        (ll.lat, ur.lat),
        (ll.lng, ur.lng)
    )


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


def draw(df, id_col, val_col, extent, color_selector, figsize=(8, 8), dpi=100, width=600, alpha=0.8, axis_visible=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.xaxis.set_visible(axis_visible)
    ax.yaxis.set_visible(axis_visible)

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


def draw_folium(df, id_col, val_col, color_selector, zoom_start, popups=["val", "link"], draw_line=False, label_col=None, latlng_popup=True):
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
            color="#080016",
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=0.3 if draw_line else 0
        )
        pop = []
        if "val" in popups:
            pop.append(val)
        if "link" in popups:
            pos = h3.h3_to_geo(id)
            pop.append(
                f'<a href="https://www.google.com/maps/search/{pos[0]},+{pos[1]}">link</a>')
        if "h3" in popups:
            pop.append(id)
        if "latlng" in popups:
            pos = h3.h3_to_geo(id)
            pop.append(pos)
        if label_col is not None:
            pop.append(df[label_col].iloc[i])
        if len(pop) > 0:
            polys[i].add_child(folium.Popup("\n".join(map(str, pop))))
    lat, lng = lats/(6*n), lngs/(6*n)
    fmap = folium.Map(
        location=[lat, lng],
        zoom_start=zoom_start
    )
    for k, v in polys.items():
        v.add_to(fmap)
    if latlng_popup:
        folium.LatLngPopup().add_to(fmap)
    return fmap


def extract_part(df, id_col, min_latlng, max_latlng):
    min_lat, min_lng = min_latlng
    max_lat, max_lng = max_latlng

    def selector(h3id):
        lat, lng = h3.h3_to_geo(h3id)
        return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng
    v = np.vectorize(selector)
    return df.loc[v(df[id_col])]


def drawp(df, poly_col, val_col, extent, color_selector,
          figsize=(8, 8), dpi=100, width=600, alpha=0.8):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    t = tmb.tiles.build_OSM()
    plotter = tmb.Plotter(extent, t, width=width)
    plotter.plot(ax, t)

    n = len(df)
    for i in range(n):
        val = df[val_col].iloc[i]
        color = color_selector(val)
        vts = df[poly_col].iloc[i]["coordinates"][0]
        xys = [tmb.project(*x) for x in vts]
        poly = plt.Polygon(xys, fc=color, alpha=alpha)
        ax.add_patch(poly)

    return fig, ax


def extentp(polys):
    vts = _flatten([gj.loads(p)["coordinates"][0] for p in polys])
    lngs = [x[0] for x in vts]
    lats = [x[1] for x in vts]
    extent = tmb.Extent.from_lonlat(
        min(lngs), max(lngs),
        min(lats), max(lats)
    )
    extent.to_aspect(1)
    return extent
