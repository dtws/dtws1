import tilemapbase as tmb
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import pandas as pd
import folium
from .maputil import copyright_osm
import numpy as np
from .colorutil import color_selector_tick
from datetime import datetime, timedelta
from geopy.distance import distance

tmb.init(create=True)


def _extend(m, M, p):
    w = M - m
    return m - p*w, M + p*w


def _expand(ex, p):
    xmin, xmax = _extend(ex.xmin, ex.xmax, p)
    ymin, ymax = _extend(ex.ymin, ex.ymax, p)
    return tmb.Extent(xmin, xmax, ymin, ymax)


def _widen(ex, p):
    return tmb.Extent(
        *_extend(ex.xmin, ex.xmax, p),
        ex.ymin, ex.ymax
    )


def _heighten(ex, p):
    return tmb.Extent(
        ex.xmin, ex.xmax,
        *_extend(ex.ymin, ex.ymax, p)
    )


def _adjust(ex, w, h):
    # Extent.xrange, yrange returns min and max in skewed way
    xmin, xmax = ex.xrange
    ymax, ymin = ex.yrange
    m = ymax - ymin
    n = xmax - xmin
    if h < w:
        p1 = w/h
        p2 = m/n
        return _widen(ex, p1 * p2)
    elif w < h:
        p1 = h/w
        p2 = n/m
        return _heighten(ex, p1 * p2)
    elif m < n:
        return _heighten(ex, n/m)
    elif n < m:
        return _widen(ex, m/n)
    else:
        return ex

    return tmb.Extent(xmin, xmax, ymin, ymax)


def draw_png(df, size=0,
             figsize=(8, 8),
             dpi=100,
             axis_visible=False,
             padding=0.03,
             adjust=True,
             latitude=None, longitude=None):
    clns = [x.lower() for x in df.columns]

    if latitude is None:
        if "latitude" in clns:
            pos = clns.index("latitude")
        elif "lat" in clns:
            pos = clns.index("lat")
        else:
            raise RuntimeError("latitude is not designated")
        latitude = df.columns[pos]
    if longitude is None:
        if "longitude" in clns:
            pos = clns.index("longitude")
        elif "lon" in clns:
            pos = clns.index("lon")
        elif "lng" in clns:
            pos = clns.index("lng")
        else:
            raise RuntimeError("longitude is not designated")
        longitude = df.columns[pos]

    n = len(df)
    if type(size) is str:
        ss = df.loc[:, size].values
    elif type(size) is int or type(size) is float:
        ss = [size] * n
    elif type(size) is pd.Series:
        ss = size.values
    elif len(size) == n:
        ss = size
    else:
        raise RuntimeError(f"size is invalid: {size}")

    lats = df.loc[:, latitude].values
    lngs = df.loc[:, longitude].values

    ex1 = tmb.Extent.from_lonlat(
        min(lngs), max(lngs),
        min(lats), max(lats)
    )
    ex2 = _expand(ex1, padding)
    extent = ex2.to_aspect(figsize[0]/figsize[1],
                           shrink=False) if adjust else ex2

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.xaxis.set_visible(axis_visible)
    ax.yaxis.set_visible(axis_visible)

    t = tmb.tiles.build_OSM()
    plotter = tmb.Plotter(extent, t,
                          width=figsize[0] * 100,
                          height=figsize[1] * 100)
    plotter.plot(ax, t)

    ps = [tmb.project(x, y) for x, y in zip(lngs, lats)]
    xs = [p[0] for p in ps]
    ys = [p[1] for p in ps]
    l2 = lines.Line2D(xs, ys)
    ax.add_line(l2)
    for i in range(n):
        x, y = ps[i]
        ax.plot(x, y, marker=".", markersize=ss[i], color="y")

    return fig, ax


def draw_folium(df,
                latitude=None, longitude=None,
                timestamp="timestamp",
                accuracy="accuracy"):
    clns = [x.lower() for x in df.columns]

    if latitude is None:
        if "latitude" in clns:
            pos = clns.index("latitude")
        elif "lat" in clns:
            pos = clns.index("lat")
        else:
            raise RuntimeError("latitude is not designated")
        latitude = df.columns[pos]
    if longitude is None:
        if "longitude" in clns:
            pos = clns.index("longitude")
        elif "lon" in clns:
            pos = clns.index("lon")
        elif "lng" in clns:
            pos = clns.index("lng")
        else:
            raise RuntimeError("longitude is not designated")
        longitude = df.columns[pos]

    n = len(df)
    ts = df.loc[:, timestamp].values
    accs = df.loc[:, accuracy].values

    lats = df.loc[:, latitude].values
    lngs = df.loc[:, longitude].values
    mlat = np.mean(lats)
    mlng = np.mean(lngs)

    fmap = folium.Map(
        location=[mlat, mlng],
        attr=copyright_osm,
        width=800, height=800
    )

    cs = color_selector_tick(np.array([5, 10, 20]) * 60, reverse=True)

    for i in range(n):
        x, y = lngs[i], lats[i]
        t = ts[i]
        acc = accs[i]
        dt = datetime.utcfromtimestamp(t) + timedelta(hours=9)
        popup = f"{dt}\nacc.={acc}"
        r1 = 10
        r2 = acc / 20
        if r1 > r2:
            folium.Circle(
                (y, x),
                color="black",
                fill=True,
                popup=popup,
                radius=r1,
                weight=0
            ).add_to(fmap)
            folium.Circle(
                (y, x),
                alpha=0.5,
                color="blue",
                fill=True,
                popup=f"acc.={acc}",
                radius=r2,
                weight=0
            ).add_to(fmap)
        else:
            folium.Circle(
                (y, x),
                alpha=0.5,
                color="blue",
                fill=True,
                popup=f"acc.={acc}",
                radius=r2,
                weight=0
            ).add_to(fmap)
            folium.Circle(
                (y, x),
                color="black",
                fill=True,
                popup=popup,
                radius=r1,
                weight=0
            ).add_to(fmap)
    for i in range(n-1):
        x, y = lngs[i], lats[i]
        nx, ny = lngs[i+1], lats[i+1]
        t, nt = ts[i], ts[i+1]
        dt = datetime.utcfromtimestamp(t) + timedelta(hours=9)
        ndt = datetime.utcfromtimestamp(nt) + timedelta(hours=9)
        col = cs(nt-t)
        ds = distance((y, x), (ny, nx)).m
        folium.PolyLine(
            locations=[(y, x), (ny, nx)],
            color=col,
            popup=f"dist.={int(ds)}m\nvelo.={int(ds/(nt-t))}m/s\ntime={nt-t}s\n"
            + f"{dt.strftime('%H:%M:%S')} - {ndt.strftime('%H:%M:%S')}"
            if nt != t else f"dist.={ds}"
        ).add_to(fmap)

    return fmap
