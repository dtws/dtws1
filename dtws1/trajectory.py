import tilemapbase as tmb
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import folium
from .maputil import copyright_osm
import numpy as np
import re

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


def _get_column(df, candidates):
    clns = [x.lower() for x in df.columns]
    candidates = [c.lower() for c in candidates if c is not None]
    pos = -1
    for c in candidates:
        if c in clns:
            pos = clns.index(c)
            break
    if pos == -1:
        raise RuntimeError(f"{candidates} is not found in {clns}")
    return df.iloc[:, pos]


def _iterable(x):
    try:
        iter(x)
    except TypeError:
        return False
    return True


def draw(df,
         latitude=None, longitude=None,
         p_size=1, p_color="#000000", p_popup=None,
         l_size=1, l_color="#000000", l_popup=None,
         output_format="png",
         **kwargs):
    lats = _get_column(df, [latitude, "latitude", "lat"]).values
    lngs = _get_column(df, [longitude, "longitude", "lon", "lng"]).values

    if type(p_size) is str:
        ps = _get_column(df, [p_size])
    elif type(p_size) is float or type(p_size) is int:
        ps = [p_size] * len(df)
    elif type(p_size) is list:
        ps = p_size
    elif _iterable(p_size):
        ps = list(p_size)
    else:
        raise ValueError(f"{p_size} is given for p_size")

    if type(p_color) is str and re.match(r"^#[0-9a-fA-F]{6}$", p_color):
        pc = [p_color] * len(df)
    elif type(p_color) is str:
        pc = _get_column(df, [p_color])
    elif type(p_color) is list:
        pc = p_color
    elif _iterable(p_color):
        pc = list(p_color)
    else:
        raise ValueError(f"{p_color} is given for p_color")

    if type(l_size) is str:
        ls = _get_column(df, [l_size])
    elif type(l_size) is float or type(l_size) is int:
        ls = [l_size] * (len(df) - 1)
    elif type(l_size) is list:
        ls = l_size
    elif _iterable(l_size):
        ls = list(l_size)
    else:
        raise ValueError(f"{l_size} is given for l_size")

    if type(l_color) is str and re.match(r"^#[0-9a-fA-F]{6}$", l_color):
        lc = [l_color] * (len(df) - 1)
    elif type(l_color) is str:
        lc = _get_column(df, [l_color])
    elif type(l_color) is list:
        lc = l_color
    elif _iterable(l_color):
        lc = list(l_color)
    else:
        raise ValueError(f"{l_color} is given for l_color")

    if output_format == "png":
        return _draw_png(
            df,
            lats, lngs,
            ps, pc,
            ls, lc,
            **kwargs
        )
    elif output_format == "html":
        return _draw_html(
            df,
            lats, lngs,
            ps, pc, p_popup,
            ls, lc, l_popup,
            ** kwargs
        )
    else:
        raise ValueError(
            f"'{output_format}' is not valid for output_format. It should be 'png' or 'html'")


def _draw_png(df,
              lats, lngs,
              p_size, p_color,
              l_size, l_color,
              figsize=(8, 8), dpi=100,
              axis_visible=False,
              padding=0.03,
              adjust=True):
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
    n = len(df)
    for i in range(n-1):
        l2 = lines.Line2D(xs[i:(i+2)], ys[i:(i+2)],
                          linewidth=l_size[i], color=l_color[i])
        ax.add_line(l2)
    for i in range(n):
        x, y = ps[i]
        ax.plot(x, y, marker=".", markersize=p_size[i], color=p_color[i])

    return fig, ax


def _draw_html(df,
               lats, lngs,
               p_size, p_color, p_popup,
               l_size, l_color, l_popup,
               zoom_start=15, width=800, height=800):
    n = len(df)
    mlat = np.mean(lats)
    mlng = np.mean(lngs)
    fmap = folium.Map(
        location=[mlat, mlng],
        attr=copyright_osm,
        width=width, height=height,
        zoom_start=zoom_start
    )

    for i in range(n):
        x, y = lngs[i], lats[i]
        folium.Circle(
            (y, x),
            color=p_color[i],
            fill=True,
            popup=p_popup[i] if p_popup is not None else None,
            radius=p_size[i],
            weight=0
        ).add_to(fmap)
    for i in range(n-1):
        x, y = lngs[i], lats[i]
        nx, ny = lngs[i+1], lats[i+1]
        col = l_color[i]
        folium.PolyLine(
            locations=[(y, x), (ny, nx)],
            color=col,
            weight=l_size[i],
            popup=l_popup[i] if l_popup is not None else None
        ).add_to(fmap)

    return fmap
