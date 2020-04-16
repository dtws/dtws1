import tilemapbase as tmb
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def draw(df, size=0,
         figsize=(8, 8), dpi=100,
         width=600, alpha=0.8,
         axis_visible=False,
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
        ss = df.loc[size]
    elif type(size) is int or type(size) is float:
        ss = [size] * n
    elif not len(size) == n:
        raise RuntimeError(f"size is invalid: {size}")

    lats = df.loc[:, latitude].values
    lngs = df.loc[:, longitude].values

    extent = tmb.Extent.from_lonlat(
        min(lngs), max(lngs),
        min(lats), max(lats)
    )
    extent.to_aspect(1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.xaxis.set_visible(axis_visible)
    ax.yaxis.set_visible(axis_visible)

    t = tmb.tiles.build_OSM()
    plotter = tmb.Plotter(extent, t, width=width)
    plotter.plot(ax, t)

    l2 = lines.Line2D(lngs, lats)
    ax.add_line(l2)
    for i in range(n):
        x, y = tmb.project(lngs[n], lats[n])
        ax.plot(x, y, marker=".", markersize=ss[n])

    return fig, ax
