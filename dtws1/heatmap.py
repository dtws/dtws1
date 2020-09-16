import itertools as it
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from h3 import h3
import numpy as np
import functools as ft
import tilemapbase as tmb
import folium
from collections import namedtuple
from .colorutil import color_selector_p
import geojson as gj
import json


def _flatten(lss):
    return ft.reduce(lambda x, y: x + y, lss)


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
        for i in range(n + 1)
    ]
    lat_ticks = [
        extent_.min_lat + i * lat_width
        for i in range(n + 1)
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
        return n * xpos + ypos
    search_position_v = np.vectorize(search_position)
    positions = search_position_v(ids)

    Return = namedtuple("Return", "extents positions")
    return Return(extents, positions)

def add_color_bar(df, val_col, fig, color_selector, cbaxes_dimension=[0.55, 0.83, 0.3, 0.03],  orientation='horizontal'):
    cpool = color_selector._cols
    cmap = colors.ListedColormap(cpool,'indexed') # custom colormap https://stackoverflow.com/questions/12073306/customize-colorbar-in-matplotlib
    norm = colors.BoundaryNorm(np.percentile(df[val_col], np.linspace(0,100,len(cpool)+1)),len(cpool)+1) if isinstance(color_selector, color_selector_p) else colors.Normalize(min(df[val_col]),max(df[val_col]))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbaxes = fig.add_axes(cbaxes_dimension) # Positioning colorbar https://stackoverflow.com/questions/13310594/positioning-the-colorbar
    cbar = fig.colorbar(sm, cax=cbaxes, orientation=orientation)
    cbar.set_label(val_col)

def draw(df, id_col, val_col, extent, color_selector, figsize=(8, 8), dpi=100, width=600, alpha=0.8, axis_visible=False, **kwargs):
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

    add_color_bar(df, val_col, fig, color_selector, **kwargs)

    fig.text(0.86, 0.125, '© DATAWISE', va='bottom', ha='right')
    return fig, ax


def draw_folium(df, id_col, val_col, zoom_start=13, control_scale=True, bins=None, fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2, title=None, **kwargs):
    """
        ヒートマップを描画する

        Attributes
        ----------
        df : pandas.core.frame.DataFrame
            対象のデータフレーム
        id_col : str
            表示するデータのidのcolumn名
        val_col : str
            表示するデータのvalueのcloumn名
        zoom_start : int
            foliumのズームの初期位置
        control_scale : bool
            縮尺を表示するかどうか
        bins : list
            choroplethの境界値。valueの最大値よりbinの最大値以上の必要があるので注意
            max(df[val_col]) <= bins[-1]
        fill_color : str
            choroplethの色。以下から選択可能
                ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’, ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, ‘YlOrRd’
        fill_opacity : float
            [0,1], 透明度（色塗り）
        line_opacity : float
            [0,1], 透明度（境界）
        title : str
            legend title
    """
    df["h3_lat"],df["h3_lng"] = zip(*df[id_col].apply(lambda x:h3.h3_to_geo(x))) # h3.h3_to_geo has a proper (lat, lng) output
    location = [df["h3_lat"].mean(),df["h3_lng"].mean()] #[lat,lng]

    fmap = folium.Map(
        location=location,
        zoom_start=zoom_start,
        control_scale=control_scale
    )

    copyright = ' <a href="https://www.datawise.co.jp/">  | © DATAWISE   </a>,'
    folium.raster_layers.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        name='OpenStreetMap2',
        attr=copyright,
        overlay=True
    ).add_to(fmap)

    df[id_col] = df[id_col].astype('str')
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    df["h3_boundary"] = df[id_col].apply(lambda x:tuple((lat, lng) for lng, lat in h3.h3_to_geo_boundary(x))) # Switching lat, lng position because h3.h3_to_geo_boundary has a reverted output.

    def _process_tpl(id_col,h3_boundary):
        tpl = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": []
            }
        }
        tpl["geometry"]["coordinates"].append(h3_boundary)
        tpl["id"] = id_col   
        return tpl

    df["tpl"] = df[[id_col,"h3_boundary"]].apply(lambda x:_process_tpl(x[0],x[1]), axis=1)
    geojson["features"].extend(df["tpl"])
    geojson = json.dumps(geojson)

    if bins is None:
        max_ = df.sort_values(val_col, ascending=False).reset_index()[val_col][0]
        bins = [max_ / 5 * i for i in range(6)]
    if title is None:
        title = val_col
    folium.Choropleth(
        geojson,   # GeoJSONデータ
        name='choropleth',
        data=df,  # DataFrameまたはSeriesを指定
        columns=[id_col, val_col],  # 行政区分コードと表示データ
        key_on='feature.id',  # GeoJSONのキー（行政区分コード）
        fill_color=fill_color,  # 色パレットを指定（※）
        bins=bins,  # 境界値を指定
        fill_opacity=fill_opacity,  # 透明度（色塗り）
        line_opacity=line_opacity,  # 透明度（境界）
        legend_name=title,  # 凡例表示名
        highlight=True,
        **kwargs
    ).add_to(fmap)

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
    
    fig.text(0.86, 0.125, '© DATAWISE', va='bottom', ha='right')
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
